#include <cmath>

#include "larq_compute_engine/core/packbits.h"
#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "llvm/ADT/Optional.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

namespace {

// Optimize LCE operations in functions.
struct OptimizeLCE : public PassWrapper<OptimizeLCE, FunctionPass> {
 public:
  // The default value must be true so that we can run with the optimisation in
  // the file-check tests.
  explicit OptimizeLCE() : experimental_enable_bitpacked_activations_(true) {}
  explicit OptimizeLCE(bool experimental_enable_bitpacked_activations)
      : experimental_enable_bitpacked_activations_(
            experimental_enable_bitpacked_activations) {}
  void runOnFunction() override;

 private:
  bool experimental_enable_bitpacked_activations_;
};

bool IsConstantValue(Attribute values, float expected_value) {
  if (!values.isa<DenseElementsAttr>()) return false;

  for (auto value : values.cast<DenseElementsAttr>().getValues<float>()) {
    if (value != expected_value) return false;
  }
  return true;
}

bool IsConv2DFilter(Attribute filter) {
  if (!filter.isa<DenseElementsAttr>()) return false;
  if (filter.getType().cast<ShapedType>().getShape().size() != 4) return false;
  return true;
}

DenseElementsAttr Bitpack(PatternRewriter& builder, Attribute x) {
  const auto& dense_elements_iter =
      x.cast<DenseElementsAttr>().getValues<float>();

  using PackedType = std::uint32_t;
  constexpr int bitwidth = std::numeric_limits<PackedType>::digits;

  auto shape = x.getType().cast<ShapedType>().getShape();
  int num_rows = shape[0] * shape[1] * shape[2];
  int unpacked_channels = shape[3];
  int packed_channels = (unpacked_channels + bitwidth - 1) / bitwidth;

  std::vector<PackedType> new_values(num_rows * packed_channels);

  const float* in_ptr = &(*dense_elements_iter.begin());
  using namespace compute_engine::core;
  packbits_matrix<BitpackOrder::Canonical>(in_ptr, num_rows, unpacked_channels,
                                           new_values.data());

  RankedTensorType out_tensor_type =
      RankedTensorType::get({shape[0], shape[1], shape[2], packed_channels},
                            builder.getIntegerType(bitwidth));

  return DenseElementsAttr::get<PackedType>(out_tensor_type, new_values);
}

#include "larq_compute_engine/mlir/transforms/generated_optimize.inc"

/**
 * =================================================
 * Computing thresholds for writing bitpacked output
 * =================================================
 *
 * Consider a single output element of a binary convolution, y. We have that:
 *
 *                       y = clamp(a - 2x, c, C) * μ + β
 *
 * where x is the xor-popcount accumulator,
 *       a is the backtransform addition,
 *       c is the clamp-min,
 *       C is the clamp-max,
 *       μ is the channel multiplier,
 *       β is the channel bias.
 *
 * We want to write a 0-bit if and only if y >= 0. To do this, we want to find
 * a threshold τ so that we can decide whether or not to write a 0-bit or a
 * 1-bit with a single comparison of x with τ.
 *
 * Throughout, the operator sign() is treated as mapping 0 |-> 1. This
 * behaviour is isomorphic to taking the sign bit of a standard signed int32
 * number.
 *
 * ----------------
 * The general case
 * ----------------
 *
 * First, suppose that μ != 0 and that that clamping range crosses 0, i.e.
 * sign((c - 2x) * μ + β) != sign((c - 2x) * μ + β), so that the effect of
 * clamping can be ignored. If either of these conditions does not hold, then
 * the sign of y is constant and does not depend on x; we will consider these
 * special cases later.
 *
 * If μ > 0:
 *                  y >= 0 <-> (a - 2x) * μ + β >= 0
 *                         <-> x <= 0.5 * (β / μ + a)
 *                         <-> x <= ⌊0.5 * (β / μ + a)⌋   (as x is an integer)
 *
 *     We can therefore use a threshold κ := ⌊0.5 * (β / μ + a)⌋.
 *
 *     x is the xor-popcount accumulator and so is non-negative by definition.
 *     As a result, if κ < 0 then x <= κ can never hold, so we can treat this
 *     as a special case later on and assume that when μ > 0, κ >= 0.
 *
 * If μ < 0:
 *                  y >= 0 <-> (a - 2x) * μ + β >= 0
 *                         <-> x >= 0.5 * (β / μ + a)
 *                         <-> x >= ⌈0.5 * (β / μ + a)⌉   (as x is an integer)
 *
 *     We can therefore use a threshold κ := ⌈0.5 * (β / μ + a)⌉.
 *
 *     Again, x is non-negative by definition, so if κ <= 0 then x >= κ will
 *     always be true, so we can treat this as a special case later on and
 *     assume that when μ < 0, κ > 0.
 *
 * The only remaining problem is that the comparison we make with κ depends on
 * the sign of μ. For an efficient implementation, we don't want to have to
 * sometimes use <= and sometimes use >=, and we don't want to depend on more
 * data than just a single threshold. Luckily, we can avoid this. Notice that,
 * whether μ > 0 or μ < 0, we always have:
 *
 *                    y >= 0 <-> sign(μ) * x <= sign(μ) * κ
 *
 * Define τ := sign(μ) * κ.
 *
 * As we assumed that (μ > 0 -> κ >= 0) and (μ < 0 -> κ > 0), it follows that
 * sign(τ) = sign(μ) * sign(κ) = sign(μ).
 *
 * We can therefore conclude that:
 *
 *                         y >= 0 <-> sign(τ) * x <= τ
 *
 * This makes it possible to decide whether to write a 0-bit or a 1-bit by
 * doing one integer multiply and one integer comparison.
 *
 * -------------
 * Special cases
 * -------------
 *
 * All of the remaining special cases (the transformed clamping range not
 * crossing 0; μ = 0; μ > 0 ∧ κ < 0; μ < 0 ∧ κ <= 0) cause the sign of y to be
 * constant and no longer depend on x. This means that we want to always write
 * a 1-bit or always write a 0-bit.
 *
 * To always write a 0-bit (if and only if y >= 0), we can set τ := ∞ (the
 * maximum signed representable number). Then, sign(τ) * x <= τ will always be
 * true.
 *
 * To always write a 1-bit (if and only if y < 0), we can set τ := -∞ (the
 * minimum signed representable number). Then, sign(τ) * x <= τ will always be
 * false.
 */
std::vector<std::int32_t> ComputeWriteBitpackedOutputThresholds(
    const float backtransform_add, const float clamp_min, const float clamp_max,
    const DenseElementsAttr& multipliers, const DenseElementsAttr& biases) {
  constexpr std::int32_t neg_inf = std::numeric_limits<std::int32_t>::min();
  constexpr std::int32_t pos_inf = std::numeric_limits<std::int32_t>::max();

  // Iterate throught the multiplier/bias pairs and compute the thresholds.
  std::vector<std::int32_t> thresholds;
  for (auto mult_bias_pair :
       llvm::zip(multipliers.getValues<float>(), biases.getValues<float>())) {
    const float mult = std::get<0>(mult_bias_pair);
    const float bias = std::get<1>(mult_bias_pair);

    const float output_range_start = (clamp_min * mult + bias);
    const float output_range_end = (clamp_max * mult + bias);

    // First, check for some special cases (detailed in the comment above).
    if (output_range_start < 0 && output_range_end < 0) {
      thresholds.push_back(neg_inf);  // We need to always write a 1-bit.
      continue;
    }
    if (output_range_start >= 0 && output_range_end >= 0) {
      thresholds.push_back(pos_inf);  // We need to always write a 0-bit.
      continue;
    }
    if (mult == 0.0f) {
      if (bias < 0) {
        thresholds.push_back(neg_inf);  // We need to always write a 1-bit.
      } else {
        thresholds.push_back(pos_inf);  // We need to always write a 0-bit.
      }
      continue;
    }

    // The general case.
    if (mult > 0.0f) {
      const std::int32_t k =
          std::floor(0.5 * (mult / bias + backtransform_add));
      if (k < 0) {
        thresholds.push_back(neg_inf);  // We need to always write a 1-bit.
      } else {
        thresholds.push_back(k);
      }
    } else /* if (mult < 0.0f) */ {
      const std::int32_t k = std::ceil(0.5 * (mult / bias + backtransform_add));
      if (k <= 0) {
        thresholds.push_back(pos_inf);  // We need to always write a 0-bit.
        continue;
      } else {
        thresholds.push_back(-1 * k);
      }
    }
  }

  return thresholds;
}

llvm::Optional<RankedTensorType> maybeGetBitpackedType(
    PatternRewriter& rewriter, ShapedType existing_type) {
  if (existing_type.getElementType().isInteger(32)) return llvm::None;

  const auto existing_shape = existing_type.getShape();
  if (existing_shape.size() != 4) return llvm::None;

  const auto channels = existing_shape[3];
  const auto packed_channels = (channels + 32 - 1) / 32;
  return RankedTensorType::get({existing_shape[0], existing_shape[1],
                                existing_shape[2], packed_channels},
                               rewriter.getIntegerType(32));
}

template <typename BinaryOp>
struct SetBitpackedActivations : public OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOp outer_binary_op,
                                PatternRewriter& rewriter) const override {
    Operation* input_op = outer_binary_op.input().getDefiningOp();

    // If the input op has more than one use, we can't apply an optimisation.
    if (!input_op || !input_op->hasOneUse()) return failure();

    // Try and match `input_op` to a binary convolution.
    auto inner_bconv_op = dyn_cast_or_null<TF::LceBconv2dOp>(input_op);
    if (inner_bconv_op) {
      if (inner_bconv_op.padding() == "SAME" &&
          inner_bconv_op.pad_values() != 1) {
        return failure();
      }

      if (auto maybe_bitpacked_type = maybeGetBitpackedType(
              rewriter, inner_bconv_op.getType().cast<ShapedType>())) {
        // As the inner bconv op will be writing bitpacked output, we need to
        // compute the thresholds for writing 1-bit or a 0-bit.

        // Compute the backtransform add and the clamp min/max.
        const auto filter_shape =
            inner_bconv_op.input().getType().cast<ShapedType>().getShape();
        const float backtransform_add = static_cast<float>(
            filter_shape[1] * filter_shape[2] * filter_shape[3]);

        const auto clamp_min_max =
            llvm::StringSwitch<std::pair<std::int32_t, std::int32_t>>(
                inner_bconv_op.activation())
                .Case("RELU", {0, std::numeric_limits<std::int32_t>::max()})
                .Case("RELU_N1_TO_1", {-1, 1})
                .Case("RELU6", {0, 6})
                .Default({std::numeric_limits<std::int32_t>::min(),
                          std::numeric_limits<std::int32_t>::max()});
        const float clamp_min = static_cast<float>(std::get<0>(clamp_min_max));
        const float clamp_max = static_cast<float>(std::get<1>(clamp_min_max));

        // Extract the post_activation multiplier and bias values.
        const auto multipliers = DenseElementsAttr::get<float>(
            inner_bconv_op.post_activation_multiplier()
                .getType()
                .cast<ShapedType>(),
            inner_bconv_op.post_activation_multiplier());
        const auto biases = DenseElementsAttr::get<float>(
            inner_bconv_op.post_activation_bias().getType().cast<ShapedType>(),
            inner_bconv_op.post_activation_bias());

        std::vector<std::int32_t> thresholds =
            ComputeWriteBitpackedOutputThresholds(
                backtransform_add, clamp_min, clamp_max, multipliers, biases);
        RankedTensorType thresholds_type = RankedTensorType::get(
            multipliers.getType().cast<ShapedType>().getShape(),
            rewriter.getIntegerType(32));

        Value thresholds_input = rewriter.create<ConstantOp>(
            inner_bconv_op.getLoc(),
            DenseElementsAttr::get<std::int32_t>(thresholds_type, thresholds));

        // We need an empty input with which to overwrite the
        // `post_activation_multiply` and `post_activation_bias` (which are no
        // longer needed, having computed the thresholds).
        Value empty_input = rewriter.create<ConstantOp>(inner_bconv_op.getLoc(),
                                                        rewriter.getNoneType(),
                                                        rewriter.getUnitAttr());

        ValueRange new_inner_bconv_op_operands(
            {inner_bconv_op.input(), inner_bconv_op.filter(), empty_input,
             empty_input, thresholds_input});

        rewriter.replaceOpWithNewOp<TF::LceBconv2dOp>(
            inner_bconv_op, *maybe_bitpacked_type, new_inner_bconv_op_operands,
            inner_bconv_op.getAttrs());

        return success();
      }
    }

    // Otherwise, try and match `input_op` to a maxpool.
    auto maxpool_op = dyn_cast_or_null<TFL::MaxPool2DOp>(input_op);
    if (maxpool_op) {
      if (maxpool_op.fused_activation_function() != "NONE") {
        return failure();
      }

      if (auto bitpacked_type = maybeGetBitpackedType(
              rewriter, maxpool_op.getType().cast<ShapedType>())) {
        rewriter.replaceOpWithNewOp<TF::LceBMaxPool2dOp>(
            maxpool_op, *bitpacked_type, maxpool_op.input(),
            rewriter.getStringAttr(maxpool_op.padding()),
            rewriter.getIntegerAttr(rewriter.getIntegerType(32),
                                    maxpool_op.stride_h()),
            rewriter.getIntegerAttr(rewriter.getIntegerType(32),
                                    maxpool_op.stride_w()),
            rewriter.getIntegerAttr(rewriter.getIntegerType(32),
                                    maxpool_op.filter_width()),
            rewriter.getIntegerAttr(rewriter.getIntegerType(32),
                                    maxpool_op.filter_height()));
        return success();
      }
    }

    return failure();
  };
};

void OptimizeLCE::runOnFunction() {
  OwningRewritePatternList patterns;
  auto* ctx = &getContext();
  auto func = getFunction();

  TFL::populateWithGenerated(ctx, &patterns);
  if (experimental_enable_bitpacked_activations_) {
    patterns.insert<SetBitpackedActivations<TF::LceBconv2dOp>>(ctx);
    patterns.insert<SetBitpackedActivations<TF::LceBMaxPool2dOp>>(ctx);
  }
  applyPatternsAndFoldGreedily(func, patterns);
}

}  // namespace

// Creates an instance of the TensorFlow dialect OptimizeLCE pass.
std::unique_ptr<OperationPass<FuncOp>> CreateOptimizeLCEPass(
    bool experimental_enable_bitpacked_activations) {
  return std::make_unique<OptimizeLCE>(
      experimental_enable_bitpacked_activations);
}

static PassRegistration<OptimizeLCE> pass(
    "tfl-optimize-lce", "Optimize within the TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
