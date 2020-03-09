#include "larq_compute_engine/mlir/ir/lce_ops.h"
#include "larq_compute_engine/mlir/transforms/utils.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

namespace {

// Optimize LCE operations in functions.
struct OptimizeLCE : public FunctionPass<OptimizeLCE> {
  void runOnFunction() override;
};

#include "larq_compute_engine/mlir/transforms/generated_optimize.inc"


struct SetBconvReadWriteBitpacked : public OpRewritePattern<TF::LqceBconv2d64Op> {
  using OpRewritePattern<TF::LqceBconv2d64Op>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::LqceBconv2d64Op bconv_op,
                                     PatternRewriter &rewriter) const override {
    Value* bconv_input = bconv_op.input();
    if (!bconv_input->hasOneUse()) return matchFailure();

    auto bconv_input_op = dyn_cast_or_null<TF::LqceBconv2d64Op>(bconv_input->getDefiningOp());
    if (!bconv_input_op) return matchFailure();

    if (bconv_input_op.write_bitpacked_output() && bconv_op.read_bitpacked_input()) {
      return matchFailure();
    }

    rewriter.replaceOpWithNewOp<TF::LqceBconv2d64Op>(
      bconv_input_op, bconv_input_op.getType(),
      bconv_input_op.input(),
      bconv_input_op.filter(),
      bconv_input_op.post_activation_multiplier(),
      bconv_input_op.post_activation_bias(),
      bconv_input_op.strides(),
      rewriter.getStringAttr(bconv_input_op.padding()),
      rewriter.getIntegerAttr(rewriter.getIntegerType(32), bconv_input_op.pad_values()),
      rewriter.getStringAttr(bconv_input_op.data_format()),
      bconv_input_op.dilations(),
      rewriter.getStringAttr(bconv_input_op.filter_format()),
      rewriter.getBoolAttr(bconv_input_op.read_bitpacked_input()),
      /*write_bitpacked_output=*/rewriter.getBoolAttr(true)
    );

    rewriter.replaceOpWithNewOp<TF::LqceBconv2d64Op>(
      bconv_op, bconv_op.getType(),
      bconv_op.input(),
      bconv_op.filter(),
      bconv_op.post_activation_multiplier(),
      bconv_op.post_activation_bias(),
      bconv_op.strides(),
      rewriter.getStringAttr(bconv_op.padding()),
      rewriter.getIntegerAttr(rewriter.getIntegerType(32), bconv_op.pad_values()),
      rewriter.getStringAttr(bconv_op.data_format()),
      bconv_op.dilations(),
      rewriter.getStringAttr(bconv_op.filter_format()),
      /*read_bitpacked_input=*/rewriter.getBoolAttr(true),
      rewriter.getBoolAttr(bconv_op.write_bitpacked_output())
    );

    return matchSuccess();
  };
};


void OptimizeLCE::runOnFunction() {
  OwningRewritePatternList patterns;
  auto* ctx = &getContext();
  auto func = getFunction();

  TFL::populateWithGenerated(ctx, &patterns);
  patterns.insert<SetBconvReadWriteBitpacked>(ctx);
  // Cleanup dead ops manually. LCE ops are not registered to the TF dialect so
  // op->hasNoSideEffect() will return false. Therefor applyPatternsGreedily
  // won't automatically remove the dead nodes. See
  // https://github.com/llvm/llvm-project/blob/master/mlir/include/mlir/IR/Operation.h#L457-L462
  patterns.insert<mlir::CleanupDeadOps<TF::LqceBconv2d64Op>>(ctx);
  applyPatternsGreedily(func, patterns);
}

}  // namespace

// Creates an instance of the TensorFlow dialect OptimizeLCE pass.
std::unique_ptr<OpPassBase<FuncOp>> CreateOptimizeLCEPass() {
  return std::make_unique<OptimizeLCE>();
}

static PassRegistration<OptimizeLCE> pass(
    "tfl-optimize-lce", "Optimize within the TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
