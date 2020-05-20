#ifndef COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_COMMON_H_
#define COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_COMMON_H_

#include "larq_compute_engine/core/bconv2d_output_transform.h"
#include "ruy/kernel_common.h"

using namespace ruy;

using compute_engine::core::OutputTransform;

// Our version of `ruy::MulParams`; The original is in `ruy/mul_params.h`.
// We simply use our `OutputTransform` struct.
template <typename tAccumScalar, typename tDstScalar>
struct BinaryMulParams {
  using AccumScalar = tAccumScalar;
  using DstScalar = tDstScalar;

  OutputTransform<AccumScalar, DstScalar> output_transform;

  static constexpr LoopStructure kLoopStructure = LoopStructure::kAuto;
  static constexpr LayoutSupport kLayoutSupport = LayoutSupport::kGeneral;
  static constexpr ZeroPointSupport kZeroPointSupport =
      ZeroPointSupport::kGeneral;
  using StandardCppKernelLhsLayout = FixedKernelLayout<Order::kColMajor, 1, 1>;
  using StandardCppKernelRhsLayout = FixedKernelLayout<Order::kColMajor, 1, 1>;
  // Returns (a reasonable estimate of) the local CPU cache size.
  // See ruy::LocalDataCacheSize() which returns some coarse, sane default for
  // each CPU architecture.
  // This may be overridden, either to provide more accurate/runtime values,
  // or to test with other values to let testcases have more coverage.
  static int local_data_cache_size() { return LocalDataCacheSize(); }
  // Same as local_data_cache_size but for the total data cache size accessible
  // to each CPU core. See ruy::SharedDataCacheSize().
  static int shared_data_cache_size() { return SharedDataCacheSize(); }
};

/**
 * `BinaryKernelParamsCommon`: parameters needed regardless of `DstScalar`.
 */

template <typename TBitpacked, typename DstScalar>
struct BinaryKernelParamsCommon {
  const TBitpacked* lhs_base_ptr;
  const TBitpacked* rhs_base_ptr;
  DstScalar* dst_base_ptr;
  std::int32_t start_row;
  std::int32_t start_col;
  std::int32_t last_row;
  std::int32_t last_col;
  std::int32_t dst_rows;
  std::int32_t dst_cols;
  std::int32_t lhs_stride;
  std::int32_t rhs_stride;
  std::int32_t dst_stride;
  std::int32_t depth;
};

template <int LhsCols, int RhsCols, typename TBitpacked, typename AccumScalar,
          typename DstScalar>
inline void MakeBinaryKernelParamsCommon(
    const PackedMatrix<TBitpacked>& lhs, const PackedMatrix<TBitpacked>& rhs,
    const Matrix<DstScalar>* dst, int start_row, int start_col, int end_row,
    int end_col, BinaryKernelParamsCommon<TBitpacked, DstScalar>* params) {
  const int depth = lhs.layout.rows;
  RUY_DCHECK_EQ(start_row % LhsCols, 0);
  RUY_DCHECK_EQ(start_col % RhsCols, 0);
  RUY_DCHECK_EQ(end_row % LhsCols, 0);
  RUY_DCHECK_EQ(end_col % RhsCols, 0);

  params->lhs_base_ptr = lhs.data + start_row * lhs.layout.stride;
  params->rhs_base_ptr = rhs.data + start_col * rhs.layout.stride;
  params->dst_base_ptr =
      dst->data.get() + start_col * dst->layout.stride + start_row;

  params->start_row = start_row;
  params->start_col = start_col;
  params->last_row = end_row - LhsCols;
  params->last_col = end_col - RhsCols;
  params->lhs_stride = sizeof(TBitpacked) * lhs.layout.stride;
  params->rhs_stride = sizeof(TBitpacked) * rhs.layout.stride;
  params->dst_stride = sizeof(DstScalar) * dst->layout.stride;
  params->depth = depth;
  params->dst_rows = dst->layout.rows;
  params->dst_cols = dst->layout.cols;

  RUY_DCHECK_LT(params->last_row, params->dst_rows);
  RUY_DCHECK_LT(params->last_col, params->dst_cols);
}

// A specialisation for the case when the LHS and RHS are uint32 bitpacked but
// we're using a kernel designed for uint64 bitpacked inputs.
template <int LhsCols, int RhsCols, typename AccumScalar, typename DstScalar>
inline void MakeBinaryKernelParamsCommon(
    const PackedMatrix<std::uint32_t>& lhs,
    const PackedMatrix<std::uint32_t>& rhs, Matrix<DstScalar>* dst,
    int start_row, int start_col, int end_row, int end_col,
    BinaryKernelParamsCommon<LhsCols, RhsCols, std::uint64_t>* params) {
  const int depth = lhs.layout.rows;
  RUY_DCHECK_EQ(start_row % LhsCols, 0);
  RUY_DCHECK_EQ(start_col % RhsCols, 0);
  RUY_DCHECK_EQ(end_row % LhsCols, 0);
  RUY_DCHECK_EQ(end_col % RhsCols, 0);

  params->lhs_base_ptr = reinterpret_cast<std::uint64_t*>(
      lhs.data + start_row * lhs.layout.stride);
  params->rhs_base_ptr = reinterpret_cast<std::uint64_t*>(
      rhs.data + start_col * rhs.layout.stride);
  params->dst_base_ptr =
      dst->data.get() + start_col * dst->layout.stride + start_row;

  params->start_row = start_row;
  params->start_col = start_col;
  params->last_row = end_row - LhsCols;
  params->last_col = end_col - RhsCols;
  params->lhs_stride = sizeof(std::uint32_t) * lhs.layout.stride;
  params->rhs_stride = sizeof(std::uint32_t) * rhs.layout.stride;
  params->dst_stride = sizeof(DstScalar) * dst->layout.stride;
  // We halve the depth to pretend that the input data is uint64.
  RUY_DCHECK_EQ(depth % 2, 0);
  params->depth = depth / 2;
  params->dst_rows = dst->layout.rows;
  params->dst_cols = dst->layout.cols;

  RUY_DCHECK_LT(params->last_row, params->dst_rows);
  RUY_DCHECK_LT(params->last_col, params->dst_cols);
}

/**
 * `BinaryKernelParams`: extensions of `BinaryKernelParamsCommon` with extra
 * parameters needed for each output type.
 */

// A specialisation for `DstScalar = float`.
template <int LhsCols, int RhsCols, typename TBitpacked, float>
struct BinaryKernelParams : BinaryKernelParamsCommon<TBitpacked, float> {
  std::int32_t backtransform_add;
  std::int32_t clamp_min;
  std::int32_t clamp_max;
  // `post_mutiply` and `post_activation_bias` are currently float in order to
  // accomodate for batchnorm scales. Later this might be changed to the int8
  // system of multipliers and shifts.
  const float* post_activation_multiplier;
  const float* post_activation_bias;
  std::uint8_t flags;
  DstScalar dst_tmp_buf[LhsCols * RhsCols];
}

// A specialisation for `DstScalar = int32`, i.e. writing bitpacked output.
template <int LhsCols, int RhsCols, typename TBitpacked, std::int32>
struct BinaryKernelParams : BinaryKernelParamsCommon<TBitpacked, std::int32> {
  // For writing bitpacked output, we don't need any bias/multipliers or
  // clamping, only a threshold for writing a 1 bit versus a 0 bit.
  const std::int32* threshold;
}

template <int LhsCols, int RhsCols, typename TBitpacked, typename AccumScalar>
inline void MakeBinaryKernelParams(
    const PackedMatrix<TBitpacked>& lhs, const PackedMatrix<TBitpacked>& rhs,
    Matrix<float>* dst, const BinaryMulParams<AccumScalar, float>& spec,
    int start_row, int start_col, int end_row, int end_col,
    BinaryKernelParams<LhsCols, RhsCols, TBitpacked, float>* params) {
  MakeBinaryKernelParamsCommon(lhs, rhs, dst, start_row, start_col, end_row,
                               end_col, params);

  params->backtransform_add = spec.output_transform.backtransform_add;
  params->clamp_min = spec.output_transform.clamp_min;
  params->clamp_max = spec.output_transform.clamp_max;
  std::uint8_t flags = 0;
  params->post_activation_multiplier =
      spec.output_transform.post_activation_multiplier;
  params->post_activation_bias = spec.output_transform.post_activation_bias;
  if (params->post_activation_multiplier && params->post_activation_bias) {
    flags |= RUY_ASM_FLAG_HAS_BIAS;
  }
  params->flags = flags;
}

template <int LhsCols, int RhsCols, typename TBitpacked, typename AccumScalar>
inline void MakeBinaryKernelParams(
    const PackedMatrix<TBitpacked>& lhs, const PackedMatrix<TBitpacked>& rhs,
    Matrix<std::int32>* dst,
    const BinaryMulParams<AccumScalar, std::int32>& spec, int start_row,
    int start_col, int end_row, int end_col,
    BinaryKernelParams<LhsCols, RhsCols, TBitpacked, std::int32>* params) {
  MakeBinaryKernelParamsCommon(lhs, rhs, dst, start_row, start_col, end_row,
                               end_col, params);


}

#endif  // COMPUTE_EGNINE_TFLITE_KERNELS_BGEMM_KERNELS_COMMON_H_
