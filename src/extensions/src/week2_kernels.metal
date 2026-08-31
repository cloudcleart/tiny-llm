#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename T>
[[kernel]] void week2_rms_norm(
    device const T* x [[buffer(0)]],
    device const T* weight [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const int& rows [[buffer(3)]],
    constant const int& dim [[buffer(4)]],
    constant const float& eps [[buffer(5)]],
    threadgroup float* partial_sums [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (row >= static_cast<uint>(rows)) {
        return;
    }

    constexpr int threads_per_threadgroup = 256;
    constexpr int simdgroups_per_threadgroup = threads_per_threadgroup / 32;

    float sum_sq = 0.0f;
    for (int col = thread_index; col < dim; col += threads_per_threadgroup) {
        const float value = static_cast<float>(x[row * dim + col]);
        sum_sq += value * value;
    }

    sum_sq = simd_sum(sum_sq);
    if (lane == 0) {
        partial_sums[simdgroup] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        float threadgroup_sum =
            lane < simdgroups_per_threadgroup ? partial_sums[lane] : 0.0f;
        threadgroup_sum = simd_sum(threadgroup_sum);
        if (lane == 0) {
            partial_sums[0] = threadgroup_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inverse_rms =
        rsqrt(partial_sums[0] / static_cast<float>(dim) + eps);
    for (int col = thread_index; col < dim; col += threads_per_threadgroup) {
        out[row * dim + col] = static_cast<T>(
            static_cast<float>(x[row * dim + col]) * inverse_rms *
            static_cast<float>(weight[col])
        );
    }
}

// Week 2, Day 4:
//   week2_rms_norm
//   week2_rope
//   week2_swiglu
// Week 2, Day 5:
//   week2_decode_attention
//
// Add each [[kernel]] function when its task asks for it. The C++ starter
// wrapper and binding already exist, but remain fail-closed until then.

instantiate_kernel("week2_rms_norm_f32", week2_rms_norm, float);
instantiate_kernel("week2_rms_norm_f16", week2_rms_norm, half);
instantiate_kernel("week2_rms_norm_bf16", week2_rms_norm, bfloat16_t);
