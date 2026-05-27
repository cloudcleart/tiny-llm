#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

template <typename T>
[[kernel]] void quantized_matmul_w4a16_g128(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant const int& M [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    constant const int& K [[buffer(7)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]]) {
  constexpr int bits = 4;
  constexpr int group_size = 128;
  constexpr int values_per_pack = 32 / bits;
  constexpr uint32_t item_mask = (1u << bits) - 1u;

  // One thread owns one output element C[i, k].
  const int i = group_id.x * threads_per_threadgroup.x + thread_id.x;
  const int k = group_id.y * threads_per_threadgroup.y + thread_id.y;
  if (i >= M || k >= K) {
    return;
  }

  const int groups_per_row = N / group_size;
  int a_loc = i * N;
  int b_loc = k * (N / values_per_pack);
  int scale_bias_loc = k * groups_per_row;
  float sum = 0.0f;

  // B is stored as packed int4, with each group sharing one scale/bias pair.
  for (int group_idx = 0; group_idx < groups_per_row; ++group_idx) {
    const float scale = static_cast<float>(scales[scale_bias_loc]);
    const float bias = static_cast<float>(biases[scale_bias_loc]);

    for (int item_idx = 0; item_idx < group_size;
         item_idx += values_per_pack) {
      const uint32_t packed = b[b_loc++];

      // Unroll the 8 nibbles in a uint32. Low bits are the first weight.
      sum +=
          (static_cast<float>((packed >> 0) & item_mask) * scale + bias) *
          static_cast<float>(a[a_loc]);
      sum +=
          (static_cast<float>((packed >> 4) & item_mask) * scale + bias) *
          static_cast<float>(a[a_loc + 1]);
      sum +=
          (static_cast<float>((packed >> 8) & item_mask) * scale + bias) *
          static_cast<float>(a[a_loc + 2]);
      sum +=
          (static_cast<float>((packed >> 12) & item_mask) * scale + bias) *
          static_cast<float>(a[a_loc + 3]);
      sum +=
          (static_cast<float>((packed >> 16) & item_mask) * scale + bias) *
          static_cast<float>(a[a_loc + 4]);
      sum +=
          (static_cast<float>((packed >> 20) & item_mask) * scale + bias) *
          static_cast<float>(a[a_loc + 5]);
      sum +=
          (static_cast<float>((packed >> 24) & item_mask) * scale + bias) *
          static_cast<float>(a[a_loc + 6]);
      sum +=
          (static_cast<float>((packed >> 28) & item_mask) * scale + bias) *
          static_cast<float>(a[a_loc + 7]);

      a_loc += values_per_pack;
    }

    ++scale_bias_loc;
  }

  out[i * K + k] = static_cast<T>(sum);
}

instantiate_kernel(
    "quantized_matmul_w4a16_g128_bf16",
    quantized_matmul_w4a16_g128,
    bfloat16_t);
