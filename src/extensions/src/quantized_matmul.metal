#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Starter interface map. Implement the named kernels at these checkpoints;
// their argument lists are defined by the matching C++ encoder you complete.
//
// Week 2, Day 3:
//   quantized_matmul_vanilla_w4a16_g128
//   quantized_matvec_x4_fast_w4a16_g128
// Week 2, Day 6:
//   quantized_matmul_simdgroup_w4a16_g128
// Week 2, Day 7:
//   quantized_matmul_simdgroup_splitk_w4a16_g128
//   quantized_matmul_splitk_reduce
// Week 3, Day 4:
//   quantized_embedding_w4a16_g128
//
// The x2/x8 tuning variants in the reference extension are deliberately not
// starter interfaces. Add an experimental variant only while running the
// optional scheduling comparison, then keep the selected course path.

// Correctness-first matrix schedule: one thread owns one C[i, k] output and
// walks the complete reduction dimension. It deliberately shares no loads so
// its indexing follows the Python dequantization equation directly.
template <typename T>
[[kernel]] void quantized_matmul_vanilla_w4a16_g128(
    device const T *scales [[buffer(0)]], device const T *biases [[buffer(1)]], device const T *a [[buffer(2)]],
    device const uint32_t *b [[buffer(3)]], device T *out [[buffer(4)]], device const int &M [[buffer(5)]],
    device const int &N [[buffer(6)]], device const int &K [[buffer(7)]],
    uint3 group_id [[threadgroup_position_in_grid]], uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]]) {
    constexpr int bits = 4;
    constexpr int group_size = 128;
    constexpr int values_per_pack = 32 / bits;
    constexpr uint32_t mask = (1 << bits) - 1;

    const int row = group_id.x * threads_per_threadgroup.x + thread_id.x;
    const int column = group_id.y * threads_per_threadgroup.y + thread_id.y;
    if (row >= M || column >= K) {
        return;
    }

    const int groups_per_row = N / group_size;
    int activation_offset = row * N;
    int weight_offset = column * (N / values_per_pack);
    int parameter_offset = column * groups_per_row;
    float sum = 0.0f;

    for (int group = 0; group < groups_per_row; ++group) {
        const float scale = static_cast<float>(scales[parameter_offset]);
        const float bias = static_cast<float>(biases[parameter_offset]);

        for (int value = 0; value < group_size; value += values_per_pack) {
            const uint32_t packed = b[weight_offset++];
#pragma clang loop unroll(full)
            for (int lane = 0; lane < values_per_pack; ++lane) {
                const float code = static_cast<float>((packed >> (lane * bits)) & mask);
                const float weight = code * scale + bias;
                sum += static_cast<float>(a[activation_offset++]) * weight;
            }
        }
        ++parameter_offset;
    }

    out[row * K + column] = static_cast<T>(sum);
}

// Decode schedule: each SIMD group reduces N for four adjacent output
// columns. Two adjacent uint32 packs give each lane 16 activations that are
// reused across all four columns; simd_sum combines the 32 lane-local sums.
template <typename T>
[[kernel]] void quantized_matvec_x4_fast_w4a16_g128(
    device const T *scales [[buffer(0)]], device const T *biases [[buffer(1)]], device const T *a [[buffer(2)]],
    device const uint32_t *b [[buffer(3)]], device T *out [[buffer(4)]], device const int &M [[buffer(5)]],
    device const int &N [[buffer(6)]], device const int &K [[buffer(7)]],
    uint output_tile [[threadgroup_position_in_grid]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int group_size = 128;
    constexpr int values_per_pack = 8;
    constexpr int packs_per_lane = 2;
    constexpr int values_per_lane = values_per_pack * packs_per_lane;
    constexpr int outputs_per_simdgroup = 4;
    constexpr int simdgroups_per_threadgroup = 2;
    constexpr int outputs_per_threadgroup = outputs_per_simdgroup * simdgroups_per_threadgroup;

    const int column_tiles = (K + outputs_per_threadgroup - 1) / outputs_per_threadgroup;
    const int row = output_tile / column_tiles;
    const int column_base =
        (output_tile - row * column_tiles) * outputs_per_threadgroup + simdgroup * outputs_per_simdgroup;
    if (row >= M || column_base >= K) {
        return;
    }

    const int packed_columns = N / values_per_pack;
    const int groups_per_row = N / group_size;
    const int activation_base = row * N;
    float sums[outputs_per_simdgroup] = {0.0f};

    for (int packed_column = lane * packs_per_lane; packed_column < packed_columns;
         packed_column += 32 * packs_per_lane) {
        const int quantization_group = packed_column / (group_size / values_per_pack);
        float scaled_activations[values_per_lane];
        float activation_sum = 0.0f;

#pragma clang loop unroll(full)
        for (int pack = 0; pack < packs_per_lane; ++pack) {
            const int activation_offset = activation_base + (packed_column + pack) * values_per_pack;
#pragma clang loop unroll(full)
            for (int value = 0; value < values_per_pack; ++value) {
                const int local = pack * values_per_pack + value;
                const float activation = static_cast<float>(a[activation_offset + value]);
                activation_sum += activation;

                // Four W4 codes occupy one uint16. Pre-scaling converts the
                // masked nibble values into their logical 0..15 codes without
                // a separate shift in every output-column accumulator.
                scaled_activations[local] = activation / static_cast<float>(1 << ((value & 3) * 4));
            }
        }

#pragma clang loop unroll(full)
        for (int output_index = 0; output_index < outputs_per_simdgroup; ++output_index) {
            const int column = column_base + output_index;
            if (column >= K) {
                continue;
            }

            const int parameter_index = column * groups_per_row + quantization_group;
            const float scale = static_cast<float>(scales[parameter_index]);
            const float bias = static_cast<float>(biases[parameter_index]);
            const device uint16_t *packed =
                reinterpret_cast<const device uint16_t *>(b + column * packed_columns + packed_column);

            float quantized_dot = 0.0f;
#pragma clang loop unroll(full)
            for (int chunk = 0; chunk < values_per_lane / 4; ++chunk) {
                const uint16_t weights = packed[chunk];
                const int local = chunk * 4;
                quantized_dot += scaled_activations[local] * (weights & 0x000f) +
                                 scaled_activations[local + 1] * (weights & 0x00f0) +
                                 scaled_activations[local + 2] * (weights & 0x0f00) +
                                 scaled_activations[local + 3] * (weights & 0xf000);
            }
            sums[output_index] += scale * quantized_dot + bias * activation_sum;
        }
    }

#pragma clang loop unroll(full)
    for (int output_index = 0; output_index < outputs_per_simdgroup; ++output_index) {
        sums[output_index] = simd_sum(sums[output_index]);
    }

    if (lane == 0) {
#pragma clang loop unroll(full)
        for (int output_index = 0; output_index < outputs_per_simdgroup; ++output_index) {
            const int column = column_base + output_index;
            if (column < K) {
                out[row * K + column] = static_cast<T>(sums[output_index]);
            }
        }
    }
}

instantiate_kernel("quantized_matmul_vanilla_w4a16_g128_f16", quantized_matmul_vanilla_w4a16_g128, half);
instantiate_kernel("quantized_matmul_vanilla_w4a16_g128_bf16", quantized_matmul_vanilla_w4a16_g128, bfloat16_t);
instantiate_kernel("quantized_matvec_x4_fast_w4a16_g128_f16", quantized_matvec_x4_fast_w4a16_g128, half);
instantiate_kernel("quantized_matvec_x4_fast_w4a16_g128_bf16", quantized_matvec_x4_fast_w4a16_g128, bfloat16_t);
