#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Float32 Flash Attention kernel for head_dim <= 128. The launch grid is
// (N, Tr, 1), so each threadgroup owns one flattened head and one Q tile.
[[kernel]] void flash_attention_f32_e128(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device const float *mask [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant const int *mask_shape [[buffer(5)]],
    constant const int64_t *mask_strides [[buffer(6)]],
    device const int &is_causal [[buffer(7)]],
    device const int &N [[buffer(8)]],
    device const int &L [[buffer(9)]],
    device const int &S [[buffer(10)]],
    device const int &E [[buffer(11)]],
    device const int &num_kv_heads [[buffer(12)]],
    device const int &num_heads [[buffer(13)]],
    device const float &scale [[buffer(14)]],
    device const int &Br [[buffer(15)]],
    device const int &Bc [[buffer(16)]],
    [[maybe_unused]] device const int &Tr [[buffer(17)]],
    device const int &Tc [[buffer(18)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    const int n = group_id.x;
    const int tile_q = group_id.y;
    const int row = simd_gid;
    const int col = simd_lid;
    const int q_row = tile_q * Br + row;
    const bool valid_q = n < N && row < Br && q_row < L;

    // GQA maps multiple flattened query heads onto one flattened KV head.
    const int q_kv_ratio = num_heads / num_kv_heads;

    device const float *q_head = q + n * L * E + tile_q * Br * E;
    device const float *k_head = k + (n / q_kv_ratio) * S * E;
    device const float *v_head = v + (n / q_kv_ratio) * S * E;

    threadgroup float q_local[32][128];
    threadgroup float out_acc[32 * 128];

    // One SIMD group handles one Q row. Lane 0 loads that row into fast
    // threadgroup memory, then all lanes reuse it across every K/V tile.
    if (simd_lid == 0) {
        for (int d = 0; d < E; ++d) {
            q_local[row][d] = valid_q ? q_head[row * E + d] : 0.0f;
            out_acc[row * E + d] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    const int causal_offset = S - L;

    // Stream K/V tiles. Only a 32x32 score tile exists at any point; the full
    // LxS attention matrix is never materialized.
    for (int tile_k = 0; tile_k < Tc; ++tile_k) {
        const int k_col = tile_k * Bc + col;
        const bool valid_k = col < Bc && k_col < S;

        // If a causal K tile is completely invisible for this Q tile, skip all
        // dot products and V reads. S-L keeps decode-with-cache aligned.
        if (is_causal) {
            const int tile_row_max = min((tile_q + 1) * Br - 1, L - 1);
            const int tile_col_min = tile_k * Bc;
            if (tile_col_min > tile_row_max + causal_offset) {
                continue;
            }
        }

        float score = -INFINITY;
        if (valid_q && valid_k) {
            // Each lane computes one score element S[row, col].
            score = 0.0f;
            for (int d = 0; d < E; ++d) {
                score += q_local[row][d] * k_head[k_col * E + d];
            }
            score *= scale;

            const int tile_row_min = tile_q * Br;
            const int tile_col_max = min((tile_k + 1) * Bc - 1, S - 1);
            const bool causal_tile_all_valid = is_causal && tile_col_max <= tile_row_min + causal_offset;
            if (!causal_tile_all_valid) {
                // Mask may be a broadcasted view, so use MLX strides rather
                // than assuming row-contiguous mask storage.
                const int64_t mask_idx = elem_to_loc(n * L * S + q_row * S + k_col, mask_shape, mask_strides, 3);
                score += mask[mask_idx];
            }
        }

        // SIMD reductions operate across the 32 lanes for the current Q row.
        const float tile_row_max = simd_max(score);
        const float new_row_max = max(row_max, tile_row_max);
        const float old_scale = exp(row_max - new_row_max);
        row_max = new_row_max;

        const float p = valid_q && valid_k ? exp(score - row_max) : 0.0f;
        row_sum = old_scale * row_sum + simd_sum(p);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        // For every output column, lanes reduce over the 32 K positions:
        // sum_b P[row, b] * V[b, d]. Lane 0 owns the row's accumulator.
        for (int d = 0; d < E; ++d) {
            const float pv = valid_q && valid_k ? p * v_head[k_col * E + d] : 0.0f;
            const float tile_out = simd_sum(pv);
            if (simd_lid == 0 && valid_q) {
                out_acc[row * E + d] = old_scale * out_acc[row * E + d] + tile_out;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Convert the online-softmax numerator into the final attention output.
    if (simd_lid == 0 && valid_q) {
        for (int d = 0; d < E; ++d) {
            out[n * L * E + q_row * E + d] = out_acc[row * E + d] / row_sum;
        }
    }
}
