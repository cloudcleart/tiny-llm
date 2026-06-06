#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext {

// Build the lazy MLX primitive. Python has already flattened batch/head dims:
// Q/M/O are (B * H_q, L, D/S), while K/V are (B * H_kv, S, D).
mx::array flash_attention(const mx::array &q, const mx::array &k, const mx::array &v, const mx::array &mask,
                          const float scale, const bool is_causal, const int num_kv_heads, const int num_heads,
                          mx::StreamOrDevice s) {
    if (q.dtype() != mx::float32 || k.dtype() != mx::float32 || v.dtype() != mx::float32 ||
        mask.dtype() != mx::float32) {
        throw std::runtime_error("flash_attention: all input arrays must be float32");
    }
    if (q.ndim() != 3 || k.ndim() != 3 || v.ndim() != 3 || mask.ndim() != 3) {
        throw std::runtime_error("flash_attention: all input arrays must be 3D");
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention: num_heads must be divisible by num_kv_heads");
    }

    const int64_t N = q.shape()[0];
    const int64_t N_kv = k.shape()[0];
    const int64_t L = q.shape()[1];
    const int64_t S = k.shape()[1];
    const int64_t D = q.shape()[2];

    if (N % num_heads != 0) {
        throw std::runtime_error("flash_attention: q.shape[0] must be divisible by num_heads");
    }
    if (N_kv % num_kv_heads != 0 || v.shape()[0] != N_kv) {
        throw std::runtime_error("flash_attention: k/v batch-head dimension must be divisible by num_kv_heads");
    }
    if (N / num_heads != N_kv / num_kv_heads) {
        throw std::runtime_error("flash_attention: query and key/value batch dimensions must match");
    }
    if (v.shape()[1] != S) {
        throw std::runtime_error("flash_attention: key and value sequence dimensions must match");
    }
    if (k.shape()[2] != D || v.shape()[2] != D) {
        throw std::runtime_error("flash_attention: q, k, and v head dimensions must match");
    }
    if (mask.shape()[0] != N || mask.shape()[1] != L || mask.shape()[2] != S) {
        throw std::runtime_error("flash_attention: mask must have shape (q.shape[0], q.shape[1], k.shape[1])");
    }

    // The result is materialized later by eval_cpu/eval_gpu when MLX evaluates
    // this array; keep the kernel parameters on the primitive object.
    return mx::array(q.shape(), mx::float32,
                     std::make_shared<FlashAttention>(to_stream(s), scale, is_causal, num_kv_heads, num_heads),
                     {q, k, v, mask});
}

void FlashAttention::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &q = inputs[0];
    const auto &k = inputs[1];
    const auto &v = inputs[2];
    const auto &mask = inputs[3];
    auto &out = outputs[0];

    if (out.dtype() != mx::float32) {
        throw std::runtime_error("flash_attention: output dtype must be float32");
    }
    if (!q.flags().row_contiguous || !k.flags().row_contiguous || !v.flags().row_contiguous ||
        !mask.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: q, k, v, and mask must be row contiguous");
    }

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto &encoder = mx::cpu::get_command_encoder(stream());
    encoder.set_input_array(q);
    encoder.set_input_array(k);
    encoder.set_input_array(v);
    encoder.set_input_array(mask);
    encoder.set_output_array(out);

    // CPU reference kernel: one flattened query head at a time, with 32x32
    // score tiles and online softmax state per query row.
    encoder.dispatch([q = mx::array::unsafe_weak_copy(q), k = mx::array::unsafe_weak_copy(k),
                      v = mx::array::unsafe_weak_copy(v), mask = mx::array::unsafe_weak_copy(mask),
                      out_ptr = out.data<float>(), out_shape = out.shape(), out_strides = out.strides(), scale = scale_,
                      is_causal = is_causal_, num_heads = num_heads_, num_kv_heads = num_kv_heads_]() {
        const int64_t N = q.shape()[0];
        const int64_t L = q.shape()[1];
        const int64_t S = k.shape()[1];
        const int64_t D = q.shape()[2];
        const int64_t q_head_stride = L * D;
        const int64_t kv_head_stride = S * D;
        const int64_t mask_head_stride = L * S;
        const int64_t Br = 32;
        const int64_t Bc = 32;
        const int64_t Tr = (L + Br - 1) / Br;
        const int64_t Tc = (S + Bc - 1) / Bc;
        // In GQA, several query heads share the same KV head. Dividing the
        // flattened Q head index maps it back to the corresponding K/V row.
        const int64_t q_kv_heads_ratio = num_heads / num_kv_heads;
        const int64_t causal_offset = S - L;

        const float *q_ptr = q.data<float>();
        const float *k_ptr = k.data<float>();
        const float *v_ptr = v.data<float>();
        const float *mask_ptr = mask.data<float>();

        for (int64_t n = 0; n < N; ++n) {
            const float *q_head = q_ptr + n * q_head_stride;
            const float *k_head = k_ptr + (n / q_kv_heads_ratio) * kv_head_stride;
            const float *v_head = v_ptr + (n / q_kv_heads_ratio) * kv_head_stride;
            const float *mask_head = mask_ptr + n * mask_head_stride;

            for (int64_t tile_q = 0; tile_q < Tr; ++tile_q) {
                const int64_t q_start = tile_q * Br;
                const int64_t q_rows = std::min<int64_t>(Br, L - q_start);

                // Online softmax state for this Q tile. row_max and row_sum
                // are the stable softmax statistics; out_acc stores the
                // unnormalized O = exp(scores - row_max) @ V.
                std::vector<float> row_max(Br, -std::numeric_limits<float>::infinity());
                std::vector<float> row_sum(Br, 0.0f);
                std::vector<float> out_acc(Br * D, 0.0f);

                for (int64_t tile_k = 0; tile_k < Tc; ++tile_k) {
                    const int64_t k_start = tile_k * Bc;
                    const int64_t k_cols = std::min<int64_t>(Bc, S - k_start);

                    const int64_t row_min = q_start;
                    const int64_t row_max_idx = q_start + q_rows - 1;
                    const int64_t col_min = k_start;
                    const int64_t col_max = k_start + k_cols - 1;

                    // For causal attention, skip K tiles that are entirely to
                    // the right of this Q tile. The S-L offset handles decode
                    // where K/V include older cached tokens.
                    if (is_causal && col_min > row_max_idx + causal_offset) {
                        continue;
                    }
                    // If the whole tile is visible under the causal mask,
                    // avoid reading the additive mask for every score.
                    const bool causal_tile_all_valid = is_causal && col_max <= row_min + causal_offset;

                    std::vector<float> scores(Br * Bc, -std::numeric_limits<float>::infinity());

                    // Compute S_tile = Q_tile @ K_tile.T * scale (+ mask).
                    for (int64_t row = 0; row < q_rows; ++row) {
                        const int64_t q_idx = q_start + row;
                        for (int64_t col = 0; col < k_cols; ++col) {
                            const int64_t k_idx = k_start + col;
                            float score = 0.0f;
                            for (int64_t d = 0; d < D; ++d) {
                                score += q_head[q_idx * D + d] * k_head[k_idx * D + d];
                            }
                            score *= scale;
                            if (!causal_tile_all_valid) {
                                score += mask_head[q_idx * S + k_idx];
                            }
                            scores[row * Bc + col] = score;
                        }
                    }

                    // Merge this tile's row maxima with the previous tiles'
                    // maxima. This keeps exp() numerically stable.
                    std::vector<float> new_row_max(Br, -std::numeric_limits<float>::infinity());
                    for (int64_t row = 0; row < q_rows; ++row) {
                        float tile_row_max = -std::numeric_limits<float>::infinity();
                        for (int64_t col = 0; col < k_cols; ++col) {
                            tile_row_max = std::max(tile_row_max, scores[row * Bc + col]);
                        }
                        new_row_max[row] = std::max(row_max[row], tile_row_max);
                    }

                    // Rebase prior softmax state from old row_max to
                    // new_row_max, then add this tile's exp scores.
                    std::vector<float> exp_scores(Br * Bc, 0.0f);
                    for (int64_t row = 0; row < q_rows; ++row) {
                        const float old_scale = std::exp(row_max[row] - new_row_max[row]);
                        float tile_sum = 0.0f;
                        for (int64_t col = 0; col < k_cols; ++col) {
                            const float p = std::exp(scores[row * Bc + col] - new_row_max[row]);
                            exp_scores[row * Bc + col] = p;
                            tile_sum += p;
                        }

                        row_sum[row] = old_scale * row_sum[row] + tile_sum;
                        for (int64_t d = 0; d < D; ++d) {
                            out_acc[row * D + d] *= old_scale;
                        }
                        row_max[row] = new_row_max[row];
                    }

                    // Accumulate the unnormalized output contribution
                    // exp(scores - row_max) @ V_tile.
                    for (int64_t row = 0; row < q_rows; ++row) {
                        for (int64_t col = 0; col < k_cols; ++col) {
                            const float p = exp_scores[row * Bc + col];
                            const int64_t k_idx = k_start + col;
                            for (int64_t d = 0; d < D; ++d) {
                                out_acc[row * D + d] += p * v_head[k_idx * D + d];
                            }
                        }
                    }
                }

                // Normalize by the final softmax denominator.
                for (int64_t row = 0; row < q_rows; ++row) {
                    const int64_t q_idx = q_start + row;
                    for (int64_t d = 0; d < D; ++d) {
                        const int64_t out_loc =
                            mx::elem_to_loc(n * q_head_stride + q_idx * D + d, out_shape, out_strides);
                        out_ptr[out_loc] = out_acc[row * D + d] / row_sum[row];
                    }
                }
            }
        }
    });
}

void FlashAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
#ifdef _METAL_
    const auto &q = inputs[0];
    const auto &k = inputs[1];
    const auto &v = inputs[2];
    const auto &mask = inputs[3];
    auto &out = outputs[0];

    if (out.dtype() != mx::float32) {
        throw std::runtime_error("flash_attention: output dtype must be float32");
    }
    if (!q.flags().row_contiguous || !k.flags().row_contiguous || !v.flags().row_contiguous ||
        !mask.flags().row_contiguous) {
        throw std::runtime_error("flash_attention: q, k, v, and mask must be row contiguous");
    }

    const int N = q.shape()[0];
    const int L = q.shape()[1];
    const int S = k.shape()[1];
    const int E = q.shape()[2];
    const int Br = 32;
    const int Bc = 32;
    const int Tr = (L + Br - 1) / Br;
    const int Tc = (S + Bc - 1) / Bc;

    auto &s = stream();
    auto &d = mx::metal::device(s.device);
    out.set_data(mx::allocator::malloc(out.nbytes()));

    // One Metal threadgroup computes one (flattened head, Q tile) output block.
    auto library = d.get_library("tiny_llm_ext");
    auto kernel = d.get_kernel("flash_attention_f32_e128", library);

    auto &compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(v, 2);
    compute_encoder.set_input_array(mask, 3);
    compute_encoder.set_output_array(out, 4);
    compute_encoder.set_vector_bytes(mask.shape(), 5);
    compute_encoder.set_vector_bytes(mask.strides(), 6);
    compute_encoder.set_bytes(static_cast<int>(is_causal_), 7);
    compute_encoder.set_bytes(N, 8);
    compute_encoder.set_bytes(L, 9);
    compute_encoder.set_bytes(S, 10);
    compute_encoder.set_bytes(E, 11);
    compute_encoder.set_bytes(num_kv_heads_, 12);
    compute_encoder.set_bytes(num_heads_, 13);
    compute_encoder.set_bytes(scale_, 14);
    compute_encoder.set_bytes(Br, 15);
    compute_encoder.set_bytes(Bc, 16);
    compute_encoder.set_bytes(Tr, 17);
    compute_encoder.set_bytes(Tc, 18);

    const size_t tgp_size = kernel->maxTotalThreadsPerThreadgroup();
    const size_t simd_width = kernel->threadExecutionWidth();
    if (simd_width * Br > tgp_size) {
        throw std::runtime_error("flash_attention: simd_width * Br must fit in one threadgroup");
    }
    if (Bc > simd_width) {
        throw std::runtime_error("flash_attention: Bc must fit within one SIMD group");
    }
    if (E > 128) {
        throw std::runtime_error("flash_attention: E must be <= 128");
    }

    // Grid x covers flattened heads; grid y covers Q tiles. Inside a group,
    // each SIMD group owns one Q row and its lanes cover the 32 K columns.
    MTL::Size threadgroups = MTL::Size(N, Tr, 1);
    MTL::Size threads_per_group = MTL::Size(Br, simd_width, 1);
    compute_encoder.dispatch_threadgroups(threadgroups, threads_per_group);
#else
    throw std::runtime_error("FlashAttention has no GPU implementation yet.");
#endif
}

}  // namespace tiny_llm_ext
