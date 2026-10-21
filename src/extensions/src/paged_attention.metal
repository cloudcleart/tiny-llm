#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include "mlx/backend/metal/kernels/utils.h"
#include "cooperative_matrix.h"

using namespace metal;

// Week 3, Day 3:
//   paged_cache_update_kernel
// Week 3, Day 4:
//   paged_attention_decode
//   paged_attention_scalar_f32
// Week 3, Day 5:
//   paged_attention_mma_bf16_d128
//
// The public C++ API remains paged_attention across Days 4-5. Day 5 replaces
// only the long-query schedule behind that stable boundary.

template <typename T>
[[kernel]] void paged_cache_update_kernel(
    device const T* values [[buffer(0)]],
    device T* pages [[buffer(1)]],
    constant const int& heads [[buffer(2)]],
    constant const int& length [[buffer(3)]],
    constant const int& head_dim [[buffer(4)]],
    constant const int& page_size [[buffer(5)]],
    constant const int& page_id [[buffer(6)]],
    constant const int& start [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    const int total = heads * length * head_dim;
    if (index >= static_cast<uint>(total)) return;
    const int head_stride = length * head_dim;
    const int head = index / head_stride;
    const int within_head = index - head * head_stride;
    const int token = within_head / head_dim;
    const int dim = within_head - token * head_dim;
    const int destination =
        ((page_id * heads + head) * page_size + start + token) * head_dim + dim;
    pages[destination] = values[index];
}

instantiate_kernel("paged_cache_update_f32", paged_cache_update_kernel, float);
instantiate_kernel("paged_cache_update_bf16", paged_cache_update_kernel, bfloat16_t);

// Day 4 short-query schedule: one threadgroup owns one query row, and 32
// SIMD groups partition each logical page. All groups keep the same online
// softmax recurrence, regardless of the GQA ratio or input dtype.
// FIXED_D=128 is instantiated only for the BF16 Qwen specialization; generic
// variants use runtime D and bounded register-array capacities.
template <typename T, int FIXED_D = 0, int MAX_VALUES_PER_LANE = 4, bool STREAMED_D = false>
[[kernel]] void paged_attention_decode(
    device const T* q [[buffer(0)]],
    device const T* keys [[buffer(1)]],
    device const T* values [[buffer(2)]],
    device const int* table [[buffer(3)]],
    device const int* lengths [[buffer(4)]],
    device T* out [[buffer(5)]],
    constant int& L [[buffer(6)]],
    constant int& D [[buffer(7)]],
    constant int& page_size [[buffer(8)]],
    constant int& table_width [[buffer(9)]],
    constant int& heads [[buffer(10)]],
    constant int& kv_heads [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    constant int& causal [[buffer(13)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[simdgroup_index_in_threadgroup]]) {
    constexpr int GROUPS = 32;
    const int dimension = FIXED_D == 0 ? D : FIXED_D;
    const int output_span = STREAMED_D ? 128 : dimension;
    const int values_per_lane = (output_span + 31) / 32;
    const int query_pos = row % L;
    const int head_row = row / L;
    const int batch = head_row / heads;
    const int head = head_row % heads;
    const int kv_head = head / (heads / kv_heads);
    const int context = lengths[batch];
    const int visible = causal ? clamp(context - L + query_pos + 1, 0, context) : context;
    const size_t q_base = size_t(row) * D;

    // Wide generic heads repeat the page walk for each 128-D output slice.
    // This trades repeated dot products for bounded registers while retaining
    // all 32 context partitions. The BF16 D=128 specialization needs one pass.
    for (int output_start = 0; output_start < dimension; output_start += output_span) {
        // For D=128, lane l owns contiguous dimensions 4*l + {0,1,2,3}.
        // Non-streamed heads load Q once before walking pages. Their Q and
        // output stay thread-local FP32 values; wide heads reload Q for the
        // complete dot product and keep only the output slice in registers.
        float query_values[MAX_VALUES_PER_LANE];
        float accumulator[MAX_VALUES_PER_LANE];
        #pragma clang loop unroll(full)
        for (int item = 0; item < values_per_lane; ++item) {
            const int dim = output_start + lane * values_per_lane + item;
            if constexpr (!STREAMED_D) {
                query_values[item] = dim < dimension ? float(q[q_base + dim]) : 0.0f;
            }
            accumulator[item] = 0.0f;
        }
        float maximum = -INFINITY;
        float denominator = 0.0f;

        for (int page_start = 0; page_start < visible; page_start += page_size) {
            // Resolve the page outside the token loop and reuse its physical base.
            // Group g handles page slots g, g+32, g+64, ...; the partial tail page
            // is bounded by visible, so its unwritten slots are never loaded.
            const int page = table[batch * table_width + page_start / page_size];
            const size_t page_base = (size_t(page) * kv_heads + kv_head) * page_size * D;
            const int live_slots = min(page_size, visible - page_start);
            for (int slot = group; slot < live_slots; slot += GROUPS) {
                const size_t kv_base = page_base + size_t(slot) * D;
                float dot = 0.0f;
                if constexpr (STREAMED_D) {
                    // The score always includes the entire head, even though
                    // this pass only accumulates a bounded output slice.
                    for (int dim = lane; dim < dimension; dim += 32) {
                        dot += float(q[q_base + dim]) * float(keys[kv_base + dim]);
                    }
                } else {
                    #pragma clang loop unroll(full)
                    for (int item = 0; item < values_per_lane; ++item) {
                        const int dim = output_start + lane * values_per_lane + item;
                        if (dim < dimension) dot += query_values[item] * float(keys[kv_base + dim]);
                    }
                }
                const float score = simd_sum(dot) * scale;
                const float next_max = max(maximum, score);
                const float previous_scale = exp(maximum - next_max);
                const float weight = exp(score - next_max);
                denominator = denominator * previous_scale + weight;
                #pragma clang loop unroll(full)
                for (int item = 0; item < values_per_lane; ++item) {
                    const int dim = output_start + lane * values_per_lane + item;
                    if (dim < dimension) {
                        accumulator[item] = accumulator[item] * previous_scale + weight * float(values[kv_base + dim]);
                    }
                }
                maximum = next_max;
            }
        }

        // Scratch holds one reusable 32x32 transpose tile and two 32-element
        // arrays: (1024 + 32 + 32) * 4 = 4,352 bytes, including D=128.
        threadgroup float* partial_outputs = scratch;
        threadgroup float* partial_maxima = partial_outputs + GROUPS * 32;
        threadgroup float* partial_sums = partial_maxima + GROUPS;
        if (lane == 0) {
            partial_maxima[group] = maximum;
            partial_sums[group] = denominator;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Lanes now read the 32 context partitions. All SIMD groups obtain the
        // same normalization reference. Empty partitions explicitly contribute
        // zero, including the all-empty context, avoiding exp(-inf - -inf).
        const float global_max = simd_max(partial_maxima[lane]);
        const float factor = partial_sums[lane] > 0.0f ? exp(partial_maxima[lane] - global_max) : 0.0f;
        const float global_sum = simd_sum(partial_sums[lane] * factor);
        #pragma clang loop unroll(full)
        for (int item = 0; item < values_per_lane; ++item) {
            // Store as [dimension owner][context partition], then read the
            // transposed ownership: group s combines dimension owner s across
            // the 32 partitions represented by its lanes. Reuse the tile for
            // each component; D=128 needs four passes through this same buffer.
            partial_outputs[lane * GROUPS + group] = accumulator[item];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            accumulator[item] = simd_sum(partial_outputs[group * GROUPS + lane] * factor);
            // No group may overwrite the next pass until every reader finishes.
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            #pragma clang loop unroll(full)
            for (int item = 0; item < values_per_lane; ++item) {
                const int dim = output_start + group * values_per_lane + item;
                if (dim < dimension) out[q_base + dim] = T(global_sum > 0.0f ? accumulator[item] / global_sum : 0.0f);
            }
        }
    }
}

instantiate_kernel("paged_attention_decode_f32", paged_attention_decode, float);
instantiate_kernel("paged_attention_decode_bf16", paged_attention_decode, bfloat16_t);
instantiate_kernel("paged_attention_decode_bf16_d128", paged_attention_decode, bfloat16_t, 128);
// Generic wide-head variants preserve runtime-D behavior beyond the Qwen
// specialization without allocating a complete output vector in scratch.
instantiate_kernel("paged_attention_decode_f32_max256", paged_attention_decode, float, 0, 8);
instantiate_kernel("paged_attention_decode_bf16_max256", paged_attention_decode, bfloat16_t, 0, 8);
instantiate_kernel("paged_attention_decode_f32_wide", paged_attention_decode, float, 0, 4, true);
instantiate_kernel("paged_attention_decode_bf16_wide", paged_attention_decode, bfloat16_t, 0, 4, true);

// Day 4 scalar paged-prefill schedule. This is intentionally independent of
// the Day 5 matrix-instruction kernel: every multiply and every softmax update
// can be inspected directly here.
//
// Launch a (B * heads, ceil(L / 16)) threadgroup grid with 512 threads per
// group. The 16 SIMD groups own 16 query rows. Within each SIMD group, lane
// k calculates the score for key k of the current 32-key tile; after softmax,
// those same lanes own output dimensions lane, lane + 32, and so on.
//
// Scratch is fixed-size even when D is large:
//   Q staging:       16 * 128 floats
//   K/V staging:    32 * 128 floats (the same buffer is reused for V)
//   probabilities:  16 * 32 floats
//   page IDs:       32 ints
// Total: 26,752 bytes. Dimensions beyond 128 use additional staging passes.
// All local output accumulation remains FP32, including BF16 inputs/outputs.
// The C++ entry point must enforce 0 < D <= 2048 for this implementation.
template <typename T>
[[kernel]] void paged_attention_scalar(
    device const T* q [[buffer(0)]],
    device const T* keys [[buffer(1)]],
    device const T* values [[buffer(2)]],
    device const int* table [[buffer(3)]],
    device const int* lengths [[buffer(4)]],
    device T* out [[buffer(5)]],
    constant int& L [[buffer(6)]],
    constant int& D [[buffer(7)]],
    constant int& page_size [[buffer(8)]],
    constant int& table_width [[buffer(9)]],
    constant int& heads [[buffer(10)]],
    constant int& kv_heads [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    constant int& causal [[buffer(13)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint2 block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[simdgroup_index_in_threadgroup]]) {
    constexpr int BQ = 16;
    constexpr int BK = 32;
    constexpr int BD = 128;
    constexpr int THREADS = BQ * BK;
    constexpr int MAX_ITEMS_PER_LANE = 2048 / BK;

    const int head_row = block.x;
    const int query_start = block.y * BQ;
    const int query_pos = query_start + group;
    const bool query_valid = query_pos < L;
    const int thread_idx = group * BK + lane;
    const int batch = head_row / heads;
    const int head = head_row % heads;
    const int kv_head = head / (heads / kv_heads);
    const int context = lengths[batch];
    const size_t q_base = size_t(head_row) * L * D;
    const int output_items = (D + BK - 1) / BK;

    threadgroup float* q_tile = scratch;
    threadgroup float* kv_tile = q_tile + BQ * BD;
    threadgroup float* probabilities = kv_tile + BK * BD;
    threadgroup int* physical_pages =
        reinterpret_cast<threadgroup int*>(probabilities + BQ * BK);

    float output_fragment[MAX_ITEMS_PER_LANE];
    for (int item = 0; item < output_items; ++item) {
        output_fragment[item] = 0.0f;
    }
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // Every thread must run the same tile and dimension loops because the
    // shared staging buffers require full-threadgroup barriers. A query tail
    // therefore stays active for synchronization, but contributes no scores
    // and never writes an output row outside L.
    const int last_query = min(query_start + BQ, L) - 1;
    const int tile_visible = context == 0
        ? 0
        : (causal ? clamp(context - L + last_query + 1, 0, context) : context);
    const int row_visible = query_valid && context > 0
        ? (causal ? clamp(context - L + query_pos + 1, 0, context) : context)
        : 0;

    for (int tile_start = 0; tile_start < tile_visible; tile_start += BK) {
        const int first_logical_page = tile_start / page_size;
        const int first_slot = tile_start % page_size;
        const bool one_page = first_slot + BK <= page_size;

        // Tile origins are multiples of BK. If this whole aligned tile fits
        // in one logical page, resolve exactly one physical page ID. Tiles
        // crossing page boundaries instead resolve each live key separately.
        // Invalid tail keys get -1 without touching their page-table entries.
        if (one_page) {
            if (thread_idx == 0) {
                physical_pages[0] = first_logical_page < table_width
                    ? table[batch * table_width + first_logical_page]
                    : -1;
            }
        } else if (thread_idx < BK) {
            const int key_pos = tile_start + thread_idx;
            const int logical_page = key_pos / page_size;
            physical_pages[thread_idx] =
                key_pos < tile_visible && logical_page < table_width
                    ? table[batch * table_width + logical_page]
                    : -1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // One SIMD group owns a query row and each lane owns one key in the
        // tile. Accumulate that Q dot K score over dimension chunks. The
        // threadgroup loads each K chunk once and shares it among all 16 rows.
        const int key_pos = tile_start + lane;
        const int key_page = one_page ? physical_pages[0] : physical_pages[lane];
        const bool score_valid = query_valid && key_pos < row_visible && key_page >= 0;
        float dot = 0.0f;
        for (int dim_start = 0; dim_start < D; dim_start += BD) {
            for (int idx = thread_idx; idx < BQ * BD; idx += THREADS) {
                const int query = query_start + idx / BD;
                const int dim = dim_start + idx % BD;
                q_tile[idx] = query < L && dim < D
                    ? float(q[q_base + size_t(query) * D + dim])
                    : 0.0f;
            }
            for (int idx = thread_idx; idx < BK * BD; idx += THREADS) {
                const int key = idx / BD;
                const int global_key = tile_start + key;
                const int dim = dim_start + idx % BD;
                const int page = one_page ? physical_pages[0] : physical_pages[key];
                float value = 0.0f;
                // The live-key guard is essential for the last partial tile:
                // a mapped physical page may contain unwritten/NaN tail slots.
                if (global_key < tile_visible && page >= 0 && dim < D) {
                    const int slot = one_page ? first_slot + key : global_key % page_size;
                    const size_t base =
                        ((size_t(page) * kv_heads + kv_head) * page_size + slot) * D;
                    value = float(keys[base + dim]);
                }
                kv_tile[idx] = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (score_valid) {
                const int live_dims = min(BD, D - dim_start);
                for (int dim = 0; dim < live_dims; ++dim) {
                    dot += q_tile[group * BD + dim] * kv_tile[lane * BD + dim];
                }
            }
            // The next staging pass can overwrite Q/K only after every row
            // has consumed the current chunk.
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Each lane now holds one score, so SIMD reductions give the tile's
        // maximum and denominator. Rescale both old denominator and old
        // output when the normalization reference changes.
        const float score = score_valid ? dot * scale : -INFINITY;
        const float tile_max = simd_max(score);
        const float next_max = max(running_max, tile_max);
        const bool has_scores = next_max != -INFINITY;
        const float previous_scale = running_max != -INFINITY && has_scores
            ? exp(running_max - next_max)
            : 0.0f;
        const float probability = score_valid && has_scores
            ? exp(score - next_max)
            : 0.0f;
        running_sum = previous_scale * running_sum + simd_sum(probability);
        running_max = next_max;
        probabilities[group * BK + lane] = probability;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reuse the K staging buffer for V. Probabilities are stored once per
        // query/key pair; every output dimension uses those same weights.
        // Output fragments stay FP32 until the final normalized T store.
        for (int dim_start = 0; dim_start < D; dim_start += BD) {
            for (int idx = thread_idx; idx < BK * BD; idx += THREADS) {
                const int key = idx / BD;
                const int global_key = tile_start + key;
                const int dim = dim_start + idx % BD;
                const int page = one_page ? physical_pages[0] : physical_pages[key];
                float value = 0.0f;
                if (global_key < tile_visible && page >= 0 && dim < D) {
                    const int slot = one_page ? first_slot + key : global_key % page_size;
                    const size_t base =
                        ((size_t(page) * kv_heads + kv_head) * page_size + slot) * D;
                    value = float(values[base + dim]);
                }
                kv_tile[idx] = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (query_valid) {
                for (int dim_in_chunk = lane; dim_in_chunk < BD; dim_in_chunk += BK) {
                    const int dim = dim_start + dim_in_chunk;
                    if (dim < D) {
                        float contribution = 0.0f;
                        for (int key = 0; key < BK; ++key) {
                            contribution += probabilities[group * BK + key] *
                                kv_tile[key * BD + dim_in_chunk];
                        }
                        const int item = dim / BK;
                        output_fragment[item] =
                            previous_scale * output_fragment[item] + contribution;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Idle batch slots have no tiles, so their zero-initialized accumulators
    // produce zeros and their invalid page IDs are never loaded or dereferenced.
    if (query_valid) {
        for (int item = 0; item < output_items; ++item) {
            const int dim = lane + item * BK;
            if (dim < D) {
                out[q_base + size_t(query_pos) * D + dim] =
                    T(running_sum > 0.0f ? output_fragment[item] / running_sum : 0.0f);
            }
        }
    }
}

instantiate_kernel("paged_attention_scalar_f32", paged_attention_scalar, float);
instantiate_kernel("paged_attention_scalar_bf16", paged_attention_scalar, bfloat16_t);

namespace {

// A row of an 8x8 SIMD matrix is distributed over lanes differing by 1 and
// 8. Each lane owns two adjacent columns, as in course_matrix_coordinate.
// Explicitly unroll fixed fragment loops to keep matrix indices constant and
// enable register allocation; the runtime-length K/V loop stays rolled.
template <int FRAGMENTS>
inline float paged_row_max(thread const simdgroup_matrix<float, 8, 8>* tiles) {
    float result = -INFINITY;
    #pragma clang loop unroll(full)
    for (int i = 0; i < FRAGMENTS; ++i) {
        const auto elements = tiles[i].thread_elements();
        result = max(result, max(elements[0], elements[1]));
    }
    result = max(result, simd_shuffle_xor(result, ushort(1)));
    return max(result, simd_shuffle_xor(result, ushort(8)));
}

template <int FRAGMENTS>
inline float paged_row_sum(thread const simdgroup_matrix<float, 8, 8>* tiles) {
    float result = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < FRAGMENTS; ++i) {
        const auto elements = tiles[i].thread_elements();
        result += elements[0] + elements[1];
    }
    result += simd_shuffle_xor(result, ushort(1));
    return result + simd_shuffle_xor(result, ushort(8));
}

}  // namespace

// Day 5: one 64-row query tile stays in threadgroup memory while 32-row
// page-backed K/V tiles stream through. Each SIMD group owns eight query rows.
// The paging and output layouts remain compatible with the Day 4 kernels.
[[kernel, max_total_threads_per_threadgroup(256)]] void paged_attention_mma_bf16_d128(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* keys [[buffer(1)]],
    device const bfloat* values [[buffer(2)]],
    device const int* table [[buffer(3)]],
    device const int* lengths [[buffer(4)]],
    device bfloat* out [[buffer(5)]],
    constant int& L [[buffer(6)]],
    constant int& D [[buffer(7)]],
    constant int& page_size [[buffer(8)]],
    constant int& table_width [[buffer(9)]],
    constant int& heads [[buffer(10)]],
    constant int& kv_heads [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    constant int& causal [[buffer(13)]],
    uint2 block [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort group [[simdgroup_index_in_threadgroup]]) {
    constexpr int BQ = 64, BK = 32, BD = 128, MMA = 8, THREADS = 256;
    constexpr int LDQ = BD + 2, LDK = BK + 2, LDV = BD + 2;
    constexpr int SCORE_FRAGMENTS = BK / MMA, OUTPUT_FRAGMENTS = BD / MMA;
    using QLoader = tiny_llm::CooperativeTileLoader<bfloat, BQ, BD, LDQ, THREADS>;
    using KLoader = tiny_llm::CooperativeTileLoader<bfloat, BK, BD, LDK, THREADS, true>;
    using VLoader = tiny_llm::CooperativeTileLoader<bfloat, BK, BD, LDV, THREADS>;
    // The 256 threads stage 32 BF16 Q elements or 16 BF16 K/V elements each.
    // Padded shared-memory strides do not add device-memory reads.

    const int head_row = block.x;
    const int batch = head_row / heads;
    const int kv_head = (head_row % heads) / (heads / kv_heads);
    const int query_start = block.y * BQ;
    const int thread_idx = group * 32 + lane;
    const ushort2 coord = tiny_llm::course_matrix_coordinate(lane);
    const int query_pos = query_start + group * MMA + coord.y;
    // Tail rows must reach every barrier; mask their scores and stores instead
    // of returning early from the threadgroup.
    const bool query_valid = query_pos < L;
    const int context = lengths[batch];
    const size_t q_base = size_t(head_row) * L * D;
    // Store logits and running_max in base-2 units so exp2 implements softmax.
    const float scale_log2 = scale * 1.4426950408889634f;

    threadgroup bfloat q_tile[BQ * LDQ];
    // K is transposed to [D, BK]; V is [BK, D]. Reuse the larger allocation.
    threadgroup bfloat kv_tile[BD * LDK];
    threadgroup int physical_pages[BK];
    // Stage Q once per threadgroup; shared-to-fragment reads repeat per K tile.
    QLoader::load(q + q_base + size_t(query_start) * BD, BD, q_tile,
                  thread_idx, min(BQ, L - query_start), BD);
    // Publish the Q tile before any SIMD group reads its matrix fragments.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, MMA, MMA> output[OUTPUT_FRAGMENTS];
    #pragma clang loop unroll(full)
    for (int i = 0; i < OUTPUT_FRAGMENTS; ++i) output[i].thread_elements() = 0.0f;
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    const int last_query = min(query_start + BQ, L) - 1;
    // Load through the last valid row's causal bound. Earlier rows apply their
    // own masks below, allowing all SIMD groups to share the same K/V tiles.
    const int visible = causal ? clamp(context - L + last_query + 1, 0, context) : context;

    for (int tile_start = 0; tile_start < visible; tile_start += BK) {
        const int first_slot = tile_start % page_size;
        // Test the nominal tile span; a partial tail can also use this path.
        const bool one_page = first_slot + BK <= page_size;
        const int live_keys = min(BK, visible - tile_start);
        if (one_page) {
            if (thread_idx == 0) {
                physical_pages[0] = table[batch * table_width + tile_start / page_size];
            }
        } else if (thread_idx < BK) {
            physical_pages[thread_idx] = thread_idx < live_keys
                ? table[batch * table_width + (tile_start + thread_idx) / page_size]
                : -1;
        }
        // Share page IDs with the threads that stage K and, later, V.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (one_page) {
            const size_t base =
                ((size_t(physical_pages[0]) * kv_heads + kv_head) * page_size + first_slot) * BD;
            KLoader::load(keys + base, BD, kv_tile, thread_idx, live_keys, BD);
        } else {
            // Keep each thread's 16 source dimensions contiguous within one
            // key, even across pages. Only the shared-memory store transposes K.
            constexpr int CHUNK = BK * BD / THREADS;
            const int key = thread_idx * CHUNK / BD;
            const int dim = thread_idx * CHUNK % BD;
            const int page = physical_pages[key];
            #pragma clang loop unroll(full)
            for (int i = 0; i < CHUNK; ++i) {
                bfloat value = bfloat(0.0f);
                if (key < live_keys && page >= 0) {
                    const int slot = (tile_start + key) % page_size;
                    const size_t base = ((size_t(page) * kv_heads + kv_head) * page_size + slot) * BD;
                    value = keys[base + dim + i];
                }
                kv_tile[(dim + i) * LDK + key] = value;
            }
        }
        // All K stores must be visible before any group starts QK^T.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scores stay in FP32 matrix fragments, then become unnormalized
        // softmax weights in place; no full attention matrix is materialized.
        simdgroup_matrix<float, MMA, MMA> scores[SCORE_FRAGMENTS];
        #pragma clang loop unroll(full)
        for (int i = 0; i < SCORE_FRAGMENTS; ++i) scores[i].thread_elements() = 0.0f;
        // Full unrolling keeps the fixed-size matrix-fragment indices constant.
        #pragma clang loop unroll(full)
        for (int dim = 0; dim < BD; dim += MMA) {
            simdgroup_matrix<bfloat, MMA, MMA> q_fragment;
            tiny_llm::course_load_matrix(q_fragment, q_tile + group * MMA * LDQ + dim, LDQ, lane);
            #pragma clang loop unroll(full)
            for (int i = 0; i < SCORE_FRAGMENTS; ++i) {
                simdgroup_matrix<bfloat, MMA, MMA> k_fragment;
                tiny_llm::course_load_matrix(k_fragment, kv_tile + dim * LDK + i * MMA, LDK, lane);
                simdgroup_multiply_accumulate(scores[i], q_fragment, k_fragment, scores[i]);
            }
        }
        // Query positions are relative to this call; context - L converts them
        // to absolute positions in the cached sequence for causal masking.
        #pragma clang loop unroll(full)
        for (int i = 0; i < SCORE_FRAGMENTS; ++i) {
            thread auto& elements = scores[i].thread_elements();
            #pragma clang loop unroll(full)
            for (int j = 0; j < 2; ++j) {
                const int key = tile_start + i * MMA + coord.x + j;
                const bool valid = query_valid && key < context &&
                    (!causal || key <= query_pos + context - L);
                elements[j] = valid ? elements[j] * scale_log2 : -INFINITY;
            }
        }

        const float next_max = max(running_max, paged_row_max<SCORE_FRAGMENTS>(scores));
        const bool has_scores = next_max != -INFINITY;
        // Change both the old denominator and output accumulator to the new
        // maximum's scale. Guard empty rows to avoid evaluating -inf - -inf.
        const float old_scale = running_max != -INFINITY && has_scores
            ? fast::exp2(running_max - next_max) : 0.0f;
        #pragma clang loop unroll(full)
        for (int i = 0; i < SCORE_FRAGMENTS; ++i) {
            thread auto& elements = scores[i].thread_elements();
            #pragma clang loop unroll(full)
            for (int j = 0; j < 2; ++j) {
                elements[j] = elements[j] != -INFINITY && has_scores
                    ? fast::exp2(elements[j] - next_max) : 0.0f;
            }
        }
        running_max = next_max;
        running_sum = old_scale * running_sum + paged_row_sum<SCORE_FRAGMENTS>(scores);
        #pragma clang loop unroll(full)
        for (int i = 0; i < OUTPUT_FRAGMENTS; ++i) output[i].thread_elements() *= old_scale;
        // The denominator uses FP32 weights; only the PV matrix input is rounded
        // to BF16. Output accumulation remains FP32 across all K/V tiles.
        simdgroup_matrix<bfloat, MMA, MMA> probabilities[SCORE_FRAGMENTS];
        #pragma clang loop unroll(full)
        for (int i = 0; i < SCORE_FRAGMENTS; ++i) {
            const auto elements = scores[i].thread_elements();
            probabilities[i].thread_elements()[0] = bfloat(elements[0]);
            probabilities[i].thread_elements()[1] = bfloat(elements[1]);
        }

        // All groups finish reading K before its buffer is reused for V.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (one_page) {
            const size_t base =
                ((size_t(physical_pages[0]) * kv_heads + kv_head) * page_size + first_slot) * BD;
            VLoader::load(values + base, BD, kv_tile, thread_idx, live_keys, BD);
        } else {
            constexpr int CHUNK = BK * BD / THREADS;
            const int key = thread_idx * CHUNK / BD;
            const int dim = thread_idx * CHUNK % BD;
            const int page = physical_pages[key];
            #pragma clang loop unroll(full)
            for (int i = 0; i < CHUNK; ++i) {
                bfloat value = bfloat(0.0f);
                if (key < live_keys && page >= 0) {
                    const int slot = (tile_start + key) % page_size;
                    const size_t base = ((size_t(page) * kv_heads + kv_head) * page_size + slot) * BD;
                    value = values[base + dim + i];
                }
                kv_tile[key * LDV + dim + i] = value;
            }
        }
        // Publish V before the SIMD groups read it for PV.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (int i = 0; i < OUTPUT_FRAGMENTS; ++i) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < SCORE_FRAGMENTS; ++j) {
                simdgroup_matrix<bfloat, MMA, MMA> v_fragment;
                tiny_llm::course_load_matrix(v_fragment, kv_tile + j * MMA * LDV + i * MMA, LDV, lane);
                simdgroup_multiply_accumulate(output[i], probabilities[j], v_fragment, output[i]);
            }
        }
        // Finish every V read before the next iteration overwrites the shared
        // K/V buffer or its page IDs.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write each output once. Empty batch slots produce zero,
    // and padded query rows never store.
    if (query_valid) {
        #pragma clang loop unroll(full)
        for (int i = 0; i < OUTPUT_FRAGMENTS; ++i) {
            const auto elements = output[i].thread_elements();
            #pragma clang loop unroll(full)
            for (int j = 0; j < 2; ++j) {
                const int dim = i * MMA + coord.x + j;
                out[q_base + size_t(query_pos) * BD + dim] =
                    bfloat(running_sum > 0.0f ? elements[j] / running_sum : 0.0f);
            }
        }
    }
}
