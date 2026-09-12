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

    // Early exit if this threadgroup's row index is out of bounds
    if (row >= static_cast<uint>(rows)) {
        return;
    }

    // Define thread configuration constants
    constexpr int threads_per_threadgroup = 256;
    constexpr int simdgroups_per_threadgroup = threads_per_threadgroup / 32;

    // ==================== Phase 1: Parallel sum-of-squares computation ====================
    // Each thread computes sum of squares for its assigned columns (stride by total threads)
    float sum_sq = 0.0f;
    for (int col = thread_index; col < dim; col += threads_per_threadgroup) {
        const float value = static_cast<float>(x[row * dim + col]);
        sum_sq += value * value;
    }

    // ==================== Phase 2: SIMD group level reduction ====================
    // Reduce sum_sq within each SIMD group (32 lanes per group)
    sum_sq = simd_sum(sum_sq);
    // First lane of each SIMD group stores the group's partial sum to shared memory
    if (lane == 0) {
        partial_sums[simdgroup] = sum_sq;
    }
    // Synchronize all threads within the threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ==================== Phase 3: Threadgroup level reduction ====================
    // SIMD group 0 aggregates all partial sums from all SIMD groups
    if (simdgroup == 0) {
        // Each lane reads from a different SIMD group's partial sum (if exists)
        float threadgroup_sum =
            lane < simdgroups_per_threadgroup ? partial_sums[lane] : 0.0f;
        // Reduce across the lanes in SIMD group 0
        threadgroup_sum = simd_sum(threadgroup_sum);
        // First lane stores the final threadgroup sum
        if (lane == 0) {
            partial_sums[0] = threadgroup_sum;
        }
    }
    // Synchronize all threads within the threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ==================== Phase 4: Normalize and scale output ====================
    // Compute the reciprocal of RMS: 1.0 / sqrt(mean_sq + eps)
    // partial_sums[0] contains sum of all squares
    const float inverse_rms =
        rsqrt(partial_sums[0] / static_cast<float>(dim) + eps);

    // Each thread writes normalized and scaled output for its assigned columns
    for (int col = thread_index; col < dim; col += threads_per_threadgroup) {
        out[row * dim + col] = static_cast<T>(
            static_cast<float>(x[row * dim + col]) * inverse_rms *
            static_cast<float>(weight[col])
        );
    }
}

template <typename T>
[[kernel]] void week2_rope(
    device const T* x [[buffer(0)]],
    device const int32_t* offsets [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const int& batch [[buffer(3)]],
    constant const int& length [[buffer(4)]],
    constant const int& heads [[buffer(5)]],
    constant const int& head_dim [[buffer(6)]],
    constant const int& dims [[buffer(7)]],
    constant const float& base [[buffer(8)]],
    constant const int& traditional [[buffer(9)]],
    uint index [[thread_position_in_grid]]) {
    
    // ==================== Step 1: Calculate work organization ====================
    // Each thread processes 4 heads at a time (heads_per_thread = 4)
    constexpr int heads_per_thread = 4;
    // Number of rotation dimensions divided by 2 (rotation is applied to pairs)
    const int half_dim = dims / 2;
    // Number of non-rotated dimensions (tail of head_dim that doesn't rotate)
    const int tail_dims = head_dim - dims;
    // Items per head block: rotation pairs + tail dimensions
    const int items_per_head_block = half_dim + tail_dims;
    // Calculate number of head blocks: ceiling division of heads by heads_per_thread
    const int head_blocks = (heads + heads_per_thread - 1) / heads_per_thread;
    // Total work items: batch * sequence_length * head_blocks * items_per_head_block
    const int total = batch * length * head_blocks * items_per_head_block;
    
    // Early exit: skip if this thread's index is out of bounds
    if (index >= static_cast<uint>(total)) {
        return;
    }

    // ==================== Step 2: Decompose flat thread index into dimensions ====================
    // Each index maps to: [batch, token, head_block, item_in_head_block]
    
    // Which item within the head block (0 to items_per_head_block-1)
    const int item = index % items_per_head_block;
    // Which head block (0 to head_blocks-1)
    const int head_block = (index / items_per_head_block) % head_blocks;
    // Which token/position in sequence (0 to length-1)
    const int token = (index / (items_per_head_block * head_blocks)) % length;
    // Which batch element (0 to batch-1)
    const int batch_index = index / (items_per_head_block * head_blocks * length);
    
    // Map head block to actual head indices
    // Each head block covers heads_per_thread heads
    const int first_head = head_block * heads_per_thread;
    // Ensure we don't exceed total heads
    const int last_head = min(first_head + heads_per_thread, heads);
    
    // Calculate base row index: (batch, token) -> flattened position in heads*head_dim layout
    const int row_base = (batch_index * length + token) * heads * head_dim;

    // ==================== Step 3: Handle non-rotated dimensions ====================
    // If item >= half_dim, we're in the tail region (non-rotated dimensions)
    if (item >= half_dim) {
        // Map item index to actual dimension: half_dim rotation dimensions are skipped
        const int dimension = dims + item - half_dim;
        // Copy non-rotated elements as-is (no rotation applied)
        for (int head = first_head; head < last_head; ++head) {
            const int element = row_base + head * head_dim + dimension;
            out[element] = x[element];
        }
        return;
    }

    // ==================== Step 4: Calculate rotation parameters ====================
    // pair: which rotation pair (0 to half_dim-1)
    const int pair = item;
    
    // Frequency for this rotation dimension pair
    // Formula: base^(-2*pair/dims)
    // This creates different rotation frequencies for different dimensions
    const float frequency_power = -static_cast<float>(pair) / half_dim;
    
    // Absolute position: offset (for KV cache) + token index
    const float position = static_cast<float>(offsets[batch_index] + token);
    
    // Calculate rotation angle: position * base^(frequency_power)
    // Use float precision for float32, faster approximation for f16/bf16
    const float angle = sizeof(T) == sizeof(float)
        ? position * pow(base, frequency_power)
        : position * fast::exp2(frequency_power * log2(base));
    
    // Pre-compute cosine and sine of the angle
    const float cosine = sizeof(T) == sizeof(float) ? cos(angle) : fast::cos(angle);
    const float sine = sizeof(T) == sizeof(float) ? sin(angle) : fast::sin(angle);

    // ==================== Step 5: Apply rotation to all assigned heads ====================
    for (int head = first_head; head < last_head; ++head) {
        // Base position for this head in output
        const int head_base = row_base + head * head_dim;
        
        // Determine positions of the value pair based on storage format
        // Traditional format: [Re0, Im0, Re1, Im1, ...] (interleaved pairs)
        // Interleaved format: [Re0, Re1, ..., Im0, Im1, ...] (all reals, then all imaginaries)
        const int real_index = traditional ? head_base + pair * 2 : head_base + pair;
        const int imaginary_index = traditional ? real_index + 1 : real_index + half_dim;
        
        // Load the complex number pair (real, imaginary)
        const float real = static_cast<float>(x[real_index]);
        const float imaginary = static_cast<float>(x[imaginary_index]);
        
        // Apply 2D rotation: multiply complex number by e^(i*angle)
        // New real = real * cos(angle) - imaginary * sin(angle)
        // New imaginary = imaginary * cos(angle) + real * sin(angle)
        out[real_index] = static_cast<T>(real * cosine - imaginary * sine);
        out[imaginary_index] = static_cast<T>(imaginary * cosine + real * sine);
    }
}

template <typename T>
[[kernel]] void week2_swiglu(
    device const T* gate [[buffer(0)]],
    device const T* up [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const int& size [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= static_cast<uint>(size)) {
        return;
    }

    const float gate_value = static_cast<float>(gate[index]);
    const float up_value = static_cast<float>(up[index]);
    const float silu = gate_value / (1.0f + exp(-gate_value));
    out[index] = static_cast<T>(silu * up_value);
}

// One threadgroup handles a query row; SIMD groups partition S and lanes partition D.
// Update softmax state and weighted V immediately, without storing full scores/probabilities.
template <typename T>
[[kernel]] void week2_decode_attention(
    device const T* q [[buffer(0)]],
    device const T* k [[buffer(1)]],
    device const T* v [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device T* out [[buffer(4)]],
    constant const int& length [[buffer(5)]],
    constant const int& context [[buffer(6)]],
    constant const int& dim [[buffer(7)]],
    constant const int& heads [[buffer(8)]],
    constant const int& kv_heads [[buffer(9)]],
    constant const float& scale [[buffer(10)]],
    constant const int& is_causal [[buffer(11)]],
    constant const int& has_mask [[buffer(12)]],
    constant const int& groups [[buffer(13)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    // row = (batch * H_q + head) * L + query_position。
    // query_head includes the batch offset; kv_head must include that offset as well.
    const int query_head = row / length;
    const int query_position = row % length;
    const int batch = query_head / heads;
    const int kv_head = batch * kv_heads + (query_head % heads) / (heads / kv_heads);
    const size_t kv_base = size_t(kv_head) * context * dim;
    const size_t q_base = size_t(row) * dim;

    // Lane t owns dimensions t, t+32, ..., up to 8 elements for D<=256.
    // Cache scaled Q for reuse across keys; all accumulation uses FP32.
    float query[8];
    float accumulator[8];
    for (int i = 0; i < 8; ++i) {
        const int d = lane + 32 * i;
        query[i] = d < dim ? float(q[q_base + d]) * scale : 0.0f;
        accumulator[i] = 0.0f;
    }
    // For positions j already processed by this SIMD group, maintain:
    // maximum=m，denominator=Σ exp(score_j-m)，accumulator=Σ exp(score_j-m)*V_j。
    // The accumulator is unnormalized; each lane stores only its assigned dimensions.
    float maximum = -INFINITY;
    float denominator = 0.0f;
    // The query's absolute position is S-L+query_position; end is an exclusive bound.
    const int end = is_causal ? min(context, context - length + query_position + 1) : context;
    // Group g visits g, g+groups, ...; groups cover the valid keys without overlap.
    for (int position = group; position < end; position += groups) {
        const size_t base = kv_base + size_t(position) * dim;
        float score = 0.0f;
        for (int i = 0; i < 8; ++i) {
            const int d = lane + 32 * i;
            if (d < dim) {
                score += query[i] * float(k[base + d]);
            }
        }
        // Sum partial dot products across 32 lanes; every lane receives the full score.
        score = simd_sum(score);
        if (has_mask) {
            score += mask[size_t(row) * context + position];
        }
        // Skip masked scores before subtracting an initial -inf maximum, which would yield NaN.
        if (score == -INFINITY) {
            continue;
        }
        // Rescale the old denominator and accumulator to the new exponential reference:
        // exp(score_j-old_m) * exp(old_m-new_m) = exp(score_j-new_m)。
        const float new_maximum = max(maximum, score);
        const float old_factor = fast::exp(maximum - new_maximum);
        const float score_factor = fast::exp(score - new_maximum);
        denominator = denominator * old_factor + score_factor;
        for (int i = 0; i < 8; ++i) {
            const int d = lane + 32 * i;
            if (d < dim) {
                accumulator[i] = accumulator[i] * old_factor + score_factor * float(v[base + d]);
            }
        }
        maximum = new_maximum;
    }

    // Shared layout: [groups maxima][groups denominators][groups*D values][total denominator].
    // Publish local state, then synchronize before reading across groups.
    threadgroup float* maxima = shared;
    threadgroup float* denominators = shared + groups;
    threadgroup float* values = shared + 2 * groups;
    threadgroup float* total = values + groups * dim;
    if (lane == 0) {
        maxima[group] = maximum;
        denominators[group] = denominator;
    }
    for (int i = 0; i < 8; ++i) {
        const int d = lane + 32 * i;
        if (d < dim) {
            values[group * dim + d] = accumulator[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The first group finds the common maximum M and replaces maxima[g] with exp(m_g-M).
    // Empty or fully masked groups have zero denominators and must receive zero merge weights.
    if (group == 0) {
        const float local_max = lane < uint(groups) ? maxima[lane] : -INFINITY;
        const float common_max = simd_max(local_max);
        if (lane < uint(groups)) {
            maxima[lane] = denominators[lane] > 0.0f ? fast::exp(local_max - common_max) : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Once all factors are written, one thread computes sum_g denominator_g * exp(m_g-M).
    if (tid == 0) {
        float sum = 0.0f;
        for (int g = 0; g < groups; ++g) {
            sum += denominators[g] * maxima[g];
        }
        *total = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Once the denominator is ready, the first D threads each merge one output dimension.
    if (tid < uint(dim)) {
        float sum = 0.0f;
        for (int g = 0; g < groups; ++g) {
            sum += values[g * dim + tid] * maxima[g];
        }
        // Softmax is undefined for a fully masked row; return NaN as the Python oracle does.
        out[q_base + tid] = T(*total > 0.0f ? sum / *total : NAN);
    }
}

instantiate_kernel("week2_rms_norm_f32", week2_rms_norm, float);
instantiate_kernel("week2_rms_norm_f16", week2_rms_norm, half);
instantiate_kernel("week2_rms_norm_bf16", week2_rms_norm, bfloat16_t);
instantiate_kernel("week2_rope_f32", week2_rope, float);
instantiate_kernel("week2_rope_f16", week2_rope, half);
instantiate_kernel("week2_rope_bf16", week2_rope, bfloat16_t);
instantiate_kernel("week2_swiglu_f32", week2_swiglu, float);
instantiate_kernel("week2_swiglu_f16", week2_swiglu, half);
instantiate_kernel("week2_swiglu_bf16", week2_swiglu, bfloat16_t);
instantiate_kernel("week2_decode_attention_f32", week2_decode_attention, float);
instantiate_kernel("week2_decode_attention_f16", week2_decode_attention, half);
instantiate_kernel("week2_decode_attention_bf16", week2_decode_attention, bfloat16_t);
