#include <algorithm>
#include <stdexcept>
#include <string>

#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext {

// Week 3, Day 3.
mx::array paged_cache_update(const mx::array &pages, const mx::array &values, int page_id, int start,
                             mx::StreamOrDevice s) {
    if ((pages.dtype() != mx::float32 && pages.dtype() != mx::bfloat16) || values.dtype() != pages.dtype()) {
        throw std::runtime_error("paged_cache_update: pages and values must have the same float32 or bfloat16 dtype");
    }
    if (pages.shape().size() != 4 || values.shape().size() != 4 || values.shape()[0] != 1) {
        throw std::runtime_error(
            "paged_cache_update: expected pages [P, H, page_size, D] and values [1, H, length, D]");
    }
    if (values.shape()[1] != pages.shape()[1] || values.shape()[3] != pages.shape()[3]) {
        throw std::runtime_error("paged_cache_update: values must match the page head count and head dimension");
    }
    if (page_id < 0 || page_id >= pages.shape()[0] || start < 0 || start + values.shape()[2] > pages.shape()[2]) {
        throw std::runtime_error("paged_cache_update: destination slice is outside page storage");
    }
    return mx::array(pages.shape(), pages.dtype(), std::make_shared<PagedCacheUpdate>(to_stream(s), page_id, start),
                     {pages, values});
}

void PagedCacheUpdate::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("paged_cache_update: the course extension is GPU-only");
}

#ifdef _METAL_
void PagedCacheUpdate::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    const auto &pages = inputs[0];
    const auto &values = inputs[1];
    auto &out = outputs[0];
    if (!pages.flags().row_contiguous || !values.flags().row_contiguous) {
        throw std::runtime_error("paged_cache_update: pages and values must be contiguous");
    }

    // Page storage is request state, so the output intentionally aliases the
    // input buffer. The kernel touches only the appended slice instead of
    // building a functional copy of the entire cache tensor.
    out.copy_shared_buffer(pages);
    auto &d = mx::metal::device(stream().device);
    auto library = d.get_library("tiny_llm_ext");
    const char *kernel_name = pages.dtype() == mx::bfloat16 ? "paged_cache_update_bf16" : "paged_cache_update_f32";
    auto kernel = d.get_kernel(kernel_name, library);
    auto &compute_encoder = mx::metal::get_command_encoder(stream());
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(values, 0);
    compute_encoder.set_output_array(out, 1);
    const int heads = pages.shape()[1];
    const int length = values.shape()[2];
    const int head_dim = pages.shape()[3];
    const int page_size = pages.shape()[2];
    compute_encoder.set_bytes(heads, 2);
    compute_encoder.set_bytes(length, 3);
    compute_encoder.set_bytes(head_dim, 4);
    compute_encoder.set_bytes(page_size, 5);
    compute_encoder.set_bytes(page_id_, 6);
    compute_encoder.set_bytes(start_, 7);
    const int threads = std::min<int>(kernel->maxTotalThreadsPerThreadgroup(), 256);
    compute_encoder.dispatch_threads(MTL::Size(values.size(), 1, 1), MTL::Size(threads, 1, 1));
}
#else
void PagedCacheUpdate::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("PagedCacheUpdate has no GPU implementation.");
}
#endif

// Python validates metadata values; this boundary validates buffer layouts.
mx::array paged_attention(const mx::array &q, const mx::array &keys, const mx::array &values,
                          const mx::array &table, const mx::array &lengths, float scale,
                          bool causal, int kv_heads, int heads, mx::StreamOrDevice s) {
    if ((q.dtype() != mx::float32 && q.dtype() != mx::bfloat16) ||
        keys.dtype() != q.dtype() || values.dtype() != q.dtype()) {
        throw std::runtime_error("paged_attention: Q/K/V must share float32 or bfloat16 dtype");
    }
    if (q.ndim() != 3 || keys.ndim() != 4 || values.shape() != keys.shape() ||
        table.ndim() != 2 || lengths.ndim() != 1 || heads <= 0 || kv_heads <= 0 ||
        heads % kv_heads != 0 || q.shape()[0] <= 0 || q.shape()[0] % heads != 0 ||
        q.shape()[1] <= 0 || q.shape()[2] <= 0 || keys.shape()[1] != kv_heads ||
        keys.shape()[2] <= 0 || keys.shape()[3] != q.shape()[2] ||
        table.shape()[0] != q.shape()[0] / heads || lengths.shape()[0] != table.shape()[0]) {
        throw std::runtime_error("paged_attention: incompatible query, page, or metadata shapes");
    }
    if (table.dtype() != mx::int32 || lengths.dtype() != mx::int32) {
        throw std::runtime_error("paged_attention: metadata must be int32");
    }
    return mx::array(q.shape(), q.dtype(),
                     std::make_shared<PagedAttention>(to_stream(s), scale, causal, kv_heads, heads),
                     {q, keys, values, table, lengths});
}

void PagedAttention::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("paged_attention: the course extension is GPU-only");
}

#ifdef _METAL_
void PagedAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    for (const auto &input : inputs) {
        if (!input.flags().row_contiguous) {
            throw std::runtime_error("paged_attention: all inputs must be row-contiguous");
        }
    }
    const auto &q = inputs[0];
    const int length = q.shape()[1], dim = q.shape()[2];
    // The generic kernels keep at most 64 FP32 components per lane. The
    // fixed-size prefill staging buffers process wider heads in 128-D chunks.
    if (dim > 2048) {
        throw std::runtime_error("paged_attention: head dimension exceeds the 2048-D kernel limit");
    }
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &device = mx::metal::device(stream().device);
    // Route by this call's query length, not the amount of cached context.
    // The matrix kernel specializes both its BF16 inputs and its D=128 tiles.
    const bool short_query = length <= 8;
    const bool tiled_prefill = !short_query && q.dtype() == mx::bfloat16 && dim == 128;
    std::string name;
    if (short_query) {
        name = q.dtype() == mx::bfloat16 ? "paged_attention_decode_bf16" : "paged_attention_decode_f32";
        if (q.dtype() == mx::bfloat16 && dim == 128) {
            // Only this exact shape uses a compile-time D=128 specialization.
            name += "_d128";
        } else if (dim > 256) {
            // Stream 128 output dimensions per pass for wide heads. Keeping
            // bounded register arrays preserves the required 1024-thread
            // launch even when a whole-head register allocation is too large.
            name += "_wide";
        } else if (dim > 128) {
            name += "_max256";
        }
    } else if (tiled_prefill) {
        name = "paged_attention_mma_bf16_d128";
    } else {
        // Keep the Day 4 schedule for other dtypes and head dimensions.
        name = q.dtype() == mx::bfloat16 ? "paged_attention_scalar_bf16" : "paged_attention_scalar_f32";
    }
    auto kernel = device.get_kernel(name, device.get_library("tiny_llm_ext"));
    const int threads = short_query ? 32 * 32 : (tiled_prefill ? 8 * 32 : 16 * 32);
    if (kernel->maxTotalThreadsPerThreadgroup() < threads) {
        throw std::runtime_error("paged_attention: device cannot launch the required threadgroup size");
    }
    auto &encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < 5; ++i) encoder.set_input_array(inputs[i], i);
    encoder.set_output_array(out, 5);
    const int page_size = inputs[1].shape()[2], width = inputs[3].shape()[1];
    const int causal = is_causal_;
    encoder.set_bytes(length, 6);
    encoder.set_bytes(dim, 7);
    encoder.set_bytes(page_size, 8);
    encoder.set_bytes(width, 9);
    encoder.set_bytes(num_heads_, 10);
    encoder.set_bytes(num_kv_heads_, 11);
    encoder.set_bytes(scale_, 12);
    encoder.set_bytes(causal, 13);
    if (short_query) {
        // 32 context partitions share one reusable 32x32 transpose tile and
        // 32 maxima/sums: 4,352 bytes, independent of D and context length.
        constexpr int groups = 32;
        encoder.set_threadgroup_memory_length((groups * 32 + 2 * groups) * sizeof(float), 0);
        encoder.dispatch_threadgroups(MTL::Size(q.shape()[0] * length, 1, 1), MTL::Size(threads, 1, 1));
    } else if (tiled_prefill) {
        // Eight SIMD groups own eight query rows each. The kernel declares
        // its BF16 Q/K/V staging statically; it needs no dynamic scratch.
        // Grid x selects a flattened (batch, query head); grid y selects a
        // 64-row Q tile. Each threadgroup streams the corresponding KV head.
        encoder.dispatch_threadgroups(MTL::Size(q.shape()[0], (length + 63) / 64, 1),
                                      MTL::Size(threads, 1, 1));
    } else {
        // Sixteen query rows reuse a 32-key tile. Q/K/V staging is bounded
        // to 128 dimensions per pass; the same K buffer is reused for V.
        constexpr int query_rows = 16, key_rows = 32, dimensions = 128;
        const size_t scratch_bytes =
            (query_rows * dimensions + key_rows * dimensions + query_rows * key_rows) * sizeof(float) +
            key_rows * sizeof(int);
        encoder.set_threadgroup_memory_length(scratch_bytes, 0);
        encoder.dispatch_threadgroups(MTL::Size(q.shape()[0], (length + query_rows - 1) / query_rows, 1),
                                      MTL::Size(threads, 1, 1));
    }
}
#else
void PagedAttention::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    throw std::runtime_error("paged_attention: Metal unavailable");
}
#endif

}  // namespace tiny_llm_ext
