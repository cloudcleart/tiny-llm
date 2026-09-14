#pragma once

#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

namespace tiny_llm {

// Cooperative contiguous chunks with an unguarded full-tile path and a
// zero-filling edge path. Transposition affects only the shared-memory store.
template <typename T, int ROWS, int COLS, int DESTINATION_STRIDE, int THREADS, bool TRANSPOSE_DESTINATION = false,
          bool COPY_16_BYTES = false>
struct CooperativeTileLoader {
    static_assert((ROWS * COLS) % THREADS == 0);
    static_assert(COLS % ((ROWS * COLS) / THREADS) == 0);
    static_assert(!COPY_16_BYTES || (!TRANSPOSE_DESTINATION && sizeof(T) * ((ROWS * COLS) / THREADS) == 16));

    struct alignas(sizeof(T)) Read16Bytes {
        uint8_t values[16];
    };

    static METAL_FUNC void load(device const T *source, int source_stride, threadgroup T *destination,
                                uint thread_index, int valid_rows = ROWS, int valid_columns = COLS) {
        constexpr int chunk = ROWS * COLS / THREADS;
        const int row = thread_index * chunk / COLS;
        const int column = thread_index * chunk % COLS;
        if (valid_rows == ROWS && valid_columns == COLS) {
            if constexpr (COPY_16_BYTES) {
                // Adjacent threads own adjacent 16-byte chunks in each row.
                *reinterpret_cast<threadgroup Read16Bytes *>(
                    destination + row * DESTINATION_STRIDE + column) =
                    *reinterpret_cast<device const Read16Bytes *>(
                        source + row * source_stride + column);
            } else {
#pragma unroll
                for (int i = 0; i < chunk; ++i) {
                    const int target = TRANSPOSE_DESTINATION
                        ? (column + i) * DESTINATION_STRIDE + row
                        : row * DESTINATION_STRIDE + column + i;
                    destination[target] = source[row * source_stride + column + i];
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < chunk; ++i) {
                const int target = TRANSPOSE_DESTINATION
                    ? (column + i) * DESTINATION_STRIDE + row
                    : row * DESTINATION_STRIDE + column + i;
                destination[target] = row < valid_rows && column + i < valid_columns
                    ? source[row * source_stride + column + i] : T(0);
            }
        }
    }
};

METAL_FUNC ushort2 course_matrix_coordinate(ushort lane) {
    return ushort2(
        (lane & 1) * 2 + (lane & 8) / 2,
        (lane & 7) / 2 + (lane & 16) / 4);
}

template <typename T>
METAL_FUNC void course_load_matrix(
    thread simdgroup_matrix<T, 8, 8>& matrix,
    threadgroup const T* source,
    int row_stride,
    ushort lane) {
    const ushort2 coordinate = course_matrix_coordinate(lane);
    matrix.thread_elements()[0] = source[coordinate.y * row_stride + coordinate.x];
    matrix.thread_elements()[1] =
        source[coordinate.y * row_stride + coordinate.x + 1];
}

// Load an 8x8 fragment while viewing row-major [output, reduction] storage as
// the transposed [reduction, output] right-hand matrix.
template <typename T>
METAL_FUNC void course_load_transposed_matrix(
    thread simdgroup_matrix<T, 8, 8>& matrix,
    threadgroup const T* source,
    int row_stride,
    ushort lane) {
    const ushort2 coordinate = course_matrix_coordinate(lane);
    matrix.thread_elements()[0] = source[coordinate.x * row_stride + coordinate.y];
    matrix.thread_elements()[1] =
        source[(coordinate.x + 1) * row_stride + coordinate.y];
}

// Four simdgroups cover the four 16x16 quadrants of one 32x32 output tile.
// The wrapper only exposes the two operations the course kernel needs: add a
// 32-wide reduction tile, then store the accumulated result with edge guards.
template <typename T, typename OutT, int TILE_STRIDE>
struct CooperativeBlockMMA {
    simdgroup_matrix<float, 8, 8> accumulators[2][2];
    ushort simdgroup_index;
    ushort lane_index;

    METAL_FUNC CooperativeBlockMMA(ushort simdgroup, ushort lane)
        : simdgroup_index(simdgroup), lane_index(lane) {
        #pragma unroll
        for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
            #pragma unroll
            for (int column_fragment = 0; column_fragment < 2; ++column_fragment) {
                accumulators[row_fragment][column_fragment].thread_elements() = 0.0f;
            }
        }
    }

    METAL_FUNC void multiply_accumulate(
        threadgroup const T* left_tile,
        threadgroup const T* right_tile) {
        const int simdgroup_row = simdgroup_index / 2;
        const int simdgroup_column = simdgroup_index % 2;

        #pragma unroll
        for (int reduction_fragment = 0; reduction_fragment < 4;
             ++reduction_fragment) {
            simdgroup_matrix<T, 8, 8> left[2];
            simdgroup_matrix<T, 8, 8> right[2];

            #pragma unroll
            for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
                const int left_row = simdgroup_row * 16 + row_fragment * 8;
                course_load_matrix(
                    left[row_fragment],
                    left_tile + left_row * TILE_STRIDE + reduction_fragment * 8,
                    TILE_STRIDE,
                    lane_index);
            }

            #pragma unroll
            for (int column_fragment = 0; column_fragment < 2;
                 ++column_fragment) {
                const int right_row =
                    simdgroup_column * 16 + column_fragment * 8;
                course_load_transposed_matrix(
                    right[column_fragment],
                    right_tile + right_row * TILE_STRIDE + reduction_fragment * 8,
                    TILE_STRIDE,
                    lane_index);
            }

            #pragma unroll
            for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
                #pragma unroll
                for (int column_fragment = 0; column_fragment < 2;
                     ++column_fragment) {
                    simdgroup_multiply_accumulate(
                        accumulators[row_fragment][column_fragment],
                        left[row_fragment],
                        right[column_fragment],
                        accumulators[row_fragment][column_fragment]);
                }
            }
        }
    }

    METAL_FUNC void store_result_safe(
        device OutT* output,
        int output_stride,
        short2 valid_shape) const {
        const int simdgroup_row = simdgroup_index / 2;
        const int simdgroup_column = simdgroup_index % 2;
        const ushort2 coordinate = course_matrix_coordinate(lane_index);

        #pragma unroll
        for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
            const int row =
                simdgroup_row * 16 + row_fragment * 8 + coordinate.y;
            if (row >= valid_shape.y) {
                continue;
            }
            #pragma unroll
            for (int column_fragment = 0; column_fragment < 2;
                 ++column_fragment) {
                const int column =
                    simdgroup_column * 16 + column_fragment * 8 + coordinate.x;
                if (column < valid_shape.x) {
                    output[row * output_stride + column] = OutT(
                        accumulators[row_fragment][column_fragment]
                            .thread_elements()[0]);
                }
                if (column + 1 < valid_shape.x) {
                    output[row * output_stride + column + 1] = OutT(
                        accumulators[row_fragment][column_fragment]
                            .thread_elements()[1]);
                }
            }
        }
    }
};

} // namespace tiny_llm
