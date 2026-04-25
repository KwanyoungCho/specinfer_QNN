#ifndef GGML_OPENCL_H
#define GGML_OPENCL_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

//
// backend API
//
GGML_BACKEND_API ggml_backend_t ggml_backend_opencl_init(void);
GGML_BACKEND_API bool ggml_backend_is_opencl(ggml_backend_t backend);

// Returns true when the backend can gather rows from a source Q4_0 tensor that
// already lives in the Adreno-optimized OpenCL layout into a destination Q4_0
// tensor with dst_rows rows. dst_rows must match the destination tensor row
// count that will later be passed to ggml_backend_opencl_gather_rows_q4_0().
GGML_BACKEND_API bool ggml_backend_opencl_supports_gather_rows_q4_0(
        ggml_backend_t backend,
        const struct ggml_tensor * src,
        int32_t dst_rows);

// Gather Q4_0 rows directly on the OpenCL device from src into dst using the
// row indices supplied in row_indices. Negative row indices produce zero-filled
// destination rows. Both src and dst must be OpenCL Q4_0 tensors compatible
// with ggml_backend_opencl_supports_gather_rows_q4_0().
GGML_BACKEND_API bool ggml_backend_opencl_gather_rows_q4_0(
        ggml_backend_t backend,
        const struct ggml_tensor * src,
        const int32_t * row_indices,
        int32_t n_rows,
        struct ggml_tensor * dst);

// Returns true when the backend can run the exact F32 top-k helper
// implemented on top of the OpenCL sort kernels.
GGML_BACKEND_API bool ggml_backend_opencl_supports_top_k_f32(
        ggml_backend_t backend);

// Computes the exact top-k indices for the F32 score array using OpenCL. out_indices must have room for at least
// min(n_scores, top_k) elements. Returned indices are sorted by descending
// score, with lower indices winning ties.
GGML_BACKEND_API bool ggml_backend_opencl_top_k_f32(
        ggml_backend_t backend,
        const float * scores,
        int32_t n_scores,
        int32_t top_k,
        int32_t * out_indices);

// Computes exact top-k indices on the OpenCL device and returns them in a
// backend-owned device buffer. The returned buffer contains min(n_scores, top_k)
// int32 indices in descending score order. Release it with
// ggml_backend_opencl_device_i32_buffer_free().
GGML_BACKEND_API bool ggml_backend_opencl_top_k_f32_to_device(
        ggml_backend_t backend,
        const float * scores,
        int32_t n_scores,
        int32_t top_k,
        void ** out_device_indices);

// Returns true when the backend can run the F32 softmax-probability threshold
// helper. The selected indices are emitted in ascending token-id order.
GGML_BACKEND_API bool ggml_backend_opencl_supports_softmax_threshold_f32(
        ggml_backend_t backend);

// Selects indices where softmax(scores)[i] >= threshold using OpenCL. max_count
// caps the emitted indices. out_count receives the emitted count; out_total_above
// receives the uncapped number of indices above threshold. If no index passes the
// threshold, the argmax index is emitted and out_total_above is 0. Release the
// returned buffer with ggml_backend_opencl_device_i32_buffer_free().
GGML_BACKEND_API bool ggml_backend_opencl_softmax_threshold_f32_to_device(
        ggml_backend_t backend,
        const float * scores,
        int32_t n_scores,
        float threshold,
        int32_t max_count,
        void ** out_device_indices,
        int32_t * out_count,
        int32_t * out_total_above);

// Copies count int32 values from a backend-owned device buffer into host memory.
GGML_BACKEND_API bool ggml_backend_opencl_device_i32_buffer_copy_to_host(
        ggml_backend_t backend,
        void * device_buffer,
        int32_t count,
        int32_t * out_values);

// Fills count int32 values in a backend-owned device buffer, starting at offset.
GGML_BACKEND_API bool ggml_backend_opencl_device_i32_buffer_fill(
        ggml_backend_t backend,
        void * device_buffer,
        int32_t offset,
        int32_t count,
        int32_t value);

// Uploads count int32 values into a backend-owned device buffer.
GGML_BACKEND_API bool ggml_backend_opencl_device_i32_buffer_from_host(
        ggml_backend_t backend,
        const int32_t * values,
        int32_t count,
        void ** out_device_buffer);

// Sorts the first count int32 values in a backend-owned device buffer in
// ascending order. The buffer contents are updated in place.
GGML_BACKEND_API bool ggml_backend_opencl_device_i32_buffer_sort_asc_inplace(
        ggml_backend_t backend,
        void * device_buffer,
        int32_t count);

// Releases a backend-owned int32 device buffer previously returned by one of
// the helper APIs above.
GGML_BACKEND_API void ggml_backend_opencl_device_i32_buffer_free(
        ggml_backend_t backend,
        void * device_buffer);

// Gather Q4_0 rows directly on the OpenCL device using an already-resident i32
// device buffer of row indices.
GGML_BACKEND_API bool ggml_backend_opencl_gather_rows_q4_0_device_i32(
        ggml_backend_t backend,
        const struct ggml_tensor * src,
        void * device_row_indices,
        int32_t n_rows,
        struct ggml_tensor * dst);

// Gather Q4_0 rows from a device i32 buffer with implicit padding. The first
// selected_rows destination rows read device_row_indices[0..selected_rows);
// remaining rows up to n_rows copy pad_row_index. This avoids materializing a
// padded ids buffer on the host when the selector already produced device ids.
GGML_BACKEND_API bool ggml_backend_opencl_gather_rows_q4_0_device_i32_padded(
        ggml_backend_t backend,
        const struct ggml_tensor * src,
        void * device_row_indices,
        int32_t selected_rows,
        int32_t n_rows,
        int32_t pad_row_index,
        struct ggml_tensor * dst);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type(void);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_opencl_reg(void);

#ifdef  __cplusplus
}
#endif

#endif // GGML_OPENCL_H
