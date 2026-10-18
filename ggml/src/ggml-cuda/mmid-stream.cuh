#pragma once

#include "common.cuh"

// MUL_MAT_ID whose expert weights are in host memory (prompt batches with streamed MoE experts, see mmid-stream.cu)
bool ggml_cuda_mul_mat_id_streamed(const ggml_tensor * dst);
void ggml_cuda_mul_mat_id_stream(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// source: bool (*)(const ggml_tensor * w, const int32_t * counts, set, sync, void * ctx) as ggml_moe_stream_copy;
// ring: device memory for n_buf chunk buffers (ring_size bytes)
void ggml_backend_cuda_moe_stream_set(void * source, void * ring, size_t ring_size, int n_buf);
// a new graph: the selected experts read for the previous one are stale
void ggml_backend_cuda_moe_stream_new_graph(void);
