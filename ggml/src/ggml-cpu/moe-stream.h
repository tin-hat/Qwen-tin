#pragma once

#include "ggml-cpu.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct ggml_cgraph;

// MoE expert streaming for models whose mmapped experts do not fit in RAM (enabled with GGML_MOE_STREAM_MIB=<cache size>)
// the experts used by MUL_MAT_ID are kept in an LRU cache in anonymous memory and the missing ones are read with O_DIRECT,
// so the page cache, its readahead and its reclaim are bypassed

// ggml_moe_stream_enabled and ggml_moe_stream_copy are declared in ggml-cpu.h

// called by thread 0 before the first MUL_MAT_ID of a set of ids: loads the selected experts of every MUL_MAT_ID in cgraph that uses these ids
void ggml_moe_stream_prepare(const struct ggml_cgraph * cgraph, int node_n);

// false while the read of expert e of w is in flight
bool ggml_moe_stream_ready(const struct ggml_tensor * w, int64_t e);

// the data of expert e of w, waiting for its read: the cached copy if it was prepared, otherwise the original data
const char * ggml_moe_stream_expert(const struct ggml_tensor * w, int64_t e);

// the map of ggml_moe_stream_set_gpu_map for w (works without streaming too), NULL if none
const int32_t * ggml_moe_stream_gpu_map(const struct ggml_tensor * w);

#ifdef __cplusplus
}
#endif
