#pragma once

#include "ggml-backend.h"

#include <cstdint>
#include <memory>

struct llama_model;

// hybrid MoE for host-resident experts (LLAMA_MOE_VRAM_MIB): the experts that generation uses most are kept on the device, and in
// generation (small batches) the device computes them while the CPU computes the others at the same time
//  - per layer, a slot map (expert -> slot of its group's banks, -1 if not cached) on the device, mirrored in host memory
//  - the device MUL_MAT_IDs read the banks with the mapped ids, -1 gives zero rows (mmvq); the CPU MUL_MAT_IDs skip the cached
//    experts (ggml_moe_stream_set_gpu_map), so each expert's output comes from exactly one side and the two are added
//  - between graphs, the experts with the highest decayed use counts replace the coldest cached ones; a background thread
//    copies them from the RAM expert cache to the device, they are mapped once their copy is done
//  - shared mode (default): the banks live at the end of the scheduler's device compute buffer, which prompt batches need but
//    generation leaves unused; a graph whose allocations reach into the banks first drops the experts it overwrites, which are
//    copied again at the next generation step
class llama_moe_vram {
public:
    // shared: the compute buffer of backend to place the banks in (size is then their share of it), nullptr for an own buffer
    llama_moe_vram(const llama_model & model, ggml_backend_t backend, ggml_backend_buffer_type_t buft, size_t size,
                   ggml_backend_buffer_t shared);
    ~llama_moe_vram();

    ggml_backend_t backend() const;

    // whether layer il computes its cached experts on the device for a batch of n_tokens
    bool active(int32_t il, int64_t n_tokens) const;

    // the slot map of layer il on the device: I32 [1, n_expert]
    ggml_tensor * get_slot_map(int32_t il) const;

    // the banks of w: [ne0, ne1, n_slots], nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // CPU op that passes the ids (a) through and records the experts used by layer il (b is only read to have the
    // scheduler copy it to the CPU in this split, before the device branch starts)
    static void pre_op(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * userdata);
    // the same, also reading c (the routing weights, which the CPU's weighted sum of its experts needs)
    static void pre_op3(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c, int ith, int nth, void * userdata);
    void * pre_userdata(int32_t il);

    // before computing a graph of n_tokens that was allocated by sched: drops the experts its allocations overwrite (shared mode),
    // maps the experts whose copies are done and starts the next copies
    void begin_graph(ggml_backend_sched_t sched, int64_t n_tokens);

    // prompt batches that are offloaded (>= min_batch tokens) also use [offset, offset + size) of the compute buffer (the ring of the
    // streamed experts, mmid-stream.cu)
    void set_prefill_ring(int64_t min_batch, size_t offset, size_t size);

    // VRAM manager (LLAMA_VRAM_MANAGER=1): before a graph of another context (an MTP draft) computes, maps what it needs of its
    // lazily backed compute buffer (and its prompt ring), taking the memory from the banks of the managing instance if the device
    // lacks it; without a managing instance (no banks) it maps what the graph needs
    static void manage_other(ggml_backend_sched_t sched, ggml_backend_t backend, int64_t n_tokens, size_t ring_off, size_t ring_size,
                             int64_t ring_min_batch);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
    static impl * owner; // the instance that manages the device memory
};

using llama_moe_vram_ptr = std::unique_ptr<llama_moe_vram>;
