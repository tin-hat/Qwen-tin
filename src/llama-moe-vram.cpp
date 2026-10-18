#include "llama-moe-vram.h"
#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

static int llama_moe_vram_env_int(const char * name, int def) {
    const char * s = getenv(name);
    return s != nullptr ? atoi(s) : def;
}

struct llama_moe_vram::impl {
    // layers with the same expert tensor layout share the banks of a group
    struct group {
        std::vector<ggml_tensor *> ref;   // gate, up, down of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t  host_bytes = 0;
        int32_t n_slots    = 0;
        std::vector<int32_t> key_of;      // [n_slots] il*n_expert + e, -1 if free
        std::vector<size_t>  slot_low;    // [n_slots] shared mode: lowest buffer offset of the slot's data
        std::vector<uint8_t> usable;      // [n_slots] 0 if generation graphs need the slot's memory (managed: not mapped)
        std::vector<int32_t> free_slots;  // usable and empty, the highest offsets are taken first
        int32_t n_cap       = 0;          // managed: slots [0, n_cap) are mapped
        size_t  slot_bytes  = 0;          // managed: bytes of one slot over the group's banks
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        std::vector<ggml_tensor *> experts; // host gate, up, down
        ggml_tensor * slot_map = nullptr;   // device view [1, n_expert]
    };

    struct pre_ud {
        impl *  self;
        int32_t il;
    };

    struct job {
        int32_t il;
        int32_t e;
        int32_t slot;
    };

    ggml_backend_t backend;
    ggml_backend_t backend_up = nullptr; // a second context on the same device: the copies run on its own stream

    int32_t n_layer  = 0;
    int32_t n_expert = 0;
    int64_t max_tokens;

    // ---- managed mode (LLAMA_VRAM_MANAGER=1, 2026-10-15): the banks are a lazily backed buffer of their own; the device memory that
    // the KV caches (at their current length) and the compute buffer (for the current graph) do not need backs bank slots. Per group
    // the usable slots are a prefix [0, n_cap) of its banks. Prompt batches take the compute buffer's memory back (the experts in
    // the slots they need are dropped and copied again afterwards), longer contexts take the KV caches' share.
    bool   managed = false;
    bool   (*vmap)(ggml_backend_buffer_t, size_t, size_t)   = nullptr;
    size_t (*vunmap)(ggml_backend_buffer_t, size_t, size_t) = nullptr;
    size_t (*vmapped)(ggml_backend_buffer_t)                = nullptr;
    size_t gran            = 2u << 20;
    size_t host_bytes_all  = 0;
    // per compute buffer (this context's and those of other contexts, e.g. an MTP draft, see manage_other)
    struct cbuf_state {
        size_t small_hwm   = 0;      // the largest end of the small graphs so far
        size_t hi          = 0;      // [0, hi) is mapped
        bool   ring_mapped = false;
        size_t ring_lo = 0, ring_hi = 0;
        ggml_backend_t be  = nullptr; // the backend that computes with it
    };

    // the compute buffers of other contexts (an MTP draft and its target) never compute at the same time: before the banks give
    // memory, what the others mapped for prompt batches beyond their generation graphs goes back (their rings stay)
    size_t reclaim_others(ggml_backend_buffer_t self, size_t bytes) {
        size_t freed = 0;
        for (auto & [b, st] : cbufs) {
            if (b == self || st.be == nullptr || freed >= bytes) {
                continue;
            }
            const size_t keep = std::min(st.hi, GGML_PAD(st.small_hwm, gran));
            if (st.hi <= keep) {
                continue;
            }
            ggml_backend_synchronize(st.be);
            ggml_backend_synchronize(backend);
            if (st.ring_mapped && st.ring_lo < st.hi && st.ring_hi > keep) {
                // around the ring, which other contexts stream through
                if (st.ring_lo > keep) {
                    freed += vunmap(b, keep, st.ring_lo - keep);
                }
                if (st.hi > st.ring_hi) {
                    freed += vunmap(b, st.ring_hi, st.hi - st.ring_hi);
                }
            } else {
                freed += vunmap(b, keep, st.hi - keep);
            }
            st.hi = keep;
        }
        return freed;
    }
    std::unordered_map<ggml_backend_buffer_t, cbuf_state> cbufs;
    std::unordered_map<uint64_t, uint32_t> kv_cells; // cells mapped per KV cache (llama_kv_cache::vram_id)
    int    grow_wait       = 0;       // graphs to wait before growing the banks again
    uint64_t n_shrinks = 0, n_grows = 0, n_moved = 0;

    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> bindings; // host experts -> banks
    std::vector<pre_ud> uds;

    ggml_context_ptr        ctx_banks;
    ggml_context_ptr        ctx_map;
    ggml_backend_buffer_ptr buf_banks;           // own buffer, if not shared
    ggml_backend_buffer_t   shared = nullptr;    // the compute buffer the banks are in
    size_t                  region_lo = 0;       // shared: offset of the banks in it
    ggml_backend_buffer_ptr buf_map;
    ggml_tensor *           map_dev = nullptr;   // I32 [n_expert, n_layer]
    ggml_backend_buffer_ptr buf_host;
    int32_t *               map_host = nullptr;  // what the graphs use: slot or -1
    bool                    disabled = false;
    bool                    after_large = false; // the last graph was a prompt batch
    int64_t                 ring_min_batch = 0;  // batches of at least this many tokens use [ring_lo, ring_hi) of the buffer (0: none)
    size_t                  ring_lo = 0, ring_hi = 0;
    size_t (*trim_pools)(ggml_backend_t) = nullptr; // ggml_backend_cuda_trim_backend_pools, if the backend has it

    // policy: decayed use counts, an expert is copied once it is among the most used
    std::vector<float>   score;   // [n_layer*n_expert]
    std::vector<uint8_t> state;   // 0: not cached, 1: being copied, 2: cached
    std::vector<int32_t> slot_of; // slot while being copied or cached
    std::vector<uint16_t> used;   // tokens of the last graph that used the expert, see pre_op
    std::vector<int32_t> used_list;
    int64_t graph_tokens = 1;     // tokens of the last graph: decay, scores and swaps are per token, so that verify batches of
                                  // k tokens adapt the cache as fast as k single-token steps (2026-10-15: with 8 swaps per graph,
                                  // 4-token steps kept 24% of the expert work on the GPU instead of 41%)
    std::vector<int32_t> cands;
    std::vector<int32_t> refill;  // cached experts dropped for a prompt batch, copied again at the next generation step
    float scale = 1.0f;
    float decay;
    int   max_swaps;

    // copies
    std::thread th;
    std::mutex mtx;
    std::condition_variable cv;
    std::condition_variable cv_idle;
    std::deque<job> todo;
    std::vector<std::pair<job, bool>> done;
    bool stop = false;
    bool busy = false;
    ggml_backend_buffer_ptr staging;

    // stats
    bool     log;
    uint64_t n_graphs = 0, n_uses = 0, n_hits = 0, n_copies = 0, n_failed = 0, n_evicted = 0, n_dropped = 0;
    uint64_t last_uses = 0, last_hits = 0;

    impl(const llama_model & model, ggml_backend_t backend, ggml_backend_buffer_type_t buft, size_t size, ggml_backend_buffer_t shared_buf) :
            backend(backend),
            max_tokens(llama_moe_vram_env_int("LLAMA_MOE_VRAM_MAX_TOKENS", 1)),
            decay(0.995f),
            max_swaps(8),
            log(true) {
        n_layer  = (int32_t) model.layers.size();
        n_expert = (int32_t) model.hparams.n_expert;
        layers.resize(n_layer);

        auto is_host_weight = [](const ggml_tensor * t) {
            return t != nullptr && t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer);
        };
        auto same_layout = [](const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
            for (size_t i = 0; i < a.size(); ++i) {
                if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
                    return false;
                }
            }
            return true;
        };

        size_t host_bytes = 0;
        for (int32_t il = 0; il < n_layer; ++il) {
            const llama_layer & l = model.layers[il];
            if (l.ffn_gate_up_exps != nullptr || l.ffn_gate_exps_s || l.ffn_up_exps_s || l.ffn_down_exps_s ||
                    !is_host_weight(l.ffn_gate_exps) || !is_host_weight(l.ffn_up_exps) || !is_host_weight(l.ffn_down_exps) ||
                    l.ffn_gate_exps->ne[2] != n_expert) {
                continue;
            }
            std::vector<ggml_tensor *> experts = { l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps };
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->ref = experts;
            }
            it->layers.push_back(il);
            layers[il].experts = experts;
            for (const ggml_tensor * t : experts) {
                it->host_bytes += ggml_nbytes(t);
                host_bytes     += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            throw std::runtime_error("hybrid MoE: no layer keeps its experts in host memory");
        }

        // shared: the banks take the end of the compute buffer, generation graphs keep at least LLAMA_MOE_VRAM_KEEP_MIB at its start
        shared = shared_buf;
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        if (shared) {
            const size_t buf_size = ggml_backend_buffer_get_size(shared);
            const size_t keep     = (size_t) 64 << 20;
            size = std::min(size, buf_size > keep ? buf_size - keep : 0);
        }

        // split the budget by the size of the experts, so each group caches the same share of its experts
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };
        size_t n_bank_tensors = 0;
        size_t max_slot_bytes = 0;
        size_t total = 0;
        for (group & g : groups) {
            const size_t budget = (size_t) ((double) size*g.host_bytes/host_bytes);
            const int32_t max_slots = (int32_t) g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots > 0) {
                total += alloc_size(g, g.n_slots);
            }
            n_bank_tensors += g.ref.size()*(1 + g.layers.size());
            size_t slot_bytes = 0;
            for (const ggml_tensor * t : g.ref) {
                slot_bytes += GGML_PAD(t->nb[2], 64);
            }
            max_slot_bytes = std::max(max_slot_bytes, slot_bytes);
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("hybrid MoE: failed to create a context");
            }
            return res;
        };
        ctx_banks = init_ctx(n_bank_tensors + 1);
        ctx_map   = init_ctx(1 + n_layer);

        map_dev = ggml_new_tensor_2d(ctx_map.get(), GGML_TYPE_I32, n_expert, n_layer);
        ggml_set_name(map_dev, "moe_vram.slot_maps");
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx_banks.get(), t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_vram.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig = (int32_t) ig;
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx_banks.get(), bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_vram.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = cached;
                }
                l.slot_map = ggml_view_2d(ctx_map.get(), map_dev, 1, n_expert, sizeof(int32_t), (size_t) il*n_expert*sizeof(int32_t));
                ggml_format_name(l.slot_map, "moe_vram.slot_map-%d", il);
            }
        }

        if (shared) {
            // place the banks at the end of the compute buffer, the views follow their banks
            char * base = (char *) ggml_backend_buffer_get_base(shared);
            total = 0;
            for (const group & g : groups) {
                for (const ggml_tensor * bank : g.banks) {
                    total += GGML_PAD(ggml_backend_buffer_get_alloc_size(shared, bank), alignment);
                }
            }
            GGML_ASSERT(total <= ggml_backend_buffer_get_size(shared));
            region_lo = (ggml_backend_buffer_get_size(shared) - total) & ~(alignment - 1);
            size_t off = region_lo;
            for (group & g : groups) {
                for (ggml_tensor * bank : g.banks) {
                    GGML_ASSERT(ggml_backend_tensor_alloc(shared, bank, base + off) == GGML_STATUS_SUCCESS);
                    off += GGML_PAD(ggml_backend_buffer_get_alloc_size(shared, bank), alignment);
                }
            }
            GGML_ASSERT(off <= ggml_backend_buffer_get_size(shared));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_banks.get()); t != nullptr; t = ggml_get_next_tensor(ctx_banks.get(), t)) {
                if (t->view_src != nullptr) {
                    GGML_ASSERT(ggml_backend_view_init(t) == GGML_STATUS_SUCCESS);
                }
            }
        } else {
            ggml_backend_dev_t dev0 = ggml_backend_get_device(backend);
            if (ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev0)) {
                vmap    = (bool (*)(ggml_backend_buffer_t, size_t, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_buffer_map");
                vunmap  = (size_t (*)(ggml_backend_buffer_t, size_t, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_buffer_unmap");
                vmapped = (size_t (*)(ggml_backend_buffer_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_buffer_mapped");
            }
            const char * mgr = getenv("LLAMA_VRAM_MANAGER");
            managed = mgr != nullptr && atoi(mgr) != 0 && vmap && vunmap && vmapped;
            if (managed) {
                // each bank starts on a granule, so that its slots are mapped and unmapped without touching another bank
                size_t tot = 0;
                for (const group & g : groups) {
                    for (const ggml_tensor * bank : g.banks) {
                        tot += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, bank), gran);
                    }
                }
                buf_banks.reset(ggml_backend_buft_alloc_buffer(buft, tot));
                if (!buf_banks) {
                    throw std::runtime_error("hybrid MoE: failed to reserve the device buffer");
                }
                char * base = (char *) ggml_backend_buffer_get_base(buf_banks.get());
                size_t off = 0;
                for (group & g : groups) {
                    for (ggml_tensor * bank : g.banks) {
                        GGML_ASSERT(ggml_backend_tensor_alloc(buf_banks.get(), bank, base + off) == GGML_STATUS_SUCCESS);
                        off += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, bank), gran);
                    }
                }
                for (ggml_tensor * t = ggml_get_first_tensor(ctx_banks.get()); t != nullptr; t = ggml_get_next_tensor(ctx_banks.get(), t)) {
                    if (t->view_src != nullptr) {
                        GGML_ASSERT(ggml_backend_view_init(t) == GGML_STATUS_SUCCESS);
                    }
                }
                total = tot;
            } else {
                buf_banks.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_banks.get(), buft));
            }
            if (!buf_banks) {
                throw std::runtime_error("hybrid MoE: failed to allocate the device buffer");
            }
            // as weights, the ops that read the banks run on the device
            ggml_backend_buffer_set_usage(buf_banks.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        buf_map.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_map.get(), buft));
        if (!buf_map) {
            throw std::runtime_error("hybrid MoE: failed to allocate the slot maps");
        }
        ggml_backend_buffer_set_usage(buf_map.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        // the lowest offset of each slot's data, and the free slots
        host_bytes_all = host_bytes;
        for (group & g : groups) {
            g.key_of.assign(g.n_slots, -1);
            g.slot_low.assign(g.n_slots, 0);
            g.usable.assign(g.n_slots, managed ? 0 : 1);
            for (const ggml_tensor * t : g.ref) {
                g.slot_bytes += t->nb[2];
            }
            for (int32_t s = 0; s < g.n_slots && shared; ++s) {
                size_t low = SIZE_MAX;
                for (const ggml_tensor * bank : g.banks) {
                    low = std::min(low, (size_t) ((const char *) bank->data - (const char *) ggml_backend_buffer_get_base(shared)) + (size_t) s*bank->nb[2]);
                }
                g.slot_low[s] = low;
            }
            for (int32_t s = 0; s < g.n_slots && !managed; ++s) {
                g.free_slots.push_back(s);
            }
        }

        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_buffer_type_t buft_host = ggml_backend_dev_host_buffer_type(dev);
        if (buft_host == nullptr) {
            buft_host = ggml_backend_cpu_buffer_type();
        }
        buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, (size_t) n_layer*n_expert*sizeof(int32_t)));
        staging.reset(ggml_backend_buft_alloc_buffer(buft_host, max_slot_bytes));
        if (!buf_host || !staging) {
            throw std::runtime_error("hybrid MoE: failed to allocate the host buffers");
        }
        map_host = (int32_t *) ggml_backend_buffer_get_base(buf_host.get());
        std::fill(map_host, map_host + (size_t) n_layer*n_expert, -1);
        ggml_backend_tensor_set(map_dev, map_host, 0, ggml_nbytes(map_dev));

        const size_t n_all = (size_t) n_layer*n_expert;
        score.assign(n_all, 0.0f);
        state.assign(n_all, 0);
        slot_of.assign(n_all, -1);
        used.assign(n_all, 0);
        uds.resize(n_layer);
        for (int32_t il = 0; il < n_layer; ++il) {
            uds[il] = { this, il };
            if (layers[il].ig >= 0) {
                for (const ggml_tensor * w : layers[il].experts) {
                    ggml_moe_stream_set_gpu_map(w, map_host + (size_t) il*n_expert);
                }
            }
        }

        if (ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev)) {
            trim_pools = (size_t (*)(ggml_backend_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_trim_backend_pools");
        }

        backend_up = ggml_backend_dev_init(dev, nullptr);
        if (backend_up == nullptr) {
            throw std::runtime_error("hybrid MoE: failed to create the copy backend");
        }
        th = std::thread([this] { worker(); });
        if (managed && llama_moe_vram::owner == nullptr) {
            llama_moe_vram::owner = this;
        }

        fprintf(stderr, "llama_moe_vram: hybrid MoE cache of %.2f MiB %s for %.2f MiB of host experts, batches up to %lld tokens, %d swaps per graph, decay %.4f\n",
            total/1048576.0, shared ? "at the end of the compute buffer" : managed ? "managed (up to; backed by what the KV caches and the compute buffer leave free)" : "in its own buffer", host_bytes/1048576.0,
            (long long) max_tokens, max_swaps, decay);
        for (const group & g : groups) {
            fprintf(stderr, "llama_moe_vram: %2zu layers, %s/%s/%s: %5d slots (%.1f%%)\n", g.layers.size(), ggml_type_name(g.ref[0]->type),
                ggml_type_name(g.ref[1]->type), ggml_type_name(g.ref[2]->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*n_expert));
        }
    }

    ~impl() {
        if (llama_moe_vram::owner == this) {
            llama_moe_vram::owner = nullptr;
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();
        if (th.joinable()) {
            th.join();
        }
        for (const layer & l : layers) {
            for (const ggml_tensor * w : l.experts) {
                ggml_moe_stream_set_gpu_map(w, nullptr);
            }
        }
        if (backend_up) {
            ggml_backend_free(backend_up);
        }
        print_stats();
    }

    void print_stats() const {
        if (!log || n_uses == 0) {
            return;
        }
        size_t n_cached = 0, n_usable = 0, n_slots = 0;
        for (const group & g : groups) {
            for (int32_t s = 0; s < g.n_slots; ++s) {
                n_cached += g.key_of[s] >= 0;
                n_usable += g.usable[s];
            }
            n_slots += g.n_slots;
        }
        fprintf(stderr, "llama_moe_vram: %llu graphs, experts on the device %.1f%% (last %.1f%%), %llu copied, %llu failed, %llu evicted, %llu dropped for prompts; slots %zu used, %zu usable of %zu\n",
            (unsigned long long) n_graphs, 100.0*n_hits/n_uses, n_uses > last_uses ? 100.0*(n_hits - last_hits)/(n_uses - last_uses) : 0.0,
            (unsigned long long) n_copies, (unsigned long long) n_failed, (unsigned long long) n_evicted, (unsigned long long) n_dropped,
            n_cached, n_usable, n_slots);
        if (managed) {
            size_t kv = 0;
            {
                std::lock_guard<std::mutex> lock(llama_kv_cache::lazy_caches_mutex());
                for (llama_kv_cache * c : llama_kv_cache::lazy_caches()) {
                    auto it = kv_cells.find(c->vram_id);
                    kv += it != kv_cells.end() ? c->vram_bytes(it->second) : 0;
                }
            }
            fprintf(stderr, "llama_moe_vram: managed: banks %.0f MiB, compute buffer %.0f MiB mapped (generation needs %.0f), KV %.0f MiB, device free %.0f MiB; %llu shrinks, %llu grows, %llu experts kept by moving\n",
                banks_mapped()/1048576.0, compute_mapped()/1048576.0, compute_small()/1048576.0, kv/1048576.0, dev_free()/1048576.0,
                (unsigned long long) n_shrinks, (unsigned long long) n_grows, (unsigned long long) n_moved);
        }
    }

    void worker() {
#if defined(__linux__)
        // the copies run next to the SSD readers, away from the cores that compute
        if (const char * cpus = getenv("GGML_MOE_STREAM_IO_CPUS")) {
            int a = -1, b = -1;
            if (sscanf(cpus, "%d-%d", &a, &b) == 2 && a >= 0 && b >= a && b < CPU_SETSIZE) {
                cpu_set_t set;
                CPU_ZERO(&set);
                for (int c = a; c <= b; ++c) {
                    CPU_SET(c, &set);
                }
                pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
            }
        }
#endif
        char * stage = (char *) ggml_backend_buffer_get_base(staging.get());
        std::unique_lock<std::mutex> lock(mtx);
        for (;;) {
            cv.wait(lock, [this] { return stop || !todo.empty(); });
            if (stop) {
                return;
            }
            const job j = todo.front();
            todo.pop_front();
            busy = true;
            lock.unlock();

            const layer & l = layers[j.il];
            const group & g = groups[l.ig];
            bool ok = true;
            size_t off = 0;
            for (size_t ip = 0; ip < l.experts.size() && ok; ++ip) {
                const ggml_tensor * w = l.experts[ip];
                const size_t nb = w->nb[2];
                if (ggml_moe_stream_enabled()) {
                    ok = ggml_moe_stream_get_cached(w, j.e, stage + off, nb);
                } else {
                    memcpy(stage + off, (const char *) w->data + (size_t) j.e*nb, nb);
                }
                off += GGML_PAD(nb, 64);
            }
            if (ok) {
                off = 0;
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const size_t nb = l.experts[ip]->nb[2];
                    ggml_backend_tensor_set_async(backend_up, g.banks[ip], stage + off, (size_t) j.slot*nb, nb);
                    off += GGML_PAD(nb, 64);
                }
                ggml_backend_synchronize(backend_up);
            }

            lock.lock();
            done.push_back({ j, ok });
            busy = false;
            cv_idle.notify_all();
        }
    }

    void record(int32_t il, const ggml_tensor * ids) {
        graph_tokens = std::max<int64_t>(1, ids->ne[1]);
        for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) {
            for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
                const int32_t e = *(const int32_t *) ((const char *) ids->data + i1*ids->nb[1] + i0*ids->nb[0]);
                if (e < 0 || e >= n_expert) {
                    continue;
                }
                const size_t idx = (size_t) il*n_expert + e;
                n_uses++;
                n_hits += map_host[idx] >= 0;
                if (used[idx]++ == 0) {
                    used_list.push_back((int32_t) idx);
                }
            }
        }
    }

    void free_slot(group & g, int32_t slot) {
        g.key_of[slot] = -1;
        if (g.usable[slot]) {
            // keep the free list sorted, the highest offsets are the last to be overwritten by prompt batches
            g.free_slots.insert(std::lower_bound(g.free_slots.begin(), g.free_slots.end(), slot), slot);
        }
    }

    // with mtx locked: map the experts whose copies are done
    bool apply_done() {
        bool dirty = false;
        for (const auto & [j, ok] : done) {
            const size_t idx = (size_t) j.il*n_expert + j.e;
            if (state[idx] != 1 || slot_of[idx] != j.slot) {
                continue;
            }
            if (ok) {
                map_host[idx] = j.slot;
                state[idx]    = 2;
                n_copies++;
                dirty = true;
            } else {
                state[idx]   = 0;
                slot_of[idx] = -1;
                free_slot(groups[layers[j.il].ig], j.slot);
                n_failed++;
            }
        }
        done.clear();
        return dirty;
    }

    // the graph's allocations end at hwm in the shared buffer: drop the experts it overwrites
    // permanent (a generation graph): those slots are not used again
    // whether slot s of g has data in [lo, hi) of the shared buffer
    bool slot_overlaps(const group & g, int32_t s, size_t lo, size_t hi) const {
        const char * base = (const char *) ggml_backend_buffer_get_base(shared);
        for (const ggml_tensor * bank : g.banks) {
            const size_t a = (size_t) ((const char *) bank->data - base) + (size_t) s*bank->nb[2];
            if (a < hi && a + bank->nb[2] > lo) {
                return true;
            }
        }
        return false;
    }

    bool drop_below(size_t hwm, bool permanent) {
        return drop_range(0, hwm, permanent);
    }

    bool drop_range(size_t lo, size_t hi, bool permanent) {
        // no copy may write into the buffer while the graph uses it
        {
            std::unique_lock<std::mutex> lock(mtx);
            for (const job & j : todo) {
                const size_t idx = (size_t) j.il*n_expert + j.e;
                state[idx]   = 0;
                slot_of[idx] = -1;
                free_slot(groups[layers[j.il].ig], j.slot);
            }
            todo.clear();
            cv_idle.wait(lock, [this] { return !busy; });
            apply_done();
        }
        bool dirty = false;
        for (group & g : groups) {
            for (int32_t s = 0; s < g.n_slots; ++s) {
                if ((lo == 0 ? g.slot_low[s] >= hi : !slot_overlaps(g, s, lo, hi)) || (!g.usable[s] && g.key_of[s] < 0)) {
                    continue;
                }
                const int32_t idx = g.key_of[s];
                if (idx >= 0) {
                    if (state[idx] == 2) {
                        map_host[idx] = -1;
                        dirty = true;
                        if (!permanent) {
                            refill.push_back(idx);
                        }
                        n_dropped++;
                    }
                    state[idx]   = 0;
                    slot_of[idx] = -1;
                    g.key_of[s]  = -1;
                } else {
                    auto it = std::lower_bound(g.free_slots.begin(), g.free_slots.end(), s);
                    if (it != g.free_slots.end() && *it == s) {
                        g.free_slots.erase(it);
                    }
                }
                if (permanent) {
                    g.usable[s] = 0;
                } else {
                    free_slot(g, s);
                }
            }
        }
        return dirty;
    }


    // ---- managed mode ----

    size_t dev_free() const {
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(backend), &free, &total);
        return free;
    }

    size_t banks_mapped() const {
        return vmapped(buf_banks.get());
    }

    // the slots per group for a total of bytes, each group the same share of its experts
    void caps_for(size_t bytes, std::vector<int32_t> & caps) const {
        caps.resize(groups.size());
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            const group & g = groups[ig];
            const double share = (double) g.host_bytes/host_bytes_all;
            caps[ig] = g.slot_bytes > 0 ? (int32_t) std::min<double>(g.n_slots, bytes*share/g.slot_bytes) : 0;
        }
    }

    // slots [0, cap) of g usable: mapping their memory, or giving up the slots from cap on (their experts are dropped, into refill
    // when the memory goes to a prompt batch); false if the device ran out of memory while growing
    bool set_cap(group & g, int32_t cap, bool & dirty, bool to_refill) {
        cap = std::max<int32_t>(0, std::min(cap, g.n_slots));
        if (cap == g.n_cap) {
            return true;
        }
        char * base = (char *) ggml_backend_buffer_get_base(buf_banks.get());
        if (cap > g.n_cap) {
            for (const ggml_tensor * bank : g.banks) {
                // + 1: the padding slot after the last one
                if (!vmap(buf_banks.get(), (const char *) bank->data - base, (size_t) (cap + 1)*bank->nb[2])) {
                    return false;
                }
            }
            for (int32_t s = g.n_cap; s < cap; ++s) {
                g.usable[s] = 1;
                if (g.key_of[s] < 0) {
                    free_slot(g, s);
                }
            }
            g.n_cap = cap;
            return true;
        }
        const int32_t ig = (int32_t) (&g - groups.data());
        {
            std::unique_lock<std::mutex> lock(mtx);
            for (auto it = todo.begin(); it != todo.end(); ) {
                if (layers[it->il].ig == ig && it->slot >= cap) {
                    const size_t idx = (size_t) it->il*n_expert + it->e;
                    state[idx]   = 0;
                    slot_of[idx] = -1;
                    it = todo.erase(it);
                } else {
                    ++it;
                }
            }
            cv_idle.wait(lock, [this] { return !busy; });
            dirty |= apply_done();
        }
        compact(g, cap, dirty);
        for (int32_t s = cap; s < g.n_cap; ++s) {
            const int32_t idx = g.key_of[s];
            if (idx >= 0) {
                if (state[idx] == 2) {
                    map_host[idx] = -1;
                    dirty = true;
                    if (to_refill) {
                        refill.push_back(idx);
                    }
                    n_dropped++;
                }
                state[idx]   = 0;
                slot_of[idx] = -1;
                g.key_of[s]  = -1;
            }
            g.usable[s] = 0;
        }
        g.free_slots.erase(std::remove_if(g.free_slots.begin(), g.free_slots.end(), [&](int32_t s) { return s >= cap; }), g.free_slots.end());
        g.n_cap = cap;
        return true;
    }

    // before g shrinks to cap slots: the hottest experts in the slots that go move into free slots that stay, or replace colder
    // experts there (device-to-device copies), so that a shrink drops the coldest experts, not whichever sit at the end
    void compact(group & g, int32_t cap, bool & dirty) {
        std::vector<int32_t> movers;
        for (int32_t s = cap; s < g.n_cap; ++s) {
            const int32_t k = g.key_of[s];
            if (k >= 0 && state[k] == 2) {
                movers.push_back(s);
            }
        }
        if (movers.empty() || cap == 0) {
            return;
        }
        std::sort(movers.begin(), movers.end(), [&](int32_t a, int32_t b) { return score[g.key_of[a]] > score[g.key_of[b]]; });
        std::vector<int32_t> frees, colds;
        for (int32_t s : g.free_slots) {
            if (s < cap) {
                frees.push_back(s);
            }
        }
        for (int32_t s = 0; s < cap; ++s) {
            const int32_t k = g.key_of[s];
            if (k >= 0 && state[k] == 2) {
                colds.push_back(s);
            }
        }
        std::sort(colds.begin(), colds.end(), [&](int32_t a, int32_t b) { return score[g.key_of[a]] < score[g.key_of[b]]; });

        // the graphs that read the banks are done before their slots are rewritten
        ggml_backend_synchronize(backend);
        ggml_init_params params = { movers.size()*2*g.banks.size()*ggml_tensor_overhead(), nullptr, true };
        ggml_context_ptr ctx(ggml_init(params));
        size_t fi = 0, ci = 0;
        for (int32_t s : movers) {
            const int32_t k = g.key_of[s];
            int32_t t = -1;
            if (fi < frees.size()) {
                t = frees[fi++];
                g.free_slots.erase(std::remove(g.free_slots.begin(), g.free_slots.end(), t), g.free_slots.end());
            } else if (ci < colds.size() && score[g.key_of[colds[ci]]] < score[k]) {
                t = colds[ci++];
                const int32_t v = g.key_of[t];
                map_host[v] = -1;
                state[v]    = 0;
                slot_of[v]  = -1;
                n_evicted++;
            } else {
                break;
            }
            for (ggml_tensor * bank : g.banks) {
                ggml_tensor * src = ggml_view_2d(ctx.get(), bank, bank->ne[0], bank->ne[1], bank->nb[1], (size_t) s*bank->nb[2]);
                ggml_tensor * dst = ggml_view_2d(ctx.get(), bank, bank->ne[0], bank->ne[1], bank->nb[1], (size_t) t*bank->nb[2]);
                GGML_ASSERT(ggml_backend_view_init(src) == GGML_STATUS_SUCCESS && ggml_backend_view_init(dst) == GGML_STATUS_SUCCESS);
                ggml_backend_tensor_copy(src, dst);
            }
            g.key_of[t]  = k;
            g.key_of[s]  = -1;
            slot_of[k]   = t;
            map_host[k]  = t;
            dirty = true;
            n_moved++;
        }
    }

    // gives back the memory of the slots beyond each group's cap (after the graphs that may read them are done)
    void unmap_beyond_caps() {
        char * base = (char *) ggml_backend_buffer_get_base(buf_banks.get());
        for (const group & g : groups) {
            for (const ggml_tensor * bank : g.banks) {
                const size_t off  = (const char *) bank->data - base;
                const size_t keep = g.n_cap > 0 ? (size_t) (g.n_cap + 1)*bank->nb[2] : 0;
                const size_t size = ggml_nbytes(bank);
                if (keep < size) {
                    vunmap(buf_banks.get(), off + keep, size - keep);
                }
            }
        }
    }

    void shrink_banks(size_t bytes, bool to_refill, bool & dirty) {
        const size_t have = banks_mapped();
        std::vector<int32_t> caps;
        caps_for(have > bytes ? have - bytes : 0, caps);
        bool any = false;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            if (caps[ig] < groups[ig].n_cap) {
                set_cap(groups[ig], caps[ig], dirty, to_refill);
                any = true;
            }
        }
        if (!any) {
            return;
        }
        // no graph may read the dropped slots: the previous one must be done and the next one sees the new maps
        ggml_backend_synchronize(backend);
        if (dirty) {
            ggml_backend_tensor_set(map_dev, map_host, 0, ggml_nbytes(map_dev));
            dirty = false;
        }
        unmap_beyond_caps();
        n_shrinks++;
        // KV growth waits a few graphs before the banks grow again (no back and forth at every batch); a prompt batch's memory
        // comes back right after it
        grow_wait = to_refill ? 0 : 4;
        if (log) {
            fprintf(stderr, "llama_moe_vram: managed: banks %.0f -> %.0f MiB (%s)\n", have/1048576.0, banks_mapped()/1048576.0,
                to_refill ? "prompt batch" : "KV cache / compute buffer");
        }
    }

    void grow_banks(size_t bytes, bool & dirty) {
        std::vector<int32_t> caps;
        caps_for(banks_mapped() + bytes, caps);
        bool grew = false;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            if (caps[ig] > groups[ig].n_cap) {
                if (!set_cap(groups[ig], caps[ig], dirty, false)) {
                    break;
                }
                grew = true;
            }
        }
        n_grows += grew;
        if (grew && log && n_grows % 16 == 1) {
            fprintf(stderr, "llama_moe_vram: managed: banks grew to %.0f MiB\n", banks_mapped()/1048576.0);
        }
    }

    // before a graph: map what it needs (its part of the compute buffer, the prompt ring, the cells of the KV caches), taking the
    // memory from the banks if the device lacks it; generation graphs give back the rest and let the banks grow into it
    size_t compute_mapped() const {
        size_t res = 0;
        for (const auto & [b, st] : cbufs) {
            res += st.hi + (st.ring_mapped ? ring_hi - ring_lo : 0);
        }
        return res;
    }

    size_t compute_small() const {
        size_t res = 0;
        for (const auto & [b, st] : cbufs) {
            res += st.small_hwm;
        }
        return res;
    }

    // be: the backend whose compute buffer of sched is managed (this context's or another's); its prompt batches of at least r_min
    // tokens also use [r_lo, r_hi) of it
    void manage(ggml_backend_sched_t sched, ggml_backend_t be, int64_t n_tokens, bool small, size_t r_lo, size_t r_hi, int64_t r_min,
            bool & dirty) {
        ggml_backend_buffer_t cbuf = ggml_backend_sched_get_buffer(sched, be);
        if (cbuf == nullptr) {
            return;
        }
        // a new compute buffer starts unmapped (an old one was freed: its entry goes)
        cbuf_state & cs = cbufs[cbuf];
        if (vmapped(cbuf) < cs.hi) {
            // a freed buffer's entry: this buffer reuses its handle and is not mapped
            cs = cbuf_state();
        }
        cs.be = be;
        if (r_hi > r_lo) {
            cs.ring_lo = r_lo;
            cs.ring_hi = r_hi;
        }
        size_t & compute_hi  = cs.hi;
        size_t & small_hwm   = cs.small_hwm;
        bool   & ring_mapped = cs.ring_mapped;
        const size_t ring_lo = r_lo, ring_hi = r_hi;
        const int64_t ring_min_batch = r_min;
        const size_t cbuf_size = ggml_backend_buffer_get_size(cbuf);
        const size_t hwm       = ggml_backend_sched_get_buffer_hwm(sched, be);
        const bool   ring      = ring_min_batch > 0 && n_tokens >= ring_min_batch && ring_hi > ring_lo;
        if (small) {
            small_hwm = std::max(small_hwm, hwm);
        }
        // device memory to leave free: the scratch pools grow with the batch (prompt batches of 4096 tokens: ~640 MiB)
        // LLAMA_MOE_VRAM_MARGIN_MIB (default 320): the part that generation keeps free
        static const size_t margin0 = [] { const char * e = getenv("LLAMA_MOE_VRAM_MARGIN_MIB"); return (size_t) (e ? atoll(e) : 320) << 20; }();
        const size_t margin  = margin0 + (small ? 0 : (size_t) std::min<int64_t>(n_tokens, 4096)*(80u << 10));
        const size_t want_hi = std::min(cbuf_size, GGML_PAD(small ? small_hwm : hwm, gran));

        std::vector<std::pair<llama_kv_cache *, uint32_t>> kvs;
        {
            std::lock_guard<std::mutex> lock(llama_kv_cache::lazy_caches_mutex());
            for (llama_kv_cache * c : llama_kv_cache::lazy_caches()) {
                kvs.push_back({ c, c->vram_cells() });
            }
        }
        size_t need = 0;
        if (want_hi > compute_hi) {
            need += want_hi - compute_hi;
        }
        if (ring && !ring_mapped) {
            need += ring_hi - ring_lo;
        }
        for (const auto & [c, n] : kvs) {
            const uint32_t have = kv_cells.count(c->vram_id) ? kv_cells[c->vram_id] : 0;
            if (n > have) {
                need += c->vram_bytes(n) - c->vram_bytes(have);
            }
        }
        if (need > 0) {
            size_t free = dev_free();
            if (need + margin > free) {
                reclaim_others(cbuf, need + margin - free);
                free = dev_free();
            }
            if (need + margin > free) {
                shrink_banks(need + margin - free, !small, dirty);
            }
        }
        for (int attempt = 0; ; ++attempt) {
            bool ok = true;
            if (want_hi > compute_hi) {
                ok = vmap(cbuf, compute_hi, want_hi - compute_hi);
                if (ok) {
                    compute_hi = want_hi;
                }
            }
            if (ok && ring && !ring_mapped) {
                ok = ring_mapped = vmap(cbuf, ring_lo, ring_hi - ring_lo);
            }
            for (const auto & [c, n] : kvs) {
                const uint32_t have = kv_cells.count(c->vram_id) ? kv_cells[c->vram_id] : 0;
                if (ok && n > have) {
                    ok = c->vram_map(n);
                    if (ok) {
                        kv_cells[c->vram_id] = n;
                    }
                }
            }
            if (ok) {
                break;
            }
            if (reclaim_others(cbuf, SIZE_MAX) > 0) {
                continue;
            }
            if (attempt > 16 || banks_mapped() == 0) {
                GGML_ABORT("llama_moe_vram: the device is out of memory for the KV caches and the compute buffer");
            }
            shrink_banks(256u << 20, !small, dirty);
        }
        if (!small) {
            return;
        }
        // generation: give back the prompt part of the compute buffer, the ring and KV cells that were freed, grow the banks
        bool synced = false;
        auto sync = [&] {
            if (!synced) {
                ggml_backend_synchronize(backend);
                ggml_backend_synchronize(be);
                synced = true;
            }
        };
        if (compute_hi > want_hi) {
            sync();
            vunmap(cbuf, want_hi, compute_hi - want_hi);
            // a deep prompt's graph may have reached into the ring: it is unmapped with the rest then
            if (ring_mapped && ring_lo < compute_hi && ring_hi > want_hi) {
                ring_mapped = false;
            }
            compute_hi = want_hi;
        }
        // (the prompt ring stays mapped: tool results arrive as short prompts between generation steps, unmapping and mapping its
        // 144 MiB every time would take that memory from the banks over and over)
        for (const auto & [c, n] : kvs) {
            const uint32_t have = kv_cells.count(c->vram_id) ? kv_cells[c->vram_id] : 0;
            if (have > n + 8192) {
                sync();
                c->vram_trim(n);
                kv_cells[c->vram_id] = n;
            }
        }
        if (grow_wait > 0) {
            grow_wait--;
            return;
        }
        const size_t free = dev_free();
        if (free > margin + (64u << 20)) {
            grow_banks(std::min(free - margin, (size_t) 1024 << 20), dirty);
        }
    }


    void disable(const char * why) {
        if (disabled) {
            return;
        }
        fprintf(stderr, "llama_moe_vram: disabled: %s\n", why);
        drop_below(SIZE_MAX, true);
        refill.clear();
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(map_dev, map_host, 0, ggml_nbytes(map_dev));
        disabled = true;
    }

    // with a slot for expert idx, returns false if there is none (that is colder)
    int32_t take_slot(int32_t idx, bool & dirty) {
        group & g = groups[layers[idx / n_expert].ig];
        if (!g.free_slots.empty()) {
            const int32_t slot = g.free_slots.back();
            g.free_slots.pop_back();
            return slot;
        }
        // the coldest cached expert of the group, if it is colder
        int32_t victim = -1;
        for (int32_t s = 0; s < g.n_slots; ++s) {
            const int32_t k = g.key_of[s];
            if (k >= 0 && state[k] == 2 && (victim < 0 || score[k] < score[victim])) {
                victim = k;
            }
        }
        if (victim < 0 || score[victim] >= score[idx]) {
            return -1;
        }
        const int32_t slot = slot_of[victim];
        map_host[victim] = -1;
        state[victim]    = 0;
        slot_of[victim]  = -1;
        g.key_of[slot]   = -1;
        n_evicted++;
        dirty = true;
        return slot;
    }

    void begin_graph(ggml_backend_sched_t sched, int64_t n_tokens) {
        if (disabled) {
            return;
        }
        bool dirty = false;
        const bool small = n_tokens <= max_tokens;

        if (managed) {
            // the memory a generation step needs (<= 8 tokens) stays mapped between graphs; larger hybrid batches (short prompts
            // computed with the banks) map what they need and give it back like prompt batches
            manage(sched, backend, n_tokens, n_tokens <= 8, ring_lo, ring_hi, ring_min_batch, dirty);
        } else if (shared) {
            if (ggml_backend_sched_get_buffer(sched, backend) != shared) {
                disable("the compute buffer was reallocated");
                return;
            }
            const size_t hwm = ggml_backend_sched_get_buffer_hwm(sched, backend);
            // offloaded prompt batches also stream their experts through the ring at the end of the buffer
            if (ring_min_batch > 0 && n_tokens >= ring_min_batch && ring_hi > region_lo) {
                dirty |= drop_range(std::max(ring_lo, region_lo), ring_hi, false);
            }
            if (hwm > region_lo) {
                dirty |= drop_below(hwm, small);
                if (small) {
                    region_lo = hwm;
                    fprintf(stderr, "llama_moe_vram: generation graphs use %.1f MiB of the compute buffer, the cache gives way\n", hwm/1048576.0);
                }
            }
        }
        if (!small) {
            if (dirty) {
                ggml_backend_synchronize(backend);
                ggml_backend_tensor_set(map_dev, map_host, 0, ggml_nbytes(map_dev));
            }
            after_large = true;
            return;
        }

        // a long prompt grows the device scratch pools, release them before generation needs room for its graphs and copies
        if (after_large) {
            after_large = false;
            if (trim_pools) {
                // only this context's pools: another context (an MTP draft) may replay graphs captured with its scratch addresses
                const size_t freed = trim_pools(backend);
                if (log && freed > (64u << 20)) {
                    fprintf(stderr, "llama_moe_vram: released %.0f MiB of device scratch pools after a prompt\n", freed/1048576.0);
                }
            }
        }

        n_graphs++;
        if (log && n_graphs % 200 == 0) {
            print_stats();
            last_uses = n_uses;
            last_hits = n_hits;
        }

        std::vector<job> jobs;
        size_t backlog;
        {
            std::lock_guard<std::mutex> lock(mtx);
            dirty |= apply_done();
            backlog = todo.size();
        }

        // after a prompt: copy the dropped experts again, the hottest first
        if (!refill.empty()) {
            std::sort(refill.begin(), refill.end(), [&](int32_t a, int32_t b) { return score[a] > score[b]; });
            for (int32_t idx : refill) {
                if (state[idx] != 0) {
                    continue;
                }
                group & g = groups[layers[idx / n_expert].ig];
                if (g.free_slots.empty()) {
                    continue;
                }
                const int32_t slot = g.free_slots.back();
                g.free_slots.pop_back();
                g.key_of[slot] = idx;
                state[idx]     = 1;
                slot_of[idx]   = slot;
                jobs.push_back({ idx / n_expert, idx % n_expert, slot });
            }
            refill.clear();
        }

        // the use counts of the last graph, and the most used experts that were computed on the CPU replace the least used cached ones
        if (!used_list.empty()) {
            for (int64_t t = 0; t < graph_tokens; ++t) {
                scale /= decay;
            }
            if (scale > 1e30f) {
                for (float & s : score) {
                    s /= scale;
                }
                scale = 1.0f;
            }
            cands.clear();
            for (int32_t idx : used_list) {
                score[idx] += scale*used[idx];
                used[idx] = 0;
                if (state[idx] == 0 && layers[idx / n_expert].ig >= 0) {
                    cands.push_back(idx);
                }
            }
            used_list.clear();

            std::sort(cands.begin(), cands.end(), [&](int32_t a, int32_t b) { return score[a] > score[b]; });
            int n_new = 0;
            // up to max_swaps per token; while slots are free (managed banks that just grew) they are filled faster
            int64_t swap_limit = max_swaps*graph_tokens;
            if (managed) {
                for (const group & g : groups) {
                    if (!g.free_slots.empty()) {
                        swap_limit = std::max<int64_t>(swap_limit, 64);
                        break;
                    }
                }
            }
            for (int32_t idx : cands) {
                if ((int64_t) (n_new + backlog) >= swap_limit) {
                    break;
                }
                const int32_t slot = take_slot(idx, dirty);
                if (slot < 0) {
                    continue;
                }
                groups[layers[idx / n_expert].ig].key_of[slot] = idx;
                state[idx]   = 1;
                slot_of[idx] = slot;
                jobs.push_back({ idx / n_expert, idx % n_expert, slot });
                n_new++;
            }
        }

        if (dirty || !jobs.empty()) {
            // the previous graph may still read the maps, use the slots that are given to new experts, or (shared) the buffer
            ggml_backend_synchronize(backend);
        }
        if (dirty) {
            ggml_backend_tensor_set(map_dev, map_host, 0, ggml_nbytes(map_dev));
        }
        if (!jobs.empty()) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                todo.insert(todo.end(), jobs.begin(), jobs.end());
            }
            cv.notify_one();
        }
    }
};

llama_moe_vram::llama_moe_vram(const llama_model & model, ggml_backend_t backend, ggml_backend_buffer_type_t buft, size_t size,
        ggml_backend_buffer_t shared) :
    pimpl(new impl(model, backend, buft, size, shared)) {
}

llama_moe_vram::~llama_moe_vram() = default;

ggml_backend_t llama_moe_vram::backend() const {
    return pimpl->backend;
}

bool llama_moe_vram::active(int32_t il, int64_t n_tokens) const {
    return !pimpl->disabled && il >= 0 && il < pimpl->n_layer && pimpl->layers[il].ig >= 0 && pimpl->groups[pimpl->layers[il].ig].n_slots > 0 &&
        n_tokens > 0 && n_tokens <= pimpl->max_tokens;
}

ggml_tensor * llama_moe_vram::get_slot_map(int32_t il) const {
    return pimpl->layers[il].slot_map;
}

ggml_tensor * llama_moe_vram::get_experts(const ggml_tensor * w) const {
    const auto it = pimpl->bindings.find(w);
    return it != pimpl->bindings.end() ? it->second : nullptr;
}

void llama_moe_vram::pre_op(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * userdata) {
    GGML_UNUSED(b);
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    GGML_ASSERT(a->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32 && ggml_are_same_shape(a, dst));
    for (int64_t i3 = 0; i3 < a->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < a->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < a->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < a->ne[0]; ++i0) {
                    *(int32_t *) ((char *) dst->data + i3*dst->nb[3] + i2*dst->nb[2] + i1*dst->nb[1] + i0*dst->nb[0]) =
                        *(const int32_t *) ((const char *) a->data + i3*a->nb[3] + i2*a->nb[2] + i1*a->nb[1] + i0*a->nb[0]);
                }
            }
        }
    }
    const impl::pre_ud * ud = (const impl::pre_ud *) userdata;
    ud->self->record(ud->il, dst);
}

void llama_moe_vram::pre_op3(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c, int ith, int nth, void * userdata) {
    GGML_UNUSED(c);
    pre_op(dst, a, b, ith, nth, userdata);
}

void * llama_moe_vram::pre_userdata(int32_t il) {
    return &pimpl->uds[il];
}

void llama_moe_vram::set_prefill_ring(int64_t min_batch, size_t offset, size_t size) {
    pimpl->ring_min_batch = min_batch;
    pimpl->ring_lo = offset;
    pimpl->ring_hi = offset + size;
}

void llama_moe_vram::begin_graph(ggml_backend_sched_t sched, int64_t n_tokens) {
    pimpl->begin_graph(sched, n_tokens);
}

llama_moe_vram::impl * llama_moe_vram::owner = nullptr;

// without a managing instance (no banks): lazily backed buffers are still mapped before every graph, what the graph needs and no more
static void llama_vram_map_plain(ggml_backend_sched_t sched, ggml_backend_t backend, int64_t n_tokens, size_t ring_off, size_t ring_size,
        int64_t ring_min_batch) {
    ggml_backend_buffer_t cbuf = ggml_backend_sched_get_buffer(sched, backend);
    if (cbuf == nullptr) {
        return;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto vmap = reg ? (bool (*)(ggml_backend_buffer_t, size_t, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_buffer_map") : nullptr;
    if (vmap == nullptr) {
        return;
    }
    bool ok = vmap(cbuf, 0, ggml_backend_sched_get_buffer_hwm(sched, backend));
    if (ok && ring_size > 0 && ring_min_batch > 0 && n_tokens >= ring_min_batch) {
        ok = vmap(cbuf, ring_off, ring_size);
    }
    {
        std::lock_guard<std::mutex> lock(llama_kv_cache::lazy_caches_mutex());
        for (llama_kv_cache * c : llama_kv_cache::lazy_caches()) {
            ok = ok && c->vram_map(c->vram_cells());
        }
    }
    if (!ok) {
        GGML_ABORT("llama_moe_vram: the device is out of memory for the KV caches and the compute buffer");
    }
}

void llama_moe_vram::manage_other(ggml_backend_sched_t sched, ggml_backend_t backend, int64_t n_tokens, size_t ring_off, size_t ring_size,
        int64_t ring_min_batch) {
    impl * m = owner;
    if (m == nullptr || m->disabled || !m->managed) {
        llama_vram_map_plain(sched, backend, n_tokens, ring_off, ring_size, ring_min_batch);
        return;
    }
    bool dirty = false;
    m->manage(sched, backend, n_tokens, n_tokens <= 8, ring_off, ring_off + ring_size, ring_min_batch, dirty);
    if (dirty) {
        // the banks changed: the owner's next graph reads the new maps
        ggml_backend_synchronize(m->backend);
        ggml_backend_tensor_set(m->map_dev, m->map_host, 0, ggml_nbytes(m->map_dev));
    }
}
