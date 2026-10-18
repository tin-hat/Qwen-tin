// Prompt batches with streamed MoE experts: a MUL_MAT_ID whose expert weights stay in host memory (see
// ggml_backend_sched_set_stream_weight_callback) uploads the experts the batch uses through a ring of device chunk buffers on its
// own stream and computes each chunk as soon as it is uploaded, so the uploads of the next chunks (and weights) overlap the compute.
// The experts come from a source registered by the user (ggml_moe_stream_copy: the RAM cache, or reads from the file) in its order.
#include "mmid-stream.cuh"
#include "mmq.cuh"

#include <string>
#include <vector>

typedef void (*moe_set_t)(void * ctx, size_t offset, const void * data, size_t size);
typedef void (*moe_sync_t)(void * ctx);
typedef bool (*moe_source_t)(const ggml_tensor * w, const int32_t * counts, moe_set_t set, moe_sync_t sync, void * ctx);

namespace {

constexpr size_t   CHUNK_PAD = 64u << 10; // MMQ may read a little past the last expert of a chunk
constexpr int32_t  SLOT_NONE = 0x7fffffff;

struct mmid_stream_state {
    moe_source_t source    = nullptr;
    char *       ring      = nullptr;
    size_t       ring_size = 0;
    int          n_buf     = 0;
    uint64_t     gen       = 0; // bumped for each graph: the ids are cached per graph

    bool         init = false;
    int          device = -1;
    cudaStream_t cs   = nullptr;
    std::vector<cudaEvent_t> ev_up;
    std::vector<cudaEvent_t> ev_free;
    std::vector<bool>        busy;     // ev_free was recorded
    std::vector<int32_t *>   map_host; // pinned [n_expert_max]
    std::vector<int32_t *>   map_dev;  // [n_expert_max]
    int64_t      n_expert_max = 0;
    int          next = 0;

    // pinned bounce buffers: experts in pageable memory are copied into them so that their uploads do not block the host
    int                      n_bounce    = 0;
    size_t                   bounce_size = 0;
    std::vector<char *>      bounce;
    std::vector<cudaEvent_t> ev_bounce;
    std::vector<bool>        bounce_busy;
    int                      cur_bounce  = -1;
    size_t                   bounce_off  = 0;
    double                   bounce_bytes = 0, direct_bytes = 0;

    const ggml_tensor *  ids_t   = nullptr;
    uint64_t             ids_gen = ~(uint64_t) 0;
    std::vector<int32_t> ids_host;
    std::vector<int32_t> counts;
    int64_t              n_used_experts = 0;

    // stats
    bool    log = false;
    int64_t n_ops = 0, n_chunks = 0;
    double  bytes = 0;
    // host time per phase (s): waiting for the ids (the GPU computing what precedes the experts), feeding the experts (source:
    // reads, copies, uploads), the rest of the op; and time between ops (the host building/launching the other GPU work)
    double  t_ids = 0, t_feed = 0, t_tail = 0, t_between = 0;
    int64_t t_last_end = 0;

    // GGML_MMID_STREAM_TIMELINE=path (diagnostic): per op host phases and device timestamps (compute stream reaching the op,
    // its first and last upload on the copy stream, its last chunk computed), written per graph
    FILE * tl = nullptr;
    struct tl_op {
        std::string name;
        int64_t n_tokens, n_used_experts;
        int64_t h0, h1, h2, h3;         // host us: start, ids known, fed, flushed
        double  memcpy_s, bwait_s, rwait_s, bytes;
        int64_t chunks;
        cudaEvent_t c0, u0, u1, c1;
        bool    has_u0 = false;
    };
    std::vector<tl_op>       tl_ops;
    std::vector<cudaEvent_t> tl_pool;
    size_t                   tl_next = 0;
    int64_t                  tl_graph = 0;
    double                   tl_memcpy = 0, tl_bwait = 0, tl_rwait = 0; // running sums, the op's share is the difference
};

mmid_stream_state g_mmid;

struct op_ctx {
    ggml_backend_cuda_context * ctx;
    const ggml_tensor * src0;
    const ggml_tensor * src1;
    ggml_tensor *       dst;
    const int32_t *     ids_d;
    int64_t             ids_nb1; // in int32
    int64_t             n_expert, n_used, n_tokens;
    size_t              nb2, buf_size;
    int64_t             cap;      // experts per chunk buffer
    int                 b = -1;   // current buffer
    int64_t             k = 0;    // experts in it
    int64_t             rows = 0; // (token, slot) pairs of its experts
    int64_t             delivered = 0;
    // ggml_mul_mat_id_reduce: each chunk's rows go to a scratch buffer of at most rows_cap rows and are added, weighted, to the
    // token's sum (dst [ne01, 1, n_tokens]), in the order of the chunks and, within a chunk, of the token's slots
    bool                reduce   = false;
    int64_t             rows_cap = 0;
    const float *       w_d      = nullptr;
    int64_t             w_s1 = 0, w_s2 = 0; // in floats
};

__global__ void mmid_stream_scatter_inv(const int32_t * __restrict__ pairs, const int n_rows, int32_t * __restrict__ inv) {
    const int r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r < n_rows) {
        inv[pairs[r]] = r;
    }
}

// one block per token: adds its rows of this chunk (inv: (token, slot) -> row or -1) to its sum, slot by slot
__global__ void mmid_stream_reduce_rows(const float * __restrict__ rows, const int32_t * __restrict__ inv, const float * __restrict__ w,
        float * __restrict__ dst, const int ne0, const int n_used, const int64_t w_s1, const int64_t w_s2) {
    const int t = blockIdx.x;
    int   rs[32];
    float ws[32];
    int   n = 0;
    for (int s = 0; s < n_used; ++s) {
        const int r = inv[t*n_used + s];
        if (r >= 0) {
            rs[n] = r;
            ws[n] = w[s*w_s1 + t*w_s2];
            n++;
        }
    }
    if (n == 0) {
        return;
    }
    float * d = dst + (int64_t) t*ne0;
    for (int c = threadIdx.x; c < ne0; c += blockDim.x) {
        float acc = d[c];
        for (int k = 0; k < n; ++k) {
            // the product and the sum rounded separately, as the unfused weighting and adds
            acc = __fadd_rn(acc, __fmul_rn(ws[k], rows[(int64_t) rs[k]*ne0 + c]));
        }
        d[c] = acc;
    }
}

__global__ void mmid_stream_remap(const int32_t * __restrict__ ids, const int64_t nb1, const int32_t * __restrict__ map,
        int32_t * __restrict__ out, const int n_used, const int n_total, const int n_expert, const int slot_lo, const int slot_hi) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_total) {
        return;
    }
    const int t = i / n_used;
    const int j = i % n_used;
    const int e = ids[t*nb1 + j];
    const int s = e >= 0 && e < n_expert ? map[e] : -1;
    out[i] = s >= slot_lo && s < slot_hi ? s : SLOT_NONE;
}

void ensure_init(int64_t n_expert) {
    mmid_stream_state & g = g_mmid;
    if (g.init && n_expert <= g.n_expert_max) {
        return;
    }
    if (!g.init) {
        g.device = ggml_cuda_get_device();
        CUDA_CHECK(cudaStreamCreateWithFlags(&g.cs, cudaStreamNonBlocking));
        g.ev_up.resize(g.n_buf);
        g.ev_free.resize(g.n_buf);
        g.busy.assign(g.n_buf, false);
        for (int b = 0; b < g.n_buf; ++b) {
            CUDA_CHECK(cudaEventCreateWithFlags(&g.ev_up[b],   cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&g.ev_free[b], cudaEventDisableTiming));
        }
        const char * e = getenv("GGML_MOE_STREAM_LOG");
        g.log = e != nullptr && atoi(e) != 0;

        // GGML_MMID_STREAM_BOUNCE: pinned bounce buffers for experts in pageable memory (16 MiB each)
        g.n_bounce    = getenv("GGML_MMID_STREAM_BOUNCE") ? std::max(2, atoi(getenv("GGML_MMID_STREAM_BOUNCE"))) : 8;
        g.bounce_size = 16u << 20;
        for (int i = 0; i < g.n_bounce; ++i) {
            char * p = nullptr;
            cudaEvent_t ev;
            CUDA_CHECK(cudaMallocHost((void **) &p, g.bounce_size));
            CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
            g.bounce.push_back(p);
            g.ev_bounce.push_back(ev);
            g.bounce_busy.push_back(false);
        }
    }
    for (int32_t * p : g.map_host) {
        CUDA_CHECK(cudaFreeHost(p));
    }
    for (int32_t * p : g.map_dev) {
        CUDA_CHECK(cudaFree(p));
    }
    g.map_host.assign(g.n_buf, nullptr);
    g.map_dev.assign(g.n_buf, nullptr);
    for (int b = 0; b < g.n_buf; ++b) {
        CUDA_CHECK(cudaMallocHost((void **) &g.map_host[b], n_expert*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc((void **) &g.map_dev[b], n_expert*sizeof(int32_t)));
    }
    g.n_expert_max = n_expert;
    g.init = true;
}

void flush(op_ctx & o) {
    mmid_stream_state & g = g_mmid;
    if (o.b < 0 || o.k == 0) {
        return;
    }
    const int b = o.b;
    cudaStream_t stream = o.ctx->stream();
    char * buf = g.ring + (size_t) b*o.buf_size;

    // MMQ reads a little past the last expert (its K loop runs over the padded row, times zero activations): stale bytes there,
    // e.g. another weight's experts at other offsets, can hold NaN or Inf scales, so the pad after the last slot is zeroed
    CUDA_CHECK(cudaMemsetAsync(buf + (size_t) o.k*o.nb2, 0, std::min(CHUNK_PAD, o.buf_size - (size_t) o.k*o.nb2), g.cs));
    CUDA_CHECK(cudaEventRecord(g.ev_up[b], g.cs));
    CUDA_CHECK(cudaMemcpyAsync(g.map_dev[b], g.map_host[b], o.n_expert*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream, g.ev_up[b], 0));

    const int n_total = (int) (o.n_used*o.n_tokens);
    {
        ggml_cuda_pool_alloc<int32_t> ids_local(o.ctx->pool(), n_total);
        mmid_stream_remap<<<(n_total + 255)/256, 256, 0, stream>>>(o.ids_d, o.ids_nb1, g.map_dev[b], ids_local.get(), (int) o.n_used, n_total, (int) o.n_expert, 0, (int) o.k);
        CUDA_CHECK(cudaGetLastError());
        if (o.reduce) {
            const int64_t ne0 = o.src0->ne[1];
            ggml_cuda_pool_alloc<float>   rows (o.ctx->pool(), o.rows*ne0);
            ggml_cuda_pool_alloc<int32_t> pairs(o.ctx->pool(), o.rows);
            ggml_cuda_pool_alloc<int32_t> inv  (o.ctx->pool(), n_total);
            ggml_cuda_mul_mat_q_ids_chunk(*o.ctx, o.src0->type, buf, o.src0->ne[0], o.src0->ne[1], o.k, o.src0->nb[1], o.nb2,
                o.src1, ids_local.get(), o.n_used, o.n_tokens, o.rows, o.dst, rows.get(), pairs.get());
            CUDA_CHECK(cudaMemsetAsync(inv.get(), 0xff, n_total*sizeof(int32_t), stream));
            mmid_stream_scatter_inv<<<(o.rows + 255)/256, 256, 0, stream>>>(pairs.get(), (int) o.rows, inv.get());
            CUDA_CHECK(cudaGetLastError());
            mmid_stream_reduce_rows<<<(int) o.n_tokens, 256, 0, stream>>>(rows.get(), inv.get(), o.w_d, (float *) o.dst->data, (int) ne0,
                (int) o.n_used, o.w_s1, o.w_s2);
            CUDA_CHECK(cudaGetLastError());
        } else {
            ggml_cuda_mul_mat_q_ids_chunk(*o.ctx, o.src0->type, buf, o.src0->ne[0], o.src0->ne[1], o.k, o.src0->nb[1], o.nb2,
                o.src1, ids_local.get(), o.n_used, o.n_tokens, o.rows, o.dst);
        }
    }

    CUDA_CHECK(cudaEventRecord(g.ev_free[b], stream));
    g.busy[b] = true;
    g.n_chunks++;
    o.b    = -1;
    o.k    = 0;
    o.rows = 0;
}

// starts the next buffer of the ring: waits until the chunk computed from it is done (its data and its slot map are free again)
void begin_buf(op_ctx & o) {
    mmid_stream_state & g = g_mmid;
    const int b = g.next;
    g.next = (g.next + 1) % g.n_buf;
    if (g.busy[b]) {
        const int64_t t0 = g.tl ? ggml_time_us() : 0;
        CUDA_CHECK(cudaEventSynchronize(g.ev_free[b]));
        g.busy[b] = false;
        if (g.tl) {
            g.tl_rwait += (ggml_time_us() - t0)/1e6;
        }
    }
    for (int64_t e = 0; e < o.n_expert; ++e) {
        g.map_host[b][e] = -1;
    }
    o.b    = b;
    o.k    = 0;
    o.rows = 0;
}

// uploads n bytes from host memory src to the device on the copy stream without blocking on pageable memory
cudaEvent_t tl_event() {
    mmid_stream_state & g = g_mmid;
    if (g.tl_next == g.tl_pool.size()) {
        cudaEvent_t ev;
        CUDA_CHECK(cudaEventCreate(&ev));
        g.tl_pool.push_back(ev);
    }
    return g.tl_pool[g.tl_next++];
}

void upload(char * dst, const char * src, size_t n) {
    mmid_stream_state & g = g_mmid;
    if (g.tl && !g.tl_ops.back().has_u0) {
        g.tl_ops.back().has_u0 = true;
        CUDA_CHECK(cudaEventRecord(g.tl_ops.back().u0, g.cs));
    }
    cudaPointerAttributes attr;
    const bool pinned = cudaPointerGetAttributes(&attr, src) == cudaSuccess && attr.type == cudaMemoryTypeHost;
    if (!pinned) {
        (void) cudaGetLastError();
    }
    if (pinned || g.n_bounce == 0 || n > g.bounce_size) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, g.cs));
        g.direct_bytes += n;
        return;
    }
    if (g.cur_bounce < 0 || g.bounce_off + n > g.bounce_size) {
        if (g.cur_bounce >= 0) {
            CUDA_CHECK(cudaEventRecord(g.ev_bounce[g.cur_bounce], g.cs));
            g.bounce_busy[g.cur_bounce] = true;
        }
        g.cur_bounce = (g.cur_bounce + 1) % g.n_bounce;
        if (g.bounce_busy[g.cur_bounce]) {
            const int64_t t0 = g.tl ? ggml_time_us() : 0;
            CUDA_CHECK(cudaEventSynchronize(g.ev_bounce[g.cur_bounce]));
            g.bounce_busy[g.cur_bounce] = false;
            if (g.tl) {
                g.tl_bwait += (ggml_time_us() - t0)/1e6;
            }
        }
        g.bounce_off = 0;
    }
    char * b = g.bounce[g.cur_bounce] + g.bounce_off;
    const int64_t tm0 = g.tl ? ggml_time_us() : 0;
    memcpy(b, src, n);
    if (g.tl) {
        g.tl_memcpy += (ggml_time_us() - tm0)/1e6;
    }
    CUDA_CHECK(cudaMemcpyAsync(dst, b, n, cudaMemcpyHostToDevice, g.cs));
    g.bounce_off   += n;
    g.bounce_bytes += n;
}

void set_cb(void * ctx, size_t offset, const void * data, size_t size) {
    op_ctx & o = *(op_ctx *) ctx;
    mmid_stream_state & g = g_mmid;
    GGML_ASSERT(offset % o.nb2 == 0);
    const int64_t e0 = offset / o.nb2;
    const int64_t n  = std::max<int64_t>(1, size / o.nb2);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t e = e0 + i;
        if (e >= o.n_expert || g.counts[e] == 0) {
            continue;
        }
        if (o.b >= 0 && (o.k == o.cap || (o.rows_cap > 0 && o.k > 0 && o.rows + g.counts[e] > o.rows_cap))) {
            flush(o);
        }
        if (o.b < 0) {
            begin_buf(o);
        }
        char * dst = g.ring + (size_t) o.b*o.buf_size + (size_t) o.k*o.nb2;
        upload(dst, (const char *) data + i*o.nb2, o.nb2);
        g.map_host[o.b][e] = (int32_t) o.k;
        o.k++;
        o.rows += g.counts[e];
        o.delivered++;
        g.bytes += o.nb2;
    }
}

void sync_cb(void * ctx) {
    GGML_UNUSED(ctx);
    CUDA_CHECK(cudaStreamSynchronize(g_mmid.cs));
}

}

bool ggml_cuda_mul_mat_id_streamed(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    return src0 != nullptr && src0->buffer != nullptr && ggml_backend_buffer_is_host(src0->buffer);
}

void ggml_cuda_mul_mat_id_stream(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    mmid_stream_state & g = g_mmid;
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(g.source != nullptr && g.ring != nullptr && g.n_buf > 0 && "streamed MUL_MAT_ID without a source");
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_cuda_get_device() == g.device || !g.init);

    const int64_t n_expert = src0->ne[2];
    const int64_t n_used   = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    ensure_init(n_expert);
    const int64_t tp0 = ggml_time_us();
    if (g.t_last_end > 0 && tp0 - g.t_last_end < 2000000) {
        g.t_between += (tp0 - g.t_last_end)/1e6;
    }
    if (g.tl) {
        mmid_stream_state::tl_op r;
        r.name = src0->name;
        r.n_tokens = n_tokens;
        r.h0 = tp0;
        r.memcpy_s = g.tl_memcpy; r.bwait_s = g.tl_bwait; r.rwait_s = g.tl_rwait; r.bytes = g.bytes; r.chunks = g.n_chunks;
        r.c0 = tl_event(); r.u0 = tl_event(); r.u1 = tl_event(); r.c1 = tl_event();
        CUDA_CHECK(cudaEventRecord(r.c0, ctx.stream()));
        g.tl_ops.push_back(r);
    }

    // the selected experts on the host, once per ids tensor and graph (gate, up and down share them)
    if (ids != g.ids_t || g.ids_gen != g.gen) {
        std::vector<int32_t> raw(ggml_nbytes(ids)/sizeof(int32_t));
        CUDA_CHECK(cudaMemcpyAsync(raw.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        g.counts.assign(n_expert, 0);
        g.n_used_experts = 0;
        const int64_t nb1 = ids->nb[1]/sizeof(int32_t);
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int64_t j = 0; j < n_used; ++j) {
                const int32_t e = raw[t*nb1 + j];
                GGML_ASSERT(e >= 0 && e < n_expert);
                g.n_used_experts += g.counts[e]++ == 0;
            }
        }
        g.ids_t   = ids;
        g.ids_gen = g.gen;
    }
    const int64_t tp1 = ggml_time_us();
    op_ctx o;
    o.ctx      = &ctx;
    o.src0     = src0;
    o.src1     = src1;
    o.dst      = dst;
    o.ids_d    = (const int32_t *) ids->data;
    o.ids_nb1  = ids->nb[1]/sizeof(int32_t);
    o.n_expert = n_expert;
    o.n_used   = n_used;
    o.n_tokens = n_tokens;
    o.nb2      = src0->nb[2];
    o.reduce   = ggml_get_op_params_i32(dst, 4) == 1;
    if (o.reduce) {
        // GGML_MMID_REDUCE_ROWS: rows of the scratch buffer of a chunk (default 12288: 120 MiB for 2560-wide rows)
        static const int64_t rows_cap = [] { const char * e = getenv("GGML_MMID_REDUCE_ROWS"); return e ? (int64_t) atoll(e) : (int64_t) 12288; }();
        const ggml_tensor * w = dst->src[3];
        GGML_ASSERT(w != nullptr && w->type == GGML_TYPE_F32 && n_used <= 32);
        GGML_ASSERT(dst->ne[0] == src0->ne[1] && dst->ne[1] == 1 && dst->ne[2] == n_tokens && ggml_is_contiguous(dst));
        o.rows_cap = std::max<int64_t>(rows_cap, n_tokens); // one expert's rows always fit
        o.w_d      = (const float *) w->data;
        o.w_s1     = w->nb[1]/sizeof(float);
        o.w_s2     = w->nb[2]/sizeof(float);
        CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), ctx.stream()));
    }
    o.buf_size = g.ring_size / g.n_buf;
    GGML_ASSERT(o.buf_size > CHUNK_PAD + o.nb2);
    o.cap      = (int64_t) ((o.buf_size - CHUNK_PAD) / o.nb2);

    // the copies of the source write the stream's buffers after the compute stream is done with them (begin_buf)
    if (!g.source(src0, g.counts.data(), set_cb, sync_cb, &o)) {
        GGML_ABORT("streamed MUL_MAT_ID: %s is not streamed by the source", src0->name);
    }
    const int64_t tp2 = ggml_time_us();
    flush(o);
    const int64_t tp3 = ggml_time_us();
    if (g.tl) {
        mmid_stream_state::tl_op & r = g.tl_ops.back();
        r.h1 = tp1; r.h2 = tp2; r.h3 = tp3;
        r.n_used_experts = g.n_used_experts;
        r.memcpy_s = g.tl_memcpy - r.memcpy_s; r.bwait_s = g.tl_bwait - r.bwait_s; r.rwait_s = g.tl_rwait - r.rwait_s;
        r.bytes = g.bytes - r.bytes; r.chunks = g.n_chunks - r.chunks;
        if (!r.has_u0) {
            r.has_u0 = true;
            CUDA_CHECK(cudaEventRecord(r.u0, g.cs));
        }
        CUDA_CHECK(cudaEventRecord(r.u1, g.cs));
        CUDA_CHECK(cudaEventRecord(r.c1, ctx.stream()));
    }
    g.t_ids  += (tp1 - tp0)/1e6;
    g.t_feed += (tp2 - tp1)/1e6;
    g.t_tail += (tp3 - tp2)/1e6;
    g.t_last_end = tp3;

    // GGML_MMID_STREAM_VERIFY=n (diagnostic): recompute the first n ops with all experts in one chunk and compare
    static int verify = [] { const char * e = getenv("GGML_MMID_STREAM_VERIFY"); return e ? atoi(e) : 0; }();
    if (verify > 0 && !o.reduce) {
        verify--;
        CUDA_CHECK(cudaDeviceSynchronize());
        const int64_t n_dst = ggml_nelements(dst);
        std::vector<float> got(n_dst), ref(n_dst);
        CUDA_CHECK(cudaMemcpy(got.data(), dst->data, n_dst*sizeof(float), cudaMemcpyDeviceToHost));
        // all used experts, packed, from the mmapped weight
        std::vector<int32_t> map(n_expert, -1);
        int64_t k = 0, rows = 0;
        for (int64_t e = 0; e < n_expert; ++e) {
            if (g.counts[e] > 0) {
                map[e] = (int32_t) k++;
                rows += g.counts[e];
            }
        }
        char * xbuf = nullptr;
        CUDA_CHECK(cudaMalloc((void **) &xbuf, (size_t) k*o.nb2 + CHUNK_PAD));
        for (int64_t e = 0; e < n_expert; ++e) {
            if (map[e] >= 0) {
                CUDA_CHECK(cudaMemcpy(xbuf + (size_t) map[e]*o.nb2, (const char *) src0->data + e*o.nb2, o.nb2, cudaMemcpyHostToDevice));
            }
        }
        int32_t * map_d = nullptr;
        CUDA_CHECK(cudaMalloc((void **) &map_d, n_expert*sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(map_d, map.data(), n_expert*sizeof(int32_t), cudaMemcpyHostToDevice));
        float * dst_save = (float *) dst->data;
        float * tmp_d = nullptr;
        CUDA_CHECK(cudaMalloc((void **) &tmp_d, n_dst*sizeof(float)));
        CUDA_CHECK(cudaMemset(tmp_d, 0xff, n_dst*sizeof(float)));
        {
            const int n_total = (int) (n_used*n_tokens);
            ggml_cuda_pool_alloc<int32_t> ids_local(ctx.pool(), n_total);
            mmid_stream_remap<<<(n_total + 255)/256, 256, 0, ctx.stream()>>>(o.ids_d, o.ids_nb1, map_d, ids_local.get(), (int) n_used, n_total, (int) n_expert, 0, (int) k);
            dst->data = tmp_d;
            ggml_cuda_mul_mat_q_ids_chunk(ctx, src0->type, xbuf, src0->ne[0], src0->ne[1], k, src0->nb[1], o.nb2, src1, ids_local.get(), n_used, n_tokens, rows, dst);
            dst->data = dst_save;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(ref.data(), tmp_d, n_dst*sizeof(float), cudaMemcpyDeviceToHost));
        int64_t n_diff_rows = 0, n_nan = 0, first = -1;
        const int64_t ne0 = dst->ne[0];
        for (int64_t r = 0; r < n_dst/ne0; ++r) {
            bool diff = false;
            for (int64_t i = 0; i < ne0; ++i) {
                const float a = got[r*ne0 + i], b = ref[r*ne0 + i];
                if (a != a) { n_nan++; }
                if (!(a == b)) { diff = true; }
            }
            if (diff) { n_diff_rows++; if (first < 0) first = r; }
        }
        fprintf(stderr, "mmid_stream verify: %s: %lld of %lld rows differ (first %lld), %lld NaN values, chunks so far %lld\n", src0->name,
            (long long) n_diff_rows, (long long) (n_dst/ne0), (long long) first, (long long) n_nan, (long long) g.n_chunks);
        CUDA_CHECK(cudaFree(xbuf));
        CUDA_CHECK(cudaFree(map_d));
        CUDA_CHECK(cudaFree(tmp_d));
    }

    GGML_ASSERT(o.delivered == g.n_used_experts && "streamed MUL_MAT_ID: not all used experts were delivered");
    g.n_ops++;
    if (g.log && g.n_ops % 144 == 0) {
        fprintf(stderr, "ggml_cuda_mmid_stream: %lld ops, %lld chunks, %.2f GiB uploaded (%.2f GiB through bounce buffers, %.2f GiB direct); host s: ids wait %.1f, feed %.1f, tail %.1f, between ops %.1f\n",
            (long long) g.n_ops, (long long) g.n_chunks, g.bytes/1073741824.0, g.bounce_bytes/1073741824.0, g.direct_bytes/1073741824.0,
            g.t_ids, g.t_feed, g.t_tail, g.t_between);
    }
}

void ggml_backend_cuda_moe_stream_set(void * source, void * ring, size_t ring_size, int n_buf) {
    mmid_stream_state & g = g_mmid;
    if (g.init) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    g.source    = (moe_source_t) source;
    g.ring      = (char *) ring;
    g.ring_size = ring_size;
    if (!g.init) {
        g.n_buf = n_buf;
    }
    GGML_ASSERT(g.n_buf == n_buf || !g.init);
    g.next = 0;
    std::fill(g.busy.begin(), g.busy.end(), false);
}

void ggml_backend_cuda_moe_stream_new_graph(void) {
    mmid_stream_state & g = g_mmid;
    g.gen++;
    static bool tl_init = false;
    if (!tl_init) {
        tl_init = true;
        const char * p = getenv("GGML_MMID_STREAM_TIMELINE");
        if (p != nullptr && *p) {
            g.tl = fopen(p, "w");
        }
    }
    if (g.tl == nullptr || g.tl_ops.empty()) {
        return;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto & f = g.tl_ops.front();
    auto ms = [&](cudaEvent_t e) { float v = 0; CUDA_CHECK(cudaEventElapsedTime(&v, f.c0, e)); return v; };
    fprintf(g.tl, "graph %lld ops %zu tokens %lld\n", (long long) g.tl_graph++, g.tl_ops.size(), (long long) f.n_tokens);
    for (const auto & r : g.tl_ops) {
        fprintf(g.tl, "%-26s used %3lld | host %9.1f ids %6.1f feed %6.1f (memcpy %5.1f bwait %5.1f rwait %5.1f) tail %5.1f | dev c0 %9.1f u0 %9.1f u1 %9.1f c1 %9.1f | %6.1f MiB %2lld ch\n",
            r.name.c_str(), (long long) r.n_used_experts, (r.h0 - f.h0)/1e3, (r.h1 - r.h0)/1e3, (r.h2 - r.h1)/1e3,
            r.memcpy_s*1e3, r.bwait_s*1e3, r.rwait_s*1e3, (r.h3 - r.h2)/1e3, ms(r.c0), ms(r.u0), ms(r.u1), ms(r.c1),
            r.bytes/1048576.0, (long long) r.chunks);
    }
    fflush(g.tl);
    g.tl_ops.clear();
    g.tl_next = 0;
}
