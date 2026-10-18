#include "moe-stream.h"

#include "ggml.h"

#include <atomic>
#include <mutex>
#include <unordered_map>

// hybrid MoE: the experts computed on another device, per weight (see ggml_moe_stream_set_gpu_map)
namespace {
std::mutex gpu_maps_mtx;
std::unordered_map<const ggml_tensor *, const int32_t *> gpu_maps;
std::atomic<int> gpu_maps_n{0};
}

void ggml_moe_stream_set_gpu_map(const struct ggml_tensor * w, const int32_t * map) {
    std::lock_guard<std::mutex> lock(gpu_maps_mtx);
    if (map != nullptr) {
        gpu_maps[w] = map;
    } else {
        gpu_maps.erase(w);
    }
    gpu_maps_n.store((int) gpu_maps.size());
}

const int32_t * ggml_moe_stream_gpu_map(const struct ggml_tensor * w) {
    if (gpu_maps_n.load(std::memory_order_relaxed) == 0) {
        return nullptr;
    }
    static thread_local const ggml_tensor * last_w = nullptr;
    static thread_local const int32_t *     last_m = nullptr;
    if (w != last_w) {
        std::lock_guard<std::mutex> lock(gpu_maps_mtx);
        auto it = gpu_maps.find(w);
        last_m = it != gpu_maps.end() ? it->second : nullptr;
        last_w = w;
    }
    return last_m;
}

#if defined(__linux__)

#include <algorithm>
#include <array>
#include <deque>
#include <atomic>
#include <memory>
#include <cerrno>
#include <condition_variable>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

constexpr size_t SLOT = 2u << 20; // address space of one cached expert slice, the largest slice that can be cached
constexpr size_t PAGE = 4096;     // O_DIRECT alignment

struct tensor_info {
    bool    stream   = false;
    int     layer    = -1; // blk.<layer>.ffn_{gate,up,down}_exps
    int     kind     = -1; // 0: gate, 1: up, 2: down
    int     fd       = -1;
    size_t  off      = 0; // file offset of the data
    size_t  nb2      = 0;
    int64_t n_expert = 0;
    std::vector<int32_t> slot; // [n_expert], -1 if not cached
};

struct slot_info {
    tensor_info * t = nullptr;
    int64_t  e        = -1;
    size_t   resident = 0; // bytes at the start of the slot that may be backed by memory
    uint64_t gen      = 0;
    uint8_t  hinted   = 0;     // read by hint() (the next layer's lookahead) and not used since: 1 + the guess's rank
    uint8_t  pinned   = 0;     // being copied by get_cached() without the lock: not evictable
    bool     hint_read = false; // the slot's last read was started by hint() (stats of the waits)
    int32_t  prev = -1;
    int32_t  next = -1;
};

struct io_job {
    int    fd;
    size_t off;
    size_t len;
    char * dst;
    int    group;
    int32_t slot; // slot marked ready when the read is done, -1 for none
    int64_t t_submit = 0; // ns, for the latency stats
};

static int64_t moe_now_ns_job() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

constexpr int    MAX_STAGE    = 256;    // staging buffers of copy(), GGML_MOE_STREAM_STAGES (default 16)
constexpr int    WARM_GROUP0  = 4;      // I/O groups: 2: prepare() and its reads into the cache, 3: hint(),
constexpr int    STAGE_GROUP0 = 7;      // 4..6: the reads into the cache of copy() for the weights of a layer, 7..: the staging buffers
constexpr int    N_GROUPS     = STAGE_GROUP0 + MAX_STAGE;
constexpr size_t STAGE        = 16u << 20; // size of a staging buffer
constexpr size_t PAD      = 512;        // bytes past an expert read with it, CUDA MMQ can read past the last expert of a copy

struct moe_stream {
    std::atomic<size_t> budget{0}; // the cache size: budget_max, or less while the memory governor makes room for the rest of the process
    size_t budget_max = 0;         // GGML_MOE_STREAM_MIB
    std::atomic<size_t> bytes{0};  // sum of the resident bytes of all slots

    char * pool = nullptr;
    std::vector<slot_info> slots;
    std::unique_ptr<std::atomic<uint8_t>[]> ready; // [n_slots], the data of the slot has been read
    std::unique_ptr<std::atomic<int32_t>[]> parts; // [n_slots], pieces of the slot's read still in flight
    // reads into slots are split into pieces read in parallel (2026-10-13: 128 KiB instead of whole 0.6-1 MiB slices, +10% generation)
    static constexpr size_t PIECE = 128u << 10;
    std::vector<int32_t>   free_slots;
    int32_t  head = -1; // least recently used
    int32_t  tail = -1;
    uint64_t gen  = 0;
    int32_t  cand = -1;  // the slot evictable_head() found

    std::unordered_map<const ggml_tensor *, tensor_info> tensors;
    std::unordered_map<std::string, int> fds;

    std::vector<std::thread> workers;
    std::mutex mtx;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    std::vector<io_job> queue[2]; // 0: needed now, 1: lookahead, read when the first is empty
    size_t q_next[2] = {};
    size_t pending[N_GROUPS] = {};
    bool   io_error = false;

    int    n_stage = 16;
    char * stage[MAX_STAGE] = {};
    bool   stage_busy[MAX_STAGE] = {}; // an upload from the buffer may be in flight

    // large batches, see copy(): the planned reads of the weights of a layer
    struct run     { int64_t first, n; size_t abeg, len, boff, skip, size; };
    struct chunk   { tensor_info * t; std::vector<run> runs; size_t used = 0; };
    struct planned {
        tensor_info * t;
        int      group;                   // I/O group of its reads into the cache
        uint64_t seq_end;                 // its chunks end before this sequence number of the ring
        std::vector<int64_t> cached;      // experts uploaded from the cache
        std::vector<int64_t> warmed;      // experts read into the cache, uploaded once read
        bool spec = false;                // planned before the counts were known: all the experts that were not cached
        std::vector<uint8_t> use;         // for such a plan: the experts to upload (the others were read for nothing)
    };
    std::deque<chunk>    ring;            // planned chunks not yet uploaded, ring.front() has the sequence number ring_base
    uint64_t             ring_base = 0;
    uint64_t             ring_submitted = 0; // the chunks before have been submitted
    std::deque<planned>  plan_q;
    std::vector<int32_t> plan_counts;     // the counts the plan was made for
    std::unordered_map<int, std::array<tensor_info *, 3>> layer_ts; // the gate, up and down weights of each layer
    uint64_t ring_syncs = 0, plan_layers = 0, plan_drains = 0, spec_plans = 0, spec_unused = 0;

    std::vector<uint8_t> used;

    // stats
    bool     log = false;
    uint64_t n_prepare = 0;
    uint64_t hits = 0, misses = 0, fallbacks = 0, bytes_read = 0;
    double   t_io = 0.0;
    std::atomic<bool> ggml_moe_stream_waiter{false};
    double   t_io_wait = 0.0;
    double   t_wait_kind[2] = {}, t_wait_kind_tensor[2][3] = {}; // waits for reads of prepare() (0) and of hint() (1), by gate/up/down
    uint64_t n_wait_kind[2] = {};
    uint64_t hint_reads = 0, hint_useful = 0, hint_wasted = 0;
    // per rank of the guess (the router's best first): how often a lookahead read was used or evicted unused. 2026-10-15: rank 1
    // 95% used, rank 5 83%, rank 10 70% - every rank saves more stalls than it wastes reads, so all ten are read
    static constexpr int HINT_RANKS = 16;
    uint64_t hint_useful_rank[HINT_RANKS] = {}, hint_wasted_rank[HINT_RANKS] = {};
    // read latency per kind (0: misses of prepare(), 1: lookahead, 2: large batches): queue and read time in ns, count
    std::atomic<int64_t> lat_queue[3] = {}, lat_read[3] = {}, lat_n[3] = {}, lat_max[3] = {};
    double ct_cached = 0, ct_stage_wait = 0, ct_stage_upload = 0, ct_warm_wait = 0, ct_warm_upload = 0;
    uint64_t copy_hits = 0, copy_misses = 0, copy_warm = 0, copy_bytes = 0;
    uint64_t copy_fallbacks = 0; // used experts that no plan delivered, uploaded from the mapped file (see copy())
    size_t   expert_bytes = 0; // all streamed experts, set by the user
    std::vector<uint8_t> warm;
    std::vector<int32_t> order;
    double   t_copy = 0.0;
    // GGML_MOE_STREAM_TRACE=path: the routing of every MUL_MAT_ID for offline cache simulations: "L<layer> <tokens> <ids>" for the
    // batches computed here (ids the GPU computes prefixed with g), "P<layer> <expert>:<count> ..." for the plans of large batches
    FILE *   trace = nullptr;

    moe_stream(size_t budget_mib, int n_io, bool log) : budget_max(budget_mib << 20), log(log) {
        budget = budget_max;
        // virtual slots, only the pages of the cached slices are backed by memory
        const size_t n_slots = std::max<size_t>(64, budget_max / (128u << 10));
        void * p = mmap(nullptr, n_slots*SLOT, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) {
            fprintf(stderr, "%s: mmap of %zu slots failed: %s\n", __func__, n_slots, strerror(errno));
            return;
        }
        // a huge page would back a whole slot
        madvise(p, n_slots*SLOT, MADV_NOHUGEPAGE);
        // GGML_MOE_STREAM_STAGES: staging buffers of large batches (pinned RAM, 16 MiB each)
        if (const char * e = getenv("GGML_MOE_STREAM_STAGES")) {
            n_stage = std::max(2, std::min(MAX_STAGE, atoi(e)));
        }
        for (int i = 0; i < n_stage; ++i) {
            stage[i] = (char *) mmap(nullptr, STAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            GGML_ASSERT(stage[i] != MAP_FAILED);
        }
        pool = (char *) p;
        slots.resize(n_slots);
        ready.reset(new std::atomic<uint8_t>[n_slots]);
        parts.reset(new std::atomic<int32_t>[n_slots]);
        for (size_t s = 0; s < n_slots; ++s) {
            ready[s].store(1, std::memory_order_relaxed);
            parts[s].store(0, std::memory_order_relaxed);
        }
        free_slots.reserve(n_slots);
        for (size_t s = n_slots; s-- > 0;) {
            free_slots.push_back((int32_t) s);
        }
        for (int i = 0; i < n_io; ++i) {
            workers.emplace_back([this] { worker(); });
        }
        fprintf(stderr, "ggml_moe_stream: cache of %.0f MiB, %zu slots, %d I/O threads, %d staging buffers of %zu MiB\n",
            budget_max/1048576.0, n_slots, n_io, n_stage, STAGE >> 20);
        governor_start();
    }

    // ---- memory governor ----
    // The cache is anonymous memory and the service runs without swap, so under the memory limit of its cgroup (or when the
    // machine runs short) only file pages can be reclaimed: the mmapped model (the n-gram table read lazily, the dense
    // weights' pages), the binaries and libraries. Everything else the process allocates grows over a session (llama-server's
    // context checkpoints, 113 MiB each for this model, its RAM prompt cache, the vision encoder's buffers), and once
    // cache + rest reach the limit the file pages thrash: every page the compute thread touches is a major fault, and a
    // turn that took seconds never ends. The governor keeps a reserve of file memory instead: every 100 ms it measures
    // what the process uses besides the cache and the file pages, lowers the cache budget so that cache + rest + reserve
    // stay under the limit (evicting least recently used slots itself, so it acts while the compute thread is busy
    // elsewhere or idle), and raises it again, up to GGML_MOE_STREAM_MIB, once the rest has stayed smaller for 2 s (prompt
    // batches swing it by ~350 MiB several times a second: raising the budget at each dip would read experts only to evict them).
    // Other processes need memory too: another process grew to 13 GiB next to the server, and with a floor of 512 MiB the
    // cache shrank too late, then grew back whenever swapping that process out raised MemAvailable for a moment, until the swap was
    // full and the kernel killed the server. So the cache also leaves the machine GGML_MOE_STREAM_AVAIL_MIB of MemAvailable, less
    // what is missing of a swap reserve (a quarter of the swap, at most 2 GiB: once the others fill the swap, only RAM can take
    // their next allocation); it grows only into MemAvailable beyond that floor plus 512 MiB (1 GiB until 2026-10-15, which left
    // 0.5-1 GiB of RAM unused in sessions; each GiB of cache saves ~10% of generation's SSD reads), and not for 30 s after
    // the machine was short.
    //   GGML_MOE_STREAM_RESERVE_MIB (default 1536): file memory to keep under a cgroup limit (memory.high, else memory.max);
    //                                               0 disables the governor
    //   GGML_MOE_STREAM_AVAIL_MIB   (default 2048): MemAvailable to leave the machine; 0 ignores the machine
    std::thread gov_thread;
    std::string gov_cg;          // the cgroup v2 directory of the process, empty if none
    size_t gov_reserve = 0;
    size_t gov_avail   = 0;
    size_t gov_min     = 0;      // the budget is never lowered below this
    uint64_t gov_shrinks = 0, gov_evicted = 0;
    size_t   gov_low = SIZE_MAX; // lowest budget set

    static bool read_u64(const std::string & path, uint64_t & v) {
        FILE * f = fopen(path.c_str(), "r");
        if (f == nullptr) {
            return false;
        }
        char buf[64] = {};
        const bool ok = fgets(buf, sizeof(buf), f) != nullptr && buf[0] >= '0' && buf[0] <= '9';
        fclose(f);
        if (ok) {
            v = strtoull(buf, nullptr, 10);
        }
        return ok;
    }

    // the reclaimable file pages of the cgroup (memory.stat active_file + inactive_file)
    bool cg_file_bytes(uint64_t & v) const {
        FILE * f = fopen((gov_cg + "/memory.stat").c_str(), "r");
        if (f == nullptr) {
            return false;
        }
        char key[64];
        unsigned long long val;
        int n = 0;
        v = 0;
        while (n < 2 && fscanf(f, "%63s %llu", key, &val) == 2) {
            if (strcmp(key, "active_file") == 0 || strcmp(key, "inactive_file") == 0) {
                v += val;
                n++;
            }
        }
        fclose(f);
        return n == 2;
    }

    // the machine's MemAvailable, SwapTotal and SwapFree
    static bool mem_available(uint64_t & avail, uint64_t & swap_total, uint64_t & swap_free) {
        FILE * f = fopen("/proc/meminfo", "r");
        if (f == nullptr) {
            return false;
        }
        char line[256];
        int n = 0;
        while (n < 3 && fgets(line, sizeof(line), f)) {
            unsigned long long kb;
            if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
                avail = (uint64_t) kb << 10;
                n++;
            } else if (sscanf(line, "SwapTotal: %llu kB", &kb) == 1) {
                swap_total = (uint64_t) kb << 10;
                n++;
            } else if (sscanf(line, "SwapFree: %llu kB", &kb) == 1) {
                swap_free = (uint64_t) kb << 10;
                n++;
            }
        }
        fclose(f);
        return n == 3;
    }

    void governor_start() {
        const char * r = getenv("GGML_MOE_STREAM_RESERVE_MIB");
        gov_reserve = (size_t) (r ? atoll(r) : 1536) << 20;
        const char * a = getenv("GGML_MOE_STREAM_AVAIL_MIB");
        gov_avail = (size_t) (a ? atoll(a) : 2048) << 20;
        if (gov_reserve == 0) {
            return;
        }
        gov_min = std::min<size_t>(budget_max, 1024u << 20);
        if (FILE * f = fopen("/proc/self/cgroup", "r")) {
            char line[4096];
            while (fgets(line, sizeof(line), f)) {
                if (strncmp(line, "0::", 3) == 0) {
                    std::string rel(line + 3);
                    while (!rel.empty() && (rel.back() == '\n' || rel.back() == ' ')) {
                        rel.pop_back();
                    }
                    gov_cg = "/sys/fs/cgroup" + (rel == "/" ? std::string() : rel);
                }
            }
            fclose(f);
        }
        uint64_t limit = 0;
        const bool has_limit = !gov_cg.empty() && cg_limit(limit);
        fprintf(stderr, "ggml_moe_stream: memory governor: %s, file reserve %zu MiB, MemAvailable floor %zu MiB\n",
            has_limit ? ("cgroup limit " + std::to_string(limit >> 20) + " MiB").c_str() : "no cgroup limit",
            gov_reserve >> 20, gov_avail >> 20);
        gov_thread = std::thread([this] { governor(); });
    }

    // the cgroup's limit: memory.high, else memory.max ("max" = none)
    bool cg_limit(uint64_t & v) const {
        uint64_t hi = 0, mx = 0;
        const bool has_hi = read_u64(gov_cg + "/memory.high", hi);
        const bool has_mx = read_u64(gov_cg + "/memory.max", mx);
        if (!has_hi && !has_mx) {
            return false;
        }
        v = has_hi && has_mx ? std::min(hi, mx) : has_hi ? hi : mx;
        return true;
    }

    void governor() {
        // off the compute threads' cores, like the readers
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
        const size_t GROW_STEP   = 256u << 20;  // raise the budget only by at least this much,
        const int    GROW_WAIT   = 20;          // and only after this many polls in a row allowed it (to the least they allowed)
        const size_t AVAIL_ABOVE = 512u << 20;  // grow only into MemAvailable beyond the floor plus this
        const int    SHORT_HOLD  = 300;         // polls (30 s) without growing after the machine was short
        int    grow_polls = 0;
        size_t grow_to    = 0;
        int    hold       = 0;
        size_t last_logged = budget_max;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            size_t target   = budget_max;
            size_t grow_cap = SIZE_MAX;
            uint64_t limit = 0, cur = 0, file = 0, avail = 0, swap_total = 0, swap_free = 0;
            const size_t b_now = bytes.load(std::memory_order_relaxed);
            if (!gov_cg.empty() && cg_limit(limit) && read_u64(gov_cg + "/memory.current", cur) && cg_file_bytes(file)) {
                // what the process uses besides the cache and the reclaimable file pages
                const uint64_t rest = cur > file + b_now ? cur - file - b_now : 0;
                target = limit > rest + gov_reserve ? (size_t) std::min<uint64_t>(target, limit - rest - gov_reserve) : 0;
            }
            if (gov_avail > 0 && mem_available(avail, swap_total, swap_free)) {
                const uint64_t swap_floor = std::min<uint64_t>(swap_total / 4, 2048ull << 20);
                const uint64_t swap_short = swap_free < swap_floor ? swap_floor - swap_free : 0;
                const uint64_t eff        = avail > swap_short ? avail - swap_short : 0;
                if (eff < gov_avail) {
                    const size_t short_by = (size_t) (gov_avail - eff);
                    target = std::min(target, b_now > short_by ? b_now - short_by : 0);
                    hold   = SHORT_HOLD;
                } else {
                    grow_cap = b_now + (size_t) (eff > gov_avail + AVAIL_ABOVE ? eff - gov_avail - AVAIL_ABOVE : 0);
                }
            }
            target = std::max(target, gov_min) & ~(size_t) ((1u << 20) - 1);
            const size_t b = budget.load(std::memory_order_relaxed);
            // what a raise may reach: grow_cap limits growth only, lowering the budget is the floor's job above
            const size_t grow_target = std::min(target, std::max(grow_cap, b)) & ~(size_t) ((1u << 20) - 1);
            if (target < b) {
                grow_polls = 0;
                budget.store(target, std::memory_order_relaxed);
                gov_low = std::min(gov_low, target);
                gov_shrinks++;
                shrink();
            } else if (hold > 0) {
                hold--;
                grow_polls = 0;
                continue;
            } else if (grow_target >= b + GROW_STEP || (grow_target == budget_max && b != budget_max)) {
                grow_to = grow_polls++ == 0 ? grow_target : std::min(grow_to, grow_target);
                if (grow_polls < GROW_WAIT) {
                    continue;
                }
                grow_polls = 0;
                budget.store(grow_to, std::memory_order_relaxed);
            } else {
                grow_polls = 0;
                continue;
            }
            const size_t nb = budget.load(std::memory_order_relaxed);
            if (log && (nb + (256u << 20) <= last_logged || nb >= last_logged + (256u << 20) || (nb == budget_max && last_logged != budget_max))) {
                fprintf(stderr, "ggml_moe_stream: memory governor: cache budget %zu MiB (limit %llu MiB, process %llu MiB, file %llu MiB, cache %zu MiB, MemAvailable %llu MiB, swap free %llu MiB)\n",
                    nb >> 20, (unsigned long long) (limit >> 20), (unsigned long long) (cur >> 20), (unsigned long long) (file >> 20),
                    bytes.load() >> 20, (unsigned long long) (avail >> 20), (unsigned long long) (swap_free >> 20));
                last_logged = nb;
            }
        }
    }

    // evict least recently used slots until the cache fits the budget; the slots of the current prepare or plan (gen),
    // the ones being read or copied by get_cached() are kept. The memory is released without the lock, in batches.
    void shrink() {
        std::vector<int32_t> victims;
        for (;;) {
            victims.clear();
            {
                std::unique_lock<std::mutex> lock(mtx);
                const size_t b = budget.load(std::memory_order_relaxed);
                size_t over = bytes > b ? bytes - b : 0;
                for (int32_t s = head; s >= 0 && over > 0 && victims.size() < 256; ) {
                    const int32_t next = slots[s].next;
                    if (slots[s].gen != gen && ready[s].load(std::memory_order_acquire) != 0 && slots[s].pinned == 0) {
                        over -= std::min(over, slots[s].resident);
                        evict(s);
                        victims.push_back(s);
                    }
                    s = next;
                }
            }
            if (victims.empty()) {
                return;
            }
            // evicted slots are in neither the LRU list nor the free list: nobody else touches them meanwhile
            size_t freed = 0;
            for (int32_t s : victims) {
                if (slots[s].resident > 0) {
                    madvise(pool + (size_t) s*SLOT, slots[s].resident, MADV_DONTNEED);
                    freed += slots[s].resident;
                }
            }
            {
                std::unique_lock<std::mutex> lock(mtx);
                for (int32_t s : victims) {
                    slots[s].resident = 0;
                    free_slots.push_back(s);
                }
                bytes -= freed;
            }
            gov_evicted += victims.size();
        }
    }

    void worker() {
        // GGML_MOE_STREAM_IO_CPUS=a-b: keep the readers off the cores that the compute threads spin on
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
        std::unique_lock<std::mutex> lock(mtx);
        for (;;) {
            cv_work.wait(lock, [this] { return q_next[0] < queue[0].size() || q_next[1] < queue[1].size(); });
            const int q = q_next[0] < queue[0].size() ? 0 : 1;
            const io_job job = queue[q][q_next[q]++];
            lock.unlock();
            const int64_t t_start = moe_now_ns_job();
            size_t done = 0;
            bool ok = true;
            while (done < job.len) {
                const ssize_t n = pread(job.fd, job.dst + done, job.len - done, job.off + done);
                if (n < 0 && errno == EINTR) {
                    continue;
                }
                if (n <= 0) {
                    ok = n == 0; // end of file: the aligned tail past the end is not needed
                    break;
                }
                done += n;
            }
            if (job.slot >= 0) {
                if (!ok) {
                    ready[job.slot].store(2, std::memory_order_release);
                } else if (parts[job.slot].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    ready[job.slot].store(1, std::memory_order_release);
                }
            }
            {
                const int k = job.group == 2 ? 0 : job.group == 3 ? 1 : 2;
                const int64_t t_end = moe_now_ns_job();
                lat_queue[k] += t_start - job.t_submit;
                lat_read[k]  += t_end - t_start;
                lat_n[k]++;
                int64_t m = lat_max[k].load();
                while (t_end - job.t_submit > m && !lat_max[k].compare_exchange_weak(m, t_end - job.t_submit)) {}
            }
            lock.lock();
            if (!ok) {
                io_error = true;
            }
            if (--pending[job.group] == 0) {
                cv_done.notify_all();
            }
        }
    }

    // with mtx locked
    void submit(int fd, size_t off, size_t len, char * dst, int group, int32_t slot = -1) {
        const int q = group == 3 ? 1 : 0;
        if (q_next[q] == queue[q].size()) {
            queue[q].clear();
            q_next[q] = 0;
        }
        // split large reads so that they run in parallel; a slot is ready when all the pieces of its read are
        const size_t PIECE = slot >= 0 ? moe_stream::PIECE : (size_t) (2u << 20);
        if (slot >= 0) {
            parts[slot].store((int32_t) ((len + PIECE - 1) / PIECE), std::memory_order_relaxed);
        }
        const int64_t t_submit = moe_now_ns_job();
        for (size_t o = 0; o < len; o += PIECE) {
            queue[q].push_back({ fd, off + o, std::min(PIECE, len - o), dst + o, group, slot, t_submit });
            pending[group]++;
        }
        cv_work.notify_all();
    }

    // with mtx locked
    void wait(std::unique_lock<std::mutex> & lock, int group) {
        cv_done.wait(lock, [&] { return pending[group] == 0; });
        if (io_error) {
            GGML_ABORT("ggml_moe_stream: an expert read failed");
        }
    }

    tensor_info & info(const ggml_tensor * w) {
        auto it = tensors.find(w);
        if (it != tensors.end()) {
            return it->second;
        }
        tensor_info & ti = tensors[w];
        ti.nb2      = w->nb[2];
        ti.n_expert = w->ne[2];
        ti.slot.assign(ti.n_expert, -1);
        {
            int layer = -1;
            char kind[16] = {};
            if (sscanf(w->name, "blk.%d.ffn_%15[a-z]_exps", &layer, kind) == 2) {
                const std::string k(kind);
                ti.kind  = k == "gate" ? 0 : k == "up" ? 1 : k == "down" ? 2 : -1;
                ti.layer = ti.kind >= 0 ? layer : -1;
            }
        }

        // find the file mapping that holds the data
        const uintptr_t addr = (uintptr_t) w->data;
        FILE * f = fopen("/proc/self/maps", "r");
        char line[4096];
        while (f && fgets(line, sizeof(line), f)) {
            unsigned long beg, end, off;
            char perms[8];
            int path_pos = 0;
            if (sscanf(line, "%lx-%lx %7s %lx %*s %*s %n", &beg, &end, perms, &off, &path_pos) < 4 || path_pos == 0) {
                continue;
            }
            if (addr < beg || addr + ggml_nbytes(w) > end) {
                continue;
            }
            std::string path(line + path_pos);
            while (!path.empty() && (path.back() == '\n' || path.back() == ' ')) {
                path.pop_back();
            }
            if (path.empty() || path[0] != '/' || ti.nb2 + PAD + 2*PAGE > SLOT) {
                break;
            }
            auto fit = fds.find(path);
            if (fit == fds.end()) {
                fit = fds.emplace(path, open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC)).first;
            }
            if (fit->second >= 0) {
                ti.fd     = fit->second;
                ti.off    = addr - beg + off;
                ti.stream = true;
            }
            break;
        }
        if (f) {
            fclose(f);
        }
        if (log) {
            fprintf(stderr, "ggml_moe_stream: %s: %s\n", w->name, ti.stream ? "streamed" : "not file backed, left as is");
        }
        if (ti.stream && ti.layer >= 0) {
            layer_ts[ti.layer][ti.kind] = &ti;
        }
        return ti;
    }

    void list_unlink(int32_t s, int32_t & h, int32_t & t) {
        slot_info & si = slots[s];
        if (si.prev >= 0) { slots[si.prev].next = si.next; } else { h = si.next; }
        if (si.next >= 0) { slots[si.next].prev = si.prev; } else { t = si.prev; }
        si.prev = si.next = -1;
    }

    void list_push_head(int32_t s, int32_t & h, int32_t & t) {
        slot_info & si = slots[s];
        si.prev = -1;
        si.next = h;
        if (h >= 0) { slots[h].prev = s; } else { t = s; }
        h = s;
    }

    void list_push_tail(int32_t s, int32_t & h, int32_t & t) {
        slot_info & si = slots[s];
        si.prev = t;
        si.next = -1;
        if (t >= 0) { slots[t].next = s; } else { h = s; }
        t = s;
    }

    // one LRU list (head = least recently used). A segmented LRU that kept generation's experts from prompt batches was tried
    // (2026-10-14) and removed (2026-10-15): prompt batches warm a share of experts the size of the whole budget, so a protected
    // part makes them evict what the next batch reads (a session trace showed 50% more prompt reads, generation unchanged)
    void unlink(int32_t s)    { list_unlink(s, head, tail); }
    void push_head(int32_t s) { list_push_head(s, head, tail); }
    void push_tail(int32_t s) { list_push_tail(s, head, tail); }

    void release(int32_t s, size_t keep) {
        slot_info & si = slots[s];
        if (si.resident > keep) {
            madvise(pool + (size_t) s*SLOT + keep, si.resident - keep, MADV_DONTNEED);
            bytes -= si.resident - keep;
            si.resident = keep;
        }
    }

    void evict(int32_t s) {
        slot_info & si = slots[s];
        if (si.hinted) {
            hint_wasted++;
            hint_wasted_rank[std::min(si.hinted - 1, HINT_RANKS - 1)]++;
            si.hinted = 0;
        }
        unlink(s);
        si.t->slot[si.e] = -1;
        si.t = nullptr;
        si.e = -1;
    }

    // a slot for a slice of need bytes, -1 if the cache is full with the experts of the current prepare
    // the least recently used slot can be evicted unless its read is in flight
    // the least recently used evictable slot (cand)
    bool evictable_head() {
        for (int i = 0; i < 64 && head >= 0 && slots[head].gen != gen; ++i) {
            if (ready[head].load(std::memory_order_acquire) != 0 && slots[head].pinned == 0) {
                cand = head;
                return true;
            }
            const int32_t s = head;
            unlink(s);
            push_tail(s);
        }
        return false;
    }

    int32_t alloc(size_t need) {
        // a full cache reuses the memory of the evicted slot as it is; releasing it (madvise, a TLB shootdown on every core,
        // under the lock the readers wait on) and faulting it in again before the read costs latency (2026-10-13)
        if (bytes + need > budget && evictable_head()) {
            const int32_t s = cand;
            evict(s);
            while (bytes - slots[s].resident + need > budget && evictable_head()) {
                const int32_t v = cand;
                evict(v);
                release(v, 0);
                free_slots.push_back(v);
            }
            slot_info & si = slots[s];
            if (bytes - si.resident + need > budget) {
                release(s, 0);
                free_slots.push_back(s);
                return -1;
            }
            if (si.resident > need) {
                release(s, need);
            } else {
                bytes += need - si.resident;
                si.resident = need;
            }
            return s;
        }
        if (bytes + need > budget) {
            return -1;
        }
        int32_t s = -1;
        if (!free_slots.empty()) {
            s = free_slots.back();
            free_slots.pop_back();
        } else if (evictable_head()) {
            s = cand;
            evict(s);
        } else {
            return -1;
        }
        slot_info & si = slots[s];
        if (si.resident > need) {
            release(s, need);
        } else {
            bytes += need - si.resident;
            si.resident = need;
        }
        return s;
    }

    void prepare(const ggml_cgraph * cgraph, int node_n) {
        const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), node_n);
        const ggml_tensor * ids  = node->src[2];
        const int64_t n_as = node->src[0]->ne[2];

        // the experts the GPU computes from its own copies are not used here: their RAM copies age out like unused ones, which
        // leaves the cache to the others (2026-10-15, session trace: 6% fewer SSD reads in generation than refreshing them)
        const int32_t * gpu_map = ggml_moe_stream_gpu_map(node->src[0]);
        used.assign(n_as, 0);
        for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) {
            for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
                const int32_t id = *(const int32_t *) ((const char *) ids->data + i1*ids->nb[1] + i0*ids->nb[0]);
                if (id >= 0 && id < n_as) {
                    used[id] = gpu_map == nullptr || gpu_map[id] < 0;
                }
            }
        }

        std::vector<tensor_info *> ws;
        const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(cgraph));
        for (int j = node_n; j < n_nodes; ++j) {
            const ggml_tensor * n = ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), j);
            if (n->op == GGML_OP_MUL_MAT_ID && n->src[2] == ids && n->src[0]->ne[2] == n_as) {
                tensor_info & ti = info(n->src[0]);
                if (ti.stream) {
                    ws.push_back(&ti);
                }
            }
        }

        // the LRU list and the slots change under the lock: the memory governor evicts from another thread
        std::unique_lock<std::mutex> lock(mtx);
        gen++;
        n_prepare++;
        if (trace && !ws.empty()) {
            fprintf(trace, "L%d %lld", ws[0]->layer, (long long) ids->ne[1]);
            for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
                    const int32_t id = *(const int32_t *) ((const char *) ids->data + i1*ids->nb[1] + i0*ids->nb[0]);
                    fprintf(trace, gpu_map && id >= 0 && id < n_as && gpu_map[id] >= 0 ? " g%d" : " %d", id);
                }
            }
            fputc('\n', trace);
        }

        // the hits first, so that the misses cannot evict them
        for (tensor_info * ti : ws) {
            for (int64_t e = 0; e < n_as; ++e) {
                if (used[e] && ti->slot[e] >= 0) {
                    const int32_t s = ti->slot[e];
                    unlink(s);
                    push_tail(s);
                    slots[s].gen = gen;
                    if (slots[s].hinted) {
                        hint_useful++;
                        hint_useful_rank[std::min(slots[s].hinted - 1, HINT_RANKS - 1)]++;
                        slots[s].hinted = 0;
                    } else {
                        hits++;
                    }
                }
            }
        }

        for (tensor_info * ti : ws) {
            for (int64_t e = 0; e < n_as; ++e) {
                if (!used[e] || ti->slot[e] >= 0) {
                    continue;
                }
                const size_t beg  = ti->off + e*ti->nb2;
                const size_t abeg = beg & ~(PAGE - 1);
                const size_t aend = (beg + ti->nb2 + PAD + PAGE - 1) & ~(PAGE - 1);
                const int32_t s = alloc(aend - abeg);
                if (s < 0) {
                    fallbacks++;
                    continue;
                }
                slot_info & si = slots[s];
                si.t   = ti;
                si.e   = e;
                si.gen = gen;
                push_tail(s);
                ti->slot[e] = s;
                ready[s].store(0, std::memory_order_relaxed);
                si.hint_read = false;
                submit(ti->fd, abeg, aend - abeg, pool + (size_t) s*SLOT, 2, s);
                misses++;
                bytes_read += aend - abeg;
            }
        }
        // the reads complete while the MUL_MAT_IDs compute the cached experts, see expert()
        lock.unlock();

        if (log && n_prepare % 4800 == 0) {
            print_stats();
        }
    }

    // ---- large batches, whose MUL_MAT_ID runs on another device: copy() passes the used experts of a weight to set() ----
    // The gate, up and down weights of a layer select the same experts, so the copy of the first one plans the reads of all
    // three: the experts the batch uses most replace the least recently used ones in the cache, as generation will likely use
    // them (see warm), the other missing experts are packed into a ring of pinned staging buffers in upload order. The disk
    // thus keeps reading the later weights of the layer while the earlier ones upload and while the GPU computes, instead of
    // starting each weight's reads at its copy.

    // a slot for a read of copy(): the least recently used one that is not used by the plan (gen) and whose read is done
    int32_t alloc_copy(size_t need) {
        auto victim = [&]() -> int32_t {
            int32_t s = head;
            for (int i = 0; i < 4096 && s >= 0; ++i, s = slots[s].next) {
                if (slots[s].gen != gen && ready[s].load(std::memory_order_acquire) != 0 && slots[s].pinned == 0) {
                    return s;
                }
            }
            return -1;
        };
        while (bytes + need > budget) {
            const int32_t s = victim();
            if (s < 0) {
                return -1;
            }
            evict(s);
            release(s, 0);
            free_slots.push_back(s);
        }
        int32_t s = -1;
        if (!free_slots.empty()) {
            s = free_slots.back();
            free_slots.pop_back();
        } else if ((s = victim()) >= 0) {
            evict(s);
        } else {
            return -1;
        }
        slot_info & si = slots[s];
        if (si.resident > need) {
            release(s, need);
        } else {
            bytes += need - si.resident;
            si.resident = need;
        }
        return s;
    }

    // with mtx locked: append the runs of the experts e with staged[e] to the ring, packed into chunks of up to STAGE bytes
    // (the ring is empty when a plan starts, so no chunk that is being read is extended)
    void add_chunks(tensor_info * t, const std::vector<uint8_t> & staged) {
        const int64_t n_as = t->n_expert;
        const size_t max_n = (STAGE - 2*PAGE - PAD) / t->nb2;
        for (int64_t e = 0; e < n_as; ) {
            if (!staged[e]) {
                e++;
                continue;
            }
            int64_t n = 1;
            while (e + n < n_as && staged[e + n] && (size_t) n < max_n) {
                n++;
            }
            const size_t beg  = t->off + e*t->nb2;
            const size_t pad  = e + n < n_as ? PAD : 0;
            const size_t abeg = beg & ~(PAGE - 1);
            const size_t aend = (beg + n*t->nb2 + pad + PAGE - 1) & ~(PAGE - 1);
            if (ring.empty() || ring.back().t != t || ring.back().used + (aend - abeg) > STAGE) {
                ring.push_back({ t, {}, 0 });
            }
            chunk & c = ring.back();
            c.runs.push_back({ e, n, abeg, aend - abeg, c.used, beg - abeg, n*t->nb2 + pad });
            c.used += aend - abeg;
            copy_bytes  += aend - abeg;
            copy_misses += n;
            e += n;
        }
    }

    // with mtx locked: plan the copies of the weights of the layer of ti (only ti if the layer is unknown or ti is not its first weight)
    // spec: the plan made for ti before its counts were known (see copy()), completed here
    void plan_layer(tensor_info & ti, const int32_t * counts, planned * spec = nullptr) {
        const int64_t n_as = ti.n_expert;
        std::array<tensor_info *, 3> ts = { &ti, nullptr, nullptr };
        if (ti.kind == 0) {
            auto it = layer_ts.find(ti.layer);
            if (it != layer_ts.end() && it->second[0] == &ti) {
                ts = it->second;
            }
        }
        gen++;
        plan_layers++;
        plan_counts.assign(counts, counts + n_as);
        if (trace) {
            fprintf(trace, "P%d", ti.layer);
            for (int64_t e = 0; e < n_as; ++e) {
                if (counts[e] > 0) {
                    fprintf(trace, " %lld:%d", (long long) e, counts[e]);
                }
            }
            fputc('\n', trace);
        }

        // the experts to keep in the cache: the top fraction of the batch, the fraction being the share of all experts the cache can hold
        warm.assign(n_as, 0);
        if (expert_bytes > 0) {
            const int64_t n_keep = std::min<int64_t>(n_as, (int64_t) ((double) budget/expert_bytes*n_as));
            order.resize(n_as);
            for (int64_t e = 0; e < n_as; ++e) {
                order[e] = (int32_t) e;
            }
            std::partial_sort(order.begin(), order.begin() + n_keep, order.end(), [&](int32_t a, int32_t b) { return counts[a] > counts[b]; });
            for (int64_t i = 0; i < n_keep && counts[order[i]] > 0; ++i) {
                warm[order[i]] = 1;
            }
        }

        // the cached experts of all weights are uploaded from their slots: no read of this plan may evict them
        for (tensor_info * t : ts) {
            if (t == nullptr || !t->stream || t->n_expert != n_as) {
                continue;
            }
            for (int64_t e = 0; e < n_as; ++e) {
                const int32_t sl = t->slot[e];
                if (counts[e] > 0 && sl >= 0) {
                    slots[sl].gen = gen;
                    if (warm[e]) {
                        unlink(sl);
                        push_tail(sl);
                    }
                }
            }
        }

        for (int k = 0; k < 3; ++k) {
            tensor_info * t = ts[k];
            if (t == nullptr || !t->stream || t->n_expert != n_as) {
                continue;
            }
            if (k == 0 && spec != nullptr) {
                // its missing experts are already being read: upload the used ones from the cache or the ring, skip the others
                spec->spec = false;
                spec->use.assign(n_as, 0);
                for (int64_t e = 0; e < n_as; ++e) {
                    spec->use[e] = counts[e] > 0;
                    if (counts[e] > 0 && t->slot[e] >= 0) {
                        spec->cached.push_back(e);
                    }
                }
                continue;
            }
            planned p;
            p.t     = t;
            p.group = WARM_GROUP0 + k;
            GGML_ASSERT(pending[p.group] == 0);
            for (int64_t e = 0; e < n_as; ++e) {
                if (counts[e] > 0 && t->slot[e] >= 0) {
                    p.cached.push_back(e);
                }
            }
            // the missing experts to keep are read into new slots
            std::vector<uint8_t> staged(n_as, 0);
            for (int64_t e = 0; e < n_as; ++e) {
                if (counts[e] == 0 || t->slot[e] >= 0) {
                    continue;
                }
                if (warm[e]) {
                    const size_t beg  = t->off + e*t->nb2;
                    const size_t abeg = beg & ~(PAGE - 1);
                    const size_t aend = (beg + t->nb2 + PAD + PAGE - 1) & ~(PAGE - 1);
                    const int32_t sl = alloc_copy(aend - abeg);
                    if (sl >= 0) {
                        slot_info & si = slots[sl];
                        si.t   = t;
                        si.e   = e;
                        si.gen = gen;
                        push_tail(sl);
                        t->slot[e] = sl;
                        ready[sl].store(0, std::memory_order_relaxed);
                        submit(t->fd, abeg, aend - abeg, pool + (size_t) sl*SLOT, p.group, sl);
                        p.warmed.push_back(e);
                        copy_warm++;
                        copy_bytes += aend - abeg;
                        continue;
                    }
                }
                staged[e] = 1;
            }
            add_chunks(t, staged);
            p.seq_end = ring_base + ring.size();
            plan_q.push_back(std::move(p));
        }
    }

    // with mtx locked: submit the planned chunks whose staging buffer is free
    // a buffer is free once its chunk has been uploaded and the upload is done, which needs a sync: wait for a quarter
    // of the ring to be blocked by uploads (or for the disk to run dry) so that one sync frees many buffers
    void ring_submit(std::unique_lock<std::mutex> & lock, void (*sync)(void * ctx), void * ctx, bool force_sync) {
        const uint64_t limit = std::min<uint64_t>(ring_base + ring.size(), ring_base + n_stage);
        if (ring_submitted >= limit) {
            return;
        }
        int n_blocked = 0;
        for (uint64_t q = ring_submitted; q < limit; ++q) {
            n_blocked += stage_busy[q % n_stage] ? 1 : 0;
        }
        const bool dry = ring_submitted == ring_base; // nothing in flight
        if (n_blocked > 0 && (force_sync || dry || n_blocked >= std::max(1, n_stage/4))) {
            lock.unlock();
            sync(ctx);
            lock.lock();
            std::fill(stage_busy, stage_busy + n_stage, false);
            ring_syncs++;
        }
        while (ring_submitted < limit && !stage_busy[ring_submitted % n_stage]) {
            const int b = (int) (ring_submitted % n_stage);
            const chunk & c = ring[ring_submitted - ring_base];
            for (const run & r : c.runs) {
                submit(c.t->fd, r.abeg, r.len, stage[b] + r.boff, STAGE_GROUP0 + b);
            }
            ring_submitted++;
        }
    }

    // with mtx locked: wait for the reads of all plans and drop them (the plan did not match the copies)
    void drain_plans(std::unique_lock<std::mutex> & lock) {
        for (int g = WARM_GROUP0; g < STAGE_GROUP0 + n_stage; ++g) {
            wait(lock, g);
        }
        ring.clear();
        ring_base      = ring_submitted;
        plan_q.clear();
        plan_drains++;
    }

    bool copy(const ggml_tensor * w, const int32_t * counts, void (*set)(void * ctx, size_t offset, const void * data, size_t size), void (*sync)(void * ctx), void * ctx) {
        tensor_info & ti = info(w);
        if (!ti.stream) {
            return false;
        }
        const int64_t n_as = ti.n_expert;
        auto tnow = [] { return std::chrono::steady_clock::now(); };
        auto tsec = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
        const auto t0 = tnow();

        std::unique_lock<std::mutex> lock(mtx);

        // the plan of this layer, made by the copy of its first weight (or begun by the copy of the previous layer's last weight)
        if (!plan_q.empty() && plan_q.front().t == &ti && plan_q.front().spec) {
            plan_layer(ti, counts, &plan_q.front());
        } else {
            const bool planned_here = plan_q.empty() || plan_q.front().t != &ti ||
                (int64_t) plan_counts.size() != n_as || !std::equal(counts, counts + n_as, plan_counts.begin());
            if (planned_here) {
                if (!plan_q.empty()) {
                    drain_plans(lock);
                }
                plan_layer(ti, counts);
            }
        }
        ring_submit(lock, sync, ctx, false);
        planned p = std::move(plan_q.front());
        plan_q.pop_front();

        // the cached experts (their uploads from pageable memory return once the data is copied)
        lock.unlock();
        const auto tc0 = tnow();
        auto expert_data = [&](int32_t s, int64_t e) {
            return pool + (size_t) s*SLOT + ((ti.off + e*ti.nb2) & (PAGE - 1));
        };
        auto expert_size = [&](int64_t e) {
            return ti.nb2 + (e < n_as - 1 ? PAD : 0);
        };
        std::vector<uint8_t> done(n_as, 0);
        for (int64_t e : p.cached) {
            set(ctx, e*ti.nb2, expert_data(ti.slot[e], e), expert_size(e));
            done[e] = 1;
        }
        copy_hits += p.cached.size();
        ct_cached += tsec(tc0, tnow());
        lock.lock();

        // the staged chunks in order, refilling the ring as buffers free up
        while (ring_base < p.seq_end) {
            ring_submit(lock, sync, ctx, ring_submitted == ring_base);
            GGML_ASSERT(ring_submitted > ring_base);
            const int b = (int) (ring_base % n_stage);
            const auto tw = tnow();
            wait(lock, STAGE_GROUP0 + b);
            const chunk c = std::move(ring.front());
            ring.pop_front();
            ring_base++;
            lock.unlock();
            const auto tu = tnow();
            ct_stage_wait += tsec(tw, tu);
            for (const run & r : c.runs) {
                if (p.use.empty()) {
                    set(ctx, r.first*ti.nb2, stage[b] + r.boff + r.skip, r.size);
                    std::fill(done.begin() + r.first, done.begin() + r.first + r.n, 1);
                    continue;
                }
                // the used experts of the run, in runs of their own (the bytes past a sub-run are the next expert's or the run's pad)
                for (int64_t e = r.first; e < r.first + r.n; ) {
                    if (!p.use[e]) {
                        spec_unused++;
                        e++;
                        continue;
                    }
                    int64_t e1 = e + 1;
                    while (e1 < r.first + r.n && p.use[e1]) {
                        e1++;
                    }
                    const size_t size = (e1 - e)*ti.nb2 + (e1 < n_as ? PAD : 0);
                    set(ctx, e*ti.nb2, stage[b] + r.boff + r.skip + (e - r.first)*ti.nb2, size);
                    std::fill(done.begin() + e, done.begin() + e1, 1);
                    e = e1;
                }
            }
            ct_stage_upload += tsec(tu, tnow());
            lock.lock();
            stage_busy[b] = true;
        }

        // the experts read into the cache
        const auto tw2 = tnow();
        wait(lock, p.group);
        lock.unlock();
        const auto tw3 = tnow();
        ct_warm_wait += tsec(tw2, tw3);
        for (int64_t e : p.warmed) {
            set(ctx, e*ti.nb2, expert_data(ti.slot[e], e), expert_size(e));
            done[e] = 1;
        }
        ct_warm_upload += tsec(tw3, tnow());

        // a used expert that no plan delivered (a slot the plan counted on was evicted before the plan protected it) is uploaded
        // from the mapped file: slower, the same bytes, and the GPU never computes with an expert missing
        uint64_t n_fallback = 0;
        for (int64_t e = 0; e < n_as; ++e) {
            if (counts[e] > 0 && !done[e]) {
                set(ctx, e*ti.nb2, (const char *) w->data + e*ti.nb2, expert_size(e));
                n_fallback++;
            }
        }
        if (n_fallback > 0) {
            if (log && copy_fallbacks == 0) {
                fprintf(stderr, "ggml_moe_stream: large batch: %llu used experts of %s were not planned, uploaded from the file\n",
                    (unsigned long long) n_fallback, w->name);
            }
            copy_fallbacks += n_fallback;
        }

        // keep the disk busy with the next weights of the layer while the GPU computes this one
        lock.lock();
        // when this batch used nearly all experts of the layer, start reading all missing experts of the next layer's first weight
        // before its counts are known (not into the final layer, whose prompt batches compute only the outputs; 2026-10-13, +4%)
        if (ti.kind == 2 && plan_q.empty() && ti.layer >= 0) {
            int64_t n_used = 0;
            for (int64_t e = 0; e < n_as; ++e) {
                n_used += counts[e] > 0;
            }
            auto it   = layer_ts.find(ti.layer + 1);
            auto last = layer_ts.find(ti.layer + 2);
            if (n_used*10 >= n_as*9 && it != layer_ts.end() && last != layer_ts.end() && it->second[0] != nullptr) {
                tensor_info * g = it->second[0];
                if (g->stream && g->n_expert == n_as) {
                    // the cached experts are not read: they must stay until plan_layer() protects the used ones with the next gen.
                    // Mark them with this gen meanwhile, or the memory governor may evict one, which then nobody reads (2026-10-14:
                    // the cache at its MemAvailable floor shrank mid-prompt and the server aborted)
                    std::vector<uint8_t> staged(n_as, 0);
                    for (int64_t e = 0; e < n_as; ++e) {
                        staged[e] = g->slot[e] < 0;
                        if (!staged[e]) {
                            slots[g->slot[e]].gen = gen;
                        }
                    }
                    planned sp;
                    sp.t     = g;
                    sp.group = WARM_GROUP0;
                    sp.spec  = true;
                    add_chunks(g, staged);
                    sp.seq_end = ring_base + ring.size();
                    plan_q.push_back(std::move(sp));
                    spec_plans++;
                }
            }
        }
        if (ring_submitted < ring_base + ring.size()) {
            ring_submit(lock, sync, ctx, true);
        }
        lock.unlock();

        t_copy += tsec(t0, tnow());
        return true;
    }

    // the staging buffers, for the user to pin them
    void staging(void ** bufs, size_t * size, int * n) const {
        for (int i = 0; i < n_stage; ++i) {
            bufs[i] = stage[i];
        }
        *size = STAGE;
        *n    = n_stage;
    }

    // lookahead: start reading the experts that ids will likely select in the MUL_MAT_IDs of ws, without waiting
    // (two layers ahead was tried and removed: 25% of its guesses were used)
    void hint(const ggml_tensor * const * ws, int n_ws, const ggml_tensor * ids) {
        if (ws[0] == nullptr || ids->ne[1] > 4) {
            return;
        }
        const int64_t n_as = ws[0]->ne[2];
        const int32_t * gpu_map = ggml_moe_stream_gpu_map(ws[0]);
        // used[e] = 1 + the best rank any token's router gave e (ids are ranked, see qwen4exp_moe_lookahead)
        used.assign(n_as, 0);
        for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) {
            for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
                const int32_t id = *(const int32_t *) ((const char *) ids->data + i1*ids->nb[1] + i0*ids->nb[0]);
                if (id >= 0 && id < n_as && (gpu_map == nullptr || gpu_map[id] < 0) && (used[id] == 0 || used[id] > i0 + 1)) {
                    used[id] = (uint8_t) (i0 + 1);
                }
            }
        }
        // the slots of the current layer are still in use
        std::unique_lock<std::mutex> lock(mtx);
        for (int i = 0; i < n_ws; ++i) {
            tensor_info & ti = info(ws[i]);
            if (!ti.stream || ti.n_expert != n_as) {
                continue;
            }
            for (int64_t e = 0; e < n_as; ++e) {
                if (!used[e] || ti.slot[e] >= 0) {
                    continue;
                }
                const size_t beg  = ti.off + e*ti.nb2;
                const size_t abeg = beg & ~(PAGE - 1);
                const size_t aend = (beg + ti.nb2 + PAD + PAGE - 1) & ~(PAGE - 1);
                const int32_t s = alloc(aend - abeg);
                if (s < 0) {
                    continue;
                }
                slot_info & si = slots[s];
                si.t      = &ti;
                si.e      = e;
                si.gen    = gen;
                si.hinted = used[e];
                push_tail(s);
                hint_reads++;
                ti.slot[e] = s;
                ready[s].store(0, std::memory_order_relaxed);
                si.hint_read = true;
                submit(ti.fd, abeg, aend - abeg, pool + (size_t) s*SLOT, 3, s);
            }
        }
    }

    void print_stats() const {
        const uint64_t n = hits + misses;
        fprintf(stderr, "ggml_moe_stream: %llu prepares, hit rate %.2f%%, read %.2f GiB, compute waited %.1f s for reads, %llu fallbacks, cache %.0f MiB\n",
            (unsigned long long) n_prepare, n ? 100.0*hits/n : 0.0, bytes_read/1073741824.0, t_io_wait,
            (unsigned long long) fallbacks, bytes/1048576.0);
        if (n_wait_kind[0] + n_wait_kind[1] > 0) {
            fprintf(stderr, "ggml_moe_stream: waits: misses %.1f s in %llu (gate %.1f up %.1f down %.1f), lookahead still reading %.1f s in %llu (gate %.1f up %.1f down %.1f)\n",
                t_wait_kind[0], (unsigned long long) n_wait_kind[0], t_wait_kind_tensor[0][0], t_wait_kind_tensor[0][1], t_wait_kind_tensor[0][2],
                t_wait_kind[1], (unsigned long long) n_wait_kind[1], t_wait_kind_tensor[1][0], t_wait_kind_tensor[1][1], t_wait_kind_tensor[1][2]);
        }
        if (gov_shrinks > 0) {
            fprintf(stderr, "ggml_moe_stream: memory governor: budget now %.0f MiB (lowest %.0f of %.0f MiB), %llu shrinks, %llu slots evicted\n",
                budget/1048576.0, gov_low/1048576.0, budget_max/1048576.0, (unsigned long long) gov_shrinks, (unsigned long long) gov_evicted);
        }
        if (hint_reads > 0) {
            fprintf(stderr, "ggml_moe_stream: lookahead: %llu reads, %llu used (%.1f%%), %llu evicted unused; used/wasted by rank:",
                (unsigned long long) hint_reads, (unsigned long long) hint_useful, 100.0*hint_useful/hint_reads, (unsigned long long) hint_wasted);
            for (int r = 0; r < HINT_RANKS; ++r) {
                if (hint_useful_rank[r] + hint_wasted_rank[r] > 0) {
                    fprintf(stderr, " %d:%llu/%llu", r + 1, (unsigned long long) hint_useful_rank[r], (unsigned long long) hint_wasted_rank[r]);
                }
            }
            fputc('\n', stderr);
        }
        static const char * lat_names[3] = { "misses", "lookahead", "large batches" };
        for (int k = 0; k < 3; ++k) {
            const int64_t n = lat_n[k].load();
            if (n > 0) {
                fprintf(stderr, "ggml_moe_stream: read latency %s: %lld reads, queue %.3f ms + read %.3f ms avg, max %.1f ms\n", lat_names[k],
                    (long long) n, lat_queue[k].load()/1e6/n, lat_read[k].load()/1e6/n, lat_max[k].load()/1e6);
            }
        }
        if (copy_hits + copy_misses > 0) {
            fprintf(stderr, "ggml_moe_stream: large batches: %llu experts from the cache, %llu read into the cache, %llu read (%.2f GiB in %.1f s, %.2f GB/s incl. upload)\n",
                (unsigned long long) copy_hits, (unsigned long long) copy_warm, (unsigned long long) copy_misses, copy_bytes/1073741824.0, t_copy, t_copy > 0 ? copy_bytes/t_copy/1e9 : 0.0);
            fprintf(stderr, "ggml_moe_stream: large batches time: cached uploads %.1f s, staged read waits %.1f s, staged uploads %.1f s, warm read waits %.1f s, warm uploads %.1f s; %llu layer plans, %llu dropped, %llu ring syncs, %llu speculative (%llu experts read unused)\n",
                ct_cached, ct_stage_wait, ct_stage_upload, ct_warm_wait, ct_warm_upload,
                (unsigned long long) plan_layers, (unsigned long long) plan_drains, (unsigned long long) ring_syncs,
                (unsigned long long) spec_plans, (unsigned long long) spec_unused);
            if (copy_fallbacks > 0) {
                fprintf(stderr, "ggml_moe_stream: large batches: %llu experts uploaded from the file (not planned)\n",
                    (unsigned long long) copy_fallbacks);
            }
        }
    }

    const char * expert(const ggml_tensor * w, int64_t e) {
        static thread_local const ggml_tensor * last_w = nullptr;
        static thread_local tensor_info *       last_t = nullptr;
        tensor_info * ti = last_t;
        if (w != last_w) {
            auto it = tensors.find(w);
            ti = it != tensors.end() ? &it->second : nullptr;
            last_w = w;
            last_t = ti;
        }
        if (ti != nullptr && ti->slot[e] >= 0) {
            const int32_t s = ti->slot[e];
            if (ready[s].load(std::memory_order_acquire) == 0) {
                const auto t0 = std::chrono::steady_clock::now();
                const bool from_hint = slots[s].hint_read;
                while (ready[s].load(std::memory_order_acquire) == 0) {
                    std::this_thread::yield();
                }
                // only one thread adds its wait, the waits of the threads overlap
                if (ggml_moe_stream_waiter.exchange(true) == false) {
                    const double w = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    t_io_wait += w;
                    t_wait_kind[from_hint] += w;
                    n_wait_kind[from_hint]++;
                    t_wait_kind_tensor[from_hint][std::max(0, std::min(2, ti->kind))] += w;
                    ggml_moe_stream_waiter.store(false);
                }
            }
            if (ready[s].load(std::memory_order_relaxed) == 2) {
                GGML_ABORT("ggml_moe_stream: an expert read failed");
            }
            const size_t beg = ti->off + e*ti->nb2;
            return pool + (size_t) s*SLOT + (beg & (PAGE - 1));
        }
        return (const char *) w->data + e*w->nb[2];
    }

    // the copy runs without the lock (the readers and the compute thread need it), the slot is pinned meanwhile
    bool get_cached(const ggml_tensor * w, int64_t e, void * dst, size_t size) {
        std::unique_lock<std::mutex> lock(mtx);
        auto it = tensors.find(w);
        if (it == tensors.end() || !it->second.stream || e < 0 || e >= it->second.n_expert || size > it->second.nb2) {
            return false;
        }
        const tensor_info & ti = it->second;
        const int32_t s = ti.slot[e];
        if (s < 0 || ready[s].load(std::memory_order_acquire) != 1) {
            return false;
        }
        const char * src = pool + (size_t) s*SLOT + ((ti.off + e*ti.nb2) & (PAGE - 1));
        slots[s].pinned++;
        lock.unlock();
        memcpy(dst, src, size);
        lock.lock();
        slots[s].pinned--;
        return true;
    }

    bool is_ready(const ggml_tensor * w, int64_t e) {
        auto it = tensors.find(w);
        if (it == tensors.end() || it->second.slot[e] < 0) {
            return true;
        }
        return ready[it->second.slot[e]].load(std::memory_order_acquire) != 0;
    }
};

moe_stream * g_stream = nullptr;

moe_stream * get_stream() {
    static bool init = false;
    if (!init) {
        init = true;
        const char * s = getenv("GGML_MOE_STREAM_MIB");
        if (s != nullptr && atoll(s) > 0) {
            const char * io = getenv("GGML_MOE_STREAM_IO_THREADS");
            const char * lg = getenv("GGML_MOE_STREAM_LOG");
            g_stream = new moe_stream((size_t) atoll(s), io ? atoi(io) : 16, lg && atoi(lg) != 0);
            if (const char * tr = getenv("GGML_MOE_STREAM_TRACE")) {
                g_stream->trace = fopen(tr, "w");
                if (g_stream->trace) {
                    atexit([] { fflush(g_stream->trace); });
                }
            }
            if (g_stream->pool == nullptr) {
                g_stream = nullptr;
            } else if (g_stream->log) {
                atexit([] { g_stream->print_stats(); });
            }
        }
    }
    return g_stream;
}

}

bool ggml_moe_stream_enabled(void) {
    return get_stream() != nullptr;
}

void ggml_moe_stream_prepare(const struct ggml_cgraph * cgraph, int node_n) {
    get_stream()->prepare(cgraph, node_n);
}

bool ggml_moe_stream_copy(const struct ggml_tensor * w, const int32_t * counts,
        void (*set)(void * ctx, size_t offset, const void * data, size_t size), void (*sync)(void * ctx), void * ctx) {
    moe_stream * st = get_stream();
    return st != nullptr && st->copy(w, counts, set, sync, ctx);
}

void ggml_moe_stream_set_expert_bytes(size_t n) {
    if (moe_stream * st = get_stream()) {
        st->expert_bytes = n;
    }
}

void ggml_moe_stream_register(const struct ggml_tensor * w) {
    if (moe_stream * st = get_stream()) {
        std::unique_lock<std::mutex> lock(st->mtx);
        st->info(w);
    }
}

bool ggml_moe_stream_is_streamed(const struct ggml_tensor * w) {
    moe_stream * st = get_stream();
    if (st == nullptr) {
        return false;
    }
    std::unique_lock<std::mutex> lock(st->mtx);
    auto it = st->tensors.find(w);
    return it != st->tensors.end() && it->second.stream;
}

int ggml_moe_stream_staging(void ** bufs, size_t * size) {
    int n = 0;
    if (moe_stream * st = get_stream()) {
        st->staging(bufs, size, &n);
    }
    return n;
}

bool ggml_moe_stream_ready(const struct ggml_tensor * w, int64_t e) {
    return g_stream == nullptr || g_stream->is_ready(w, e);
}

void ggml_moe_stream_hint(const struct ggml_tensor * const * ws, int n_ws, const struct ggml_tensor * ids) {
    if (moe_stream * st = get_stream()) {
        st->hint(ws, n_ws, ids);
    }
}

const char * ggml_moe_stream_expert(const struct ggml_tensor * w, int64_t e) {
    if (g_stream == nullptr) {
        return (const char *) w->data + e*w->nb[2];
    }
    return g_stream->expert(w, e);
}

bool ggml_moe_stream_get_cached(const struct ggml_tensor * w, int64_t e, void * dst, size_t size) {
    return g_stream != nullptr && g_stream->get_cached(w, e, dst, size);
}

#else

bool ggml_moe_stream_get_cached(const struct ggml_tensor * w, int64_t e, void * dst, size_t size) {
    (void) w; (void) e; (void) dst; (void) size;
    return false;
}

bool ggml_moe_stream_enabled(void) {
    return false;
}

bool ggml_moe_stream_ready(const struct ggml_tensor * w, int64_t e) {
    (void) w; (void) e;
    return true;
}

void ggml_moe_stream_hint(const struct ggml_tensor * const * ws, int n_ws, const struct ggml_tensor * ids) {
    (void) ws; (void) n_ws; (void) ids;
}

bool ggml_moe_stream_copy(const struct ggml_tensor * w, const int32_t * counts,
        void (*set)(void * ctx, size_t offset, const void * data, size_t size), void (*sync)(void * ctx), void * ctx) {
    (void) w; (void) counts; (void) set; (void) sync; (void) ctx;
    return false;
}

void ggml_moe_stream_set_expert_bytes(size_t n) {
    (void) n;
}

void ggml_moe_stream_register(const struct ggml_tensor * w) {
    (void) w;
}

int ggml_moe_stream_staging(void ** bufs, size_t * size) {
    (void) bufs; (void) size;
    return 0;
}

bool ggml_moe_stream_is_streamed(const struct ggml_tensor * w) {
    (void) w;
    return false;
}

void ggml_moe_stream_prepare(const struct ggml_cgraph * cgraph, int node_n) {
    (void) cgraph;
    (void) node_n;
}

const char * ggml_moe_stream_expert(const struct ggml_tensor * w, int64_t e) {
    return (const char *) w->data + e*w->nb[2];
}

#endif
