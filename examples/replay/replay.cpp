// llama-replay: teacher-forced generation benchmark and logit comparison
// the text of -f is tokenized; the first REPLAY_PROMPT tokens (default 64) are decoded as one batch, then every following token is
// decoded alone, as in generation, so that every run sees the same tokens whatever the numerics
//   REPLAY_N=n          decode at most n single tokens
//   REPLAY_SAVE=path    save the logits of every single-token step (float32, n_vocab each)
//   REPLAY_REF=path     compare with saved logits: KL divergence and top-1 agreement
//   REPLAY_REPEAT=k     run the single-token steps k times (the KV cache is rolled back), timing each pass
//   REPLAY_STEP=k       decode k teacher-forced tokens per call (all with logits, as a speculative verify batch); timing only
//   REPLAY_PLAN=path    replay a recorded session (prompt and generation segments), see below
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void batch_set(llama_batch & b, int i, llama_token t, llama_pos pos, bool logits) {
    b.token[i]     = t;
    b.pos[i]       = pos;
    b.n_seq_id[i]  = 1;
    b.seq_id[i][0] = 0;
    b.logits[i]    = logits;
    b.n_tokens     = i + 1;
}

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void log_softmax(const float * logits, int n, std::vector<double> & out) {
    out.resize(n);
    double mx = -INFINITY;
    for (int i = 0; i < n; ++i) mx = std::max(mx, (double) logits[i]);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += std::exp((double) logits[i] - mx);
    const double lse = mx + std::log(sum);
    for (int i = 0; i < n; ++i) out[i] = (double) logits[i] - lse;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    const int n_prompt = getenv("REPLAY_PROMPT") ? atoi(getenv("REPLAY_PROMPT")) : 64;
    const int n_max    = getenv("REPLAY_N")      ? atoi(getenv("REPLAY_N"))      : 1 << 30;
    const int repeat   = getenv("REPLAY_REPEAT") ? atoi(getenv("REPLAY_REPEAT")) : 1;
    const int step_k   = getenv("REPLAY_STEP")   ? std::max(1, atoi(getenv("REPLAY_STEP"))) : 1;
    const char * save  = getenv("REPLAY_SAVE");
    const char * ref   = getenv("REPLAY_REF");

    // one output per batch, as in the server: the prompt only needs the logits of its last token
    params.n_outputs_max         = step_k;
    params.n_outputs_max_per_seq = 1;

    params.n_outputs_max_per_seq = std::max(params.n_outputs_max_per_seq, step_k);
    // REPLAY_DUMP=text (diagnostic): print a checksum of every tensor whose name contains text (e.g. "l_last") in the
    // first REPLAY_DUMP_GRAPHS (default 1) graphs, to find where two builds diverge
    static const char * dump_sfx = getenv("REPLAY_DUMP");
    static int dump_graphs = getenv("REPLAY_DUMP_GRAPHS") ? atoi(getenv("REPLAY_DUMP_GRAPHS")) : 1;
    if (dump_sfx) {
        params.cb_eval = [](struct ggml_tensor * t, bool ask, void *) -> bool {
            static const ggml_tensor * first = nullptr;
            static int n_graph = 0;
            const bool want = n_graph < dump_graphs && strstr(t->name, dump_sfx) != nullptr && t->type == GGML_TYPE_F32;
            if (ask) {
                if (strcmp(t->name, "result_output") == 0) n_graph++;
                return want;
            }
            if (want) {
                std::vector<uint8_t> buf(ggml_nbytes(t));
                ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
                uint64_t h = 1469598103934665603ull;
                for (uint8_t b : buf) { h = (h ^ b) * 1099511628211ull; }
                double sum = 0; const float * f = (const float *) buf.data();
                for (size_t i = 0; i < buf.size()/4; ++i) sum += f[i];
                fprintf(stderr, "dump: %-32s %-10s [%lld,%lld,%lld] sum %.9g fnv %016llx\n", t->name, ggml_op_desc(t),
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], sum, (unsigned long long) h);
            }
            (void) first;
            return true;
        };
        params.cb_eval_user_data = nullptr;
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    auto init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to load the model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    FILE * fsave = save ? fopen(save, "wb") : nullptr;
    FILE * fref  = ref  ? fopen(ref,  "rb") : nullptr;
    std::vector<float> ref_logits(n_vocab);
    std::vector<double> lp_ref, lp_test;
    std::vector<double> klds;
    int top1_same = 0, n_cmp = 0, n_exact = 0;
    // REPLAY_SAVE / REPLAY_REF for one position's logits
    auto check = [&](const float * logits) {
        if (fsave) {
            fwrite(logits, sizeof(float), n_vocab, fsave);
        }
        if (fref && fread(ref_logits.data(), sizeof(float), n_vocab, fref) == (size_t) n_vocab) {
            log_softmax(ref_logits.data(), n_vocab, lp_ref);
            log_softmax(logits, n_vocab, lp_test);
            double kld = 0.0;
            for (int j = 0; j < n_vocab; ++j) {
                kld += std::exp(lp_ref[j]) * (lp_ref[j] - lp_test[j]);
            }
            klds.push_back(kld);
            top1_same += std::max_element(ref_logits.data(), ref_logits.data() + n_vocab) - ref_logits.data() ==
                         std::max_element(logits, logits + n_vocab) - logits;
            n_exact += memcmp(ref_logits.data(), logits, n_vocab*sizeof(float)) == 0;
            n_cmp++;
        }
    };
    auto report = [&]() {
        if (n_cmp > 0) {
            std::vector<double> sorted = klds;
            std::sort(sorted.begin(), sorted.end());
            double mean = 0.0;
            for (double k : klds) mean += k;
            mean /= klds.size();
            fprintf(stderr, "replay: KLD vs reference over %d positions: mean %.6f, median %.6f, p99 %.6f, max %.6f; top-1 same %.2f%%; bit-identical %d of %d\n",
                n_cmp, mean, sorted[sorted.size()/2], sorted[std::min(sorted.size() - 1, (size_t) (sorted.size()*0.99))], sorted.back(),
                100.0*top1_same/n_cmp, n_exact, n_cmp);
        }
    };

    // REPLAY_PLAN=path: replay a recorded session instead of -f: lines "P <n> <tokens>" are processed as prompt batches (n_batch
    // tokens per call, logits for the last), lines "D <n> <tokens>" are decoded one token at a time as in generation; the context
    // only grows (no rollback), so that the expert routing of a real conversation can be traced and timed
    if (const char * plan = getenv("REPLAY_PLAN")) {
        std::ifstream in(plan);
        std::string line;
        llama_pos pos = 0;
        double t_p = 0.0, t_d = 0.0;
        int64_t n_p = 0, n_d = 0;
        int seg = 0;
        // REPLAY_STATE=path: start from a saved sequence (a llama-server slot file), the plan continues after its last position
        if (const char * state = getenv("REPLAY_STATE")) {
            std::vector<llama_token> stoks(llama_n_ctx(ctx));
            size_t n_stoks = 0;
            const double t0 = now_s();
            if (llama_state_seq_load_file(ctx, state, 0, stoks.data(), stoks.size(), &n_stoks) == 0) {
                LOG_ERR("failed to load the state %s\n", state);
                return 1;
            }
            pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
            fprintf(stderr, "replay: state %s loaded in %.1f s, context %d\n", state, now_s() - t0, pos);
        }
        llama_batch bp = llama_batch_init(params.n_batch, 0, 1);
        llama_batch b1 = llama_batch_init(1, 0, 1);
        while (std::getline(in, line)) {
            std::istringstream ss(line);
            char kind = 0;
            int n = 0;
            ss >> kind >> n;
            std::vector<llama_token> seg_tokens(n);
            for (int i = 0; i < n; ++i) {
                ss >> seg_tokens[i];
            }
            if (n <= 0 || (kind != 'P' && kind != 'D')) {
                continue;
            }
            if (pos + n > (llama_pos) llama_n_ctx(ctx)) {
                LOG_ERR("plan exceeds the context at segment %d\n", seg);
                break;
            }
            const double t0 = now_s();
            if (kind == 'P') {
                for (int i = 0; i < n; i += params.n_batch) {
                    const int m = std::min(n - i, params.n_batch);
                    bp.n_tokens = 0;
                    for (int j = 0; j < m; ++j) {
                        batch_set(bp, j, seg_tokens[i + j], pos + i + j, i + j == n - 1);
                    }
                    if (llama_decode(ctx, bp) != 0) {
                        LOG_ERR("prompt decode failed at segment %d\n", seg);
                        return 1;
                    }
                }
                llama_synchronize(ctx);
                check(llama_get_logits_ith(ctx, -1));
            } else {
                for (int i = 0; i < n; ++i) {
                    batch_set(b1, 0, seg_tokens[i], pos + i, true);
                    if (llama_decode(ctx, b1) != 0) {
                        LOG_ERR("decode failed at segment %d\n", seg);
                        return 1;
                    }
                    check(llama_get_logits_ith(ctx, -1));
                }
            }
            const double dt = now_s() - t0;
            (kind == 'P' ? t_p : t_d) += dt;
            (kind == 'P' ? n_p : n_d) += n;
            fprintf(stderr, "replay: segment %d %c %d tokens at %d: %.2f s, %.2f t/s\n", seg, kind, n, pos, dt, n/dt);
            pos += n;
            seg++;
        }
        fprintf(stderr, "replay: plan done: prompts %lld tokens in %.1f s (%.1f t/s), decode %lld tokens in %.1f s (%.2f t/s), context %d\n",
            (long long) n_p, t_p, n_p/std::max(t_p, 1e-9), (long long) n_d, t_d, n_d/std::max(t_d, 1e-9), pos);
        if (fsave) {
            fclose(fsave);
        }
        report();
        llama_batch_free(bp);
        llama_batch_free(b1);
        llama_perf_context_print(ctx);
        return 0;
    }

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, true);
    if ((int) tokens.size() <= n_prompt + 1) {
        LOG_ERR("text too short: %zu tokens\n", tokens.size());
        return 1;
    }
    const int n_steps = std::min<int>(n_max, (int) tokens.size() - n_prompt);
    fprintf(stderr, "replay: %zu tokens, prompt %d, %d single-token steps x %d\n", tokens.size(), n_prompt, n_steps, repeat);

    // prompt
    {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; ++i) {
            batch_set(batch, i, tokens[i], i, i == n_prompt - 1);
        }
        const double t0 = now_s();
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("prompt decode failed\n");
            return 1;
        }
        llama_synchronize(ctx);
        fprintf(stderr, "replay: prompt %d tokens in %.2f s\n", n_prompt, now_s() - t0);
        llama_batch_free(batch);
    }

    llama_memory_t mem = llama_get_memory(ctx);
    if (step_k > 1) {
        // verify-batch timing: k tokens per decode call
        llama_batch bk = llama_batch_init(step_k, 0, 1);
        const int n_calls = n_steps / step_k;
        double t_total = 0.0;
        for (int c = 0; c < n_calls; ++c) {
            bk.n_tokens = 0;
            for (int i = 0; i < step_k; ++i) {
                const int pos = n_prompt + c*step_k + i;
                batch_set(bk, i, tokens[pos], pos, true);
            }
            const double t0 = now_s();
            if (llama_decode(ctx, bk) != 0) {
                LOG_ERR("decode failed at call %d\n", c);
                return 1;
            }
            (void) llama_get_logits_ith(ctx, -1);
            t_total += now_s() - t0;
            // REPLAY_SAVE / REPLAY_REF also work per position of the k-token batches (2026-10-15: checks the hybrid MoE in batches)
            for (int i = 0; i < step_k && (fsave || fref); ++i) {
                const float * logits = llama_get_logits_ith(ctx, i);
                if (fsave) {
                    fwrite(logits, sizeof(float), n_vocab, fsave);
                }
                if (fref && fread(ref_logits.data(), sizeof(float), n_vocab, fref) == (size_t) n_vocab) {
                    log_softmax(ref_logits.data(), n_vocab, lp_ref);
                    log_softmax(logits, n_vocab, lp_test);
                    double kld = 0.0;
                    for (int j = 0; j < n_vocab; ++j) {
                        kld += std::exp(lp_ref[j]) * (lp_ref[j] - lp_test[j]);
                    }
                    klds.push_back(kld);
                    top1_same += std::max_element(ref_logits.data(), ref_logits.data() + n_vocab) - ref_logits.data() ==
                                 std::max_element(logits, logits + n_vocab) - logits;
                    n_cmp++;
                }
            }
        }
        fprintf(stderr, "replay: %d calls of %d tokens in %.2f s = %.2f ms/call, %.2f t/s\n", n_calls, step_k, t_total,
            1000.0*t_total/n_calls, n_calls*step_k/t_total);
        if (fsave) {
            fclose(fsave);
        }
        if (n_cmp > 0) {
            std::vector<double> sorted = klds;
            std::sort(sorted.begin(), sorted.end());
            double mean = 0.0;
            for (double k : klds) mean += k;
            mean /= klds.size();
            fprintf(stderr, "replay: KLD vs reference over %d positions: mean %.6f, median %.6f, p99 %.6f, max %.6f; top-1 same %.2f%%\n",
                n_cmp, mean, sorted[sorted.size()/2], sorted[std::min(sorted.size() - 1, (size_t) (sorted.size()*0.99))], sorted.back(),
                100.0*top1_same/n_cmp);
        }
        llama_batch_free(bk);
        llama_perf_context_print(ctx);
        return 0;
    }
    llama_batch batch = llama_batch_init(1, 0, 1);
    for (int r = 0; r < repeat; ++r) {
        if (r > 0) {
            llama_memory_seq_rm(mem, 0, n_prompt, -1);
        }
        double t_total = 0.0, t_block = 0.0;
        for (int s = 0; s < n_steps; ++s) {
            const int pos = n_prompt + s;
            batch_set(batch, 0, tokens[pos], pos, true);
            const double t0 = now_s();
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("decode failed at step %d\n", s);
                return 1;
            }
            const float * logits = llama_get_logits_ith(ctx, -1);
            const double dt = now_s() - t0;
            t_total += dt;
            t_block += dt;
            if ((s + 1) % 100 == 0) {
                fprintf(stderr, "replay: pass %d steps %d-%d: %.2f t/s\n", r, s - 99, s, 100.0/t_block);
                t_block = 0.0;
            }
            if (r == 0 && fsave) {
                fwrite(logits, sizeof(float), n_vocab, fsave);
            }
            if (r == 0 && fref) {
                if (fread(ref_logits.data(), sizeof(float), n_vocab, fref) != (size_t) n_vocab) {
                    fclose(fref);
                    fref = nullptr;
                    continue;
                }
                log_softmax(ref_logits.data(), n_vocab, lp_ref);
                log_softmax(logits, n_vocab, lp_test);
                double kld = 0.0;
                for (int i = 0; i < n_vocab; ++i) {
                    kld += std::exp(lp_ref[i]) * (lp_ref[i] - lp_test[i]);
                }
                klds.push_back(kld);
                const int a = (int) (std::max_element(ref_logits.data(), ref_logits.data() + n_vocab) - ref_logits.data());
                const int b = (int) (std::max_element(logits, logits + n_vocab) - logits);
                top1_same += a == b;
                n_cmp++;
            }
        }
        fprintf(stderr, "replay: pass %d: %d steps in %.2f s = %.2f t/s\n", r, n_steps, t_total, n_steps/t_total);
    }
    llama_batch_free(batch);
    if (fsave) {
        fclose(fsave);
    }
    if (n_cmp > 0) {
        std::vector<double> sorted = klds;
        std::sort(sorted.begin(), sorted.end());
        double mean = 0.0;
        for (double k : klds) mean += k;
        mean /= klds.size();
        fprintf(stderr, "replay: KLD vs reference over %d steps: mean %.6f, median %.6f, p99 %.6f, max %.6f; top-1 same %.2f%%\n",
            n_cmp, mean, sorted[sorted.size()/2], sorted[std::min(sorted.size() - 1, (size_t) (sorted.size()*0.99))], sorted.back(),
            100.0*top1_same/n_cmp);
    }
    llama_perf_context_print(ctx);
    return 0;
}
