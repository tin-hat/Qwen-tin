// llama-kldapi: compare the local model with reference answers that carry top-k logprobs (e.g. an API serving the BF16 model)
// Each line of KLD_IN is a JSON object {"id", "messages": [...], "response": <OpenAI chat completion with logprobs>}. The messages
// are rendered with the model's chat template (non-thinking), tokenized, and checked against the reference's prompt token count;
// the answer tokens are mapped to the local vocabulary by their bytes. The model is then teacher-forced on the answer: the prompt
// is decoded as batches, every answer token alone, as in generation. Per answer position it computes
//   kl_top: KL(reference || local) over the reference's top-k tokens plus one bucket for the rest (a lower bound of the full KL)
//   agree:  whether the local most likely token is the reference's most likely one
//   nll:    -log p of the sampled token, for the reference and the local model
// and writes one JSON line per position to KLD_OUT (with the local top KLD_TOPN tokens, for comparisons between local runs).
//   KLD_IN=path, KLD_OUT=path, KLD_ONLY=id1,id2 (subset), KLD_TOPN=n (default 16)
#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::ordered_json;

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

// the bytes of a logprob entry: its "bytes" array if present, else its token string
static std::string entry_bytes(const json & e) {
    if (e.contains("bytes") && e["bytes"].is_array() && !e["bytes"].empty()) {
        std::string s;
        for (const auto & b : e["bytes"]) s.push_back((char) b.get<int>());
        return s;
    }
    return e.value("token", std::string());
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    const char * in_path  = getenv("KLD_IN");
    const char * out_path = getenv("KLD_OUT");
    const int    top_n    = getenv("KLD_TOPN") ? atoi(getenv("KLD_TOPN")) : 16;
    std::set<std::string> only;
    if (const char * o = getenv("KLD_ONLY")) {
        std::stringstream ss(o);
        for (std::string s; std::getline(ss, s, ',');) only.insert(s);
    }
    if (!in_path || !out_path) {
        LOG_ERR("set KLD_IN and KLD_OUT\n");
        return 1;
    }

    // one output per batch, as in the server
    params.n_outputs_max         = 1;
    params.n_outputs_max_per_seq = 1;

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
    const int n_batch = (int) llama_n_batch(ctx);
    const int n_ctx   = (int) llama_n_ctx(ctx);

    // bytes -> token, for the reference's tokens
    std::unordered_map<std::string, llama_token> by_bytes;
    int n_dup = 0;
    for (llama_token t = 0; t < n_vocab; ++t) {
        const std::string piece = common_token_to_piece(ctx, t, true);
        if (piece.empty()) continue;
        if (!by_bytes.emplace(piece, t).second) n_dup++;
    }
    fprintf(stderr, "kldapi: vocab %d tokens, %zu distinct pieces (%d duplicates kept as the first id)\n", n_vocab, by_bytes.size(), n_dup);

    auto tmpls = common_chat_templates_init(model, "");

    std::ifstream fin(in_path);
    FILE * fout = fopen(out_path, "w");
    std::vector<double> lp;
    double sum_kl = 0, sum_nll_l = 0, sum_nll_r = 0;
    long n_pos = 0, n_all = 0, n_agree = 0, n_unmapped = 0, n_items = 0, n_skipped = 0;
    llama_memory_t mem = llama_get_memory(ctx);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    const double t_start = now_s();

    for (std::string line; std::getline(fin, line);) {
        if (line.empty()) continue;
        const json item = json::parse(line);
        const std::string id = item["id"];
        if (!only.empty() && !only.count(id)) continue;

        // the prompt, as the reference rendered it
        common_chat_templates_inputs inputs;
        for (const auto & m : item["messages"]) {
            common_chat_msg msg;
            msg.role    = m["role"];
            msg.content = m["content"];
            inputs.messages.push_back(msg);
        }
        inputs.add_generation_prompt = true;
        inputs.use_jinja             = true;
        inputs.enable_thinking       = false;
        inputs.chat_template_kwargs["enable_thinking"] = "false";
        const std::string prompt = common_chat_templates_apply(tmpls.get(), inputs).prompt;
        // the template's own markup is parsed for special tokens, the message texts are not (an API tokenizes them as plain
        // text, and documents can quote markup like <|im_start|>)
        std::vector<llama_token> ptoks;
        {
            size_t at = 0;
            for (const auto & m : inputs.messages) {
                // templates may trim the message
                std::string text = m.content;
                size_t k = text.empty() ? std::string::npos : prompt.find(text, at);
                if (k == std::string::npos) {
                    const size_t b = text.find_first_not_of(" \t\r\n"), e = text.find_last_not_of(" \t\r\n");
                    text = b == std::string::npos ? std::string() : text.substr(b, e - b + 1);
                    k = text.empty() ? std::string::npos : prompt.find(text, at);
                }
                if (k == std::string::npos) {
                    fprintf(stderr, "kldapi: %s: a message was not found in the rendered prompt, tokenized with the markup\n", id.c_str());
                    continue;
                }
                for (llama_token t : common_tokenize(ctx, prompt.substr(at, k - at), false, true)) ptoks.push_back(t);
                for (llama_token t : common_tokenize(ctx, text, false, false)) ptoks.push_back(t);
                at = k + text.size();
            }
            for (llama_token t : common_tokenize(ctx, prompt.substr(at), false, true)) ptoks.push_back(t);
        }
        const auto & resp = item["response"];
        const int ref_prompt = resp["usage"]["prompt_tokens"];
        if ((int) ptoks.size() != ref_prompt) {
            fprintf(stderr, "kldapi: %s: SKIPPED, prompt is %zu tokens here vs %d for the reference\n", id.c_str(), ptoks.size(), ref_prompt);
            n_skipped++;
            continue;
        }

        // the answer tokens and the reference's top-k
        const auto & entries = resp["choices"][0]["logprobs"]["content"];
        struct pos_ref { llama_token tok; double lp; std::vector<std::pair<llama_token, double>> top; bool ok; };
        std::vector<pos_ref> refs;
        bool broken = false;
        for (const auto & e : entries) {
            pos_ref r;
            auto it = by_bytes.find(entry_bytes(e));
            if (it == by_bytes.end()) { broken = true; break; }
            r.tok = it->second;
            r.lp  = e["logprob"];
            r.ok  = true;
            for (const auto & t : e["top_logprobs"]) {
                auto jt = by_bytes.find(entry_bytes(t));
                if (jt == by_bytes.end()) { r.ok = false; continue; }
                r.top.push_back({ jt->second, t["logprob"].get<double>() });
            }
            refs.push_back(r);
        }
        if (broken || refs.empty()) {
            fprintf(stderr, "kldapi: %s: SKIPPED, answer token %zu has no local token\n", id.c_str(), refs.size());
            n_skipped++;
            continue;
        }
        if ((int) (ptoks.size() + refs.size()) > n_ctx) {
            fprintf(stderr, "kldapi: %s: SKIPPED, %zu tokens exceed the context\n", id.c_str(), ptoks.size() + refs.size());
            n_skipped++;
            continue;
        }

        // prompt in batches, the logits of its last token predict the first answer token
        llama_memory_clear(mem, true);
        const double t0 = now_s();
        for (size_t i0 = 0; i0 < ptoks.size(); i0 += n_batch) {
            const int n = (int) std::min<size_t>(n_batch, ptoks.size() - i0);
            for (int i = 0; i < n; ++i) {
                batch_set(batch, i, ptoks[i0 + i], (llama_pos) (i0 + i), i0 + i == ptoks.size() - 1);
            }
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("prompt decode failed\n");
                return 1;
            }
        }
        const double t1 = now_s();

        double it_kl = 0, it_nll_l = 0, it_nll_r = 0;
        long it_n = 0, it_agree = 0;
        for (size_t j = 0; j < refs.size(); ++j) {
            if (j > 0) {
                batch_set(batch, 0, refs[j - 1].tok, (llama_pos) (ptoks.size() + j - 1), true);
                if (llama_decode(ctx, batch) != 0) {
                    LOG_ERR("decode failed\n");
                    return 1;
                }
            }
            const float * logits = llama_get_logits_ith(ctx, -1);
            log_softmax(logits, n_vocab, lp);
            const pos_ref & r = refs[j];
            const int amax = (int) (std::max_element(logits, logits + n_vocab) - logits);

            // KL over the reference's top-k plus a bucket for the rest
            double kl = 0, p_sum = 0, q_sum = 0;
            for (const auto & [t, l] : r.top) {
                const double p = std::exp(l);
                kl    += p * (l - lp[t]);
                p_sum += p;
                q_sum += std::exp(lp[t]);
            }
            const double p_rest = std::max(0.0, 1.0 - p_sum);
            const double q_rest = std::max(1e-30, 1.0 - q_sum);
            if (p_rest > 1e-9) {
                kl += p_rest * (std::log(p_rest) - std::log(q_rest));
            }
            const bool agree = !r.top.empty() && amax == r.top[0].first;
            if (r.ok) {
                it_kl += kl;
                it_n++;
                it_agree += agree;
            } else {
                n_unmapped++;
            }
            it_nll_l += -lp[r.tok];
            it_nll_r += -r.lp;

            // local top-n for comparisons between local runs
            std::vector<int> idx(n_vocab);
            for (int i = 0; i < n_vocab; ++i) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + top_n, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
            json o;
            o["id"] = id; o["j"] = j; o["tok"] = r.tok; o["ok"] = r.ok;
            o["kl_top"] = kl; o["agree"] = agree; o["nll_ref"] = -r.lp; o["nll_loc"] = -lp[r.tok];
            json rt = json::array();
            for (const auto & [t, l] : r.top) rt.push_back({ t, l });
            o["ref_top"] = rt;
            json lt = json::array();
            for (int i = 0; i < top_n; ++i) lt.push_back({ idx[i], lp[idx[i]] });
            o["loc_top"] = lt;
            fprintf(fout, "%s\n", o.dump().c_str());
        }
        fflush(fout);
        const double t2 = now_s();
        sum_kl += it_kl; sum_nll_l += it_nll_l; sum_nll_r += it_nll_r;
        n_pos += it_n; n_all += (long) refs.size(); n_agree += it_agree; n_items++;
        fprintf(stderr, "kldapi: %-10s prompt %5zu (%.1f s) answer %3zu (%.1f t/s) | kl_top %.4f | agree %.1f%% | nll ref %.3f loc %.3f\n",
            id.c_str(), ptoks.size(), t1 - t0, refs.size(), (refs.size() - 1)/std::max(1e-9, t2 - t1),
            it_n ? it_kl/it_n : 0.0, it_n ? 100.0*it_agree/it_n : 0.0, it_nll_r/refs.size(), it_nll_l/refs.size());
    }
    llama_batch_free(batch);
    fclose(fout);
    fprintf(stderr, "kldapi: TOTAL %ld items (%ld skipped), %ld positions (%ld with unmapped alternatives): kl_top %.5f | agree %.2f%% | "
        "nll ref %.4f loc %.4f | %.0f s\n", n_items, n_skipped, n_pos, n_unmapped, n_pos ? sum_kl/n_pos : 0.0,
        n_pos ? 100.0*n_agree/n_pos : 0.0, n_all ? sum_nll_r/n_all : 0.0, n_all ? sum_nll_l/n_all : 0.0, now_s() - t_start);
    llama_perf_context_print(ctx);
    return 0;
}
