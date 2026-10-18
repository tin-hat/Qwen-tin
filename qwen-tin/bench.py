#!/usr/bin/env python3
# Qwen-tin benchmark: prompt and generation speed of a running qwen-tin-server, on public text (the Markdown files of this repo).
#   short    a 30-token request, 1024 tokens generated
#   8k, 32k, 100k  a document of that many tokens + a question, 512 tokens generated
#   session  an agent-like chat: 12 turns, each adds ~2.5k tokens of a new document and generates 400 tokens (context ~35k)
# Speeds are the server's own timings. Generation is forced to its full length (ignore_eos), thinking is off.
# Usage: qwen-tin/bench.py [--url http://127.0.0.1:8080] [--only short,8k,32k,session,100k]
import argparse
import glob
import json
import os
import time
import urllib.request

here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.dirname(here)


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def ssd_temp():
    # the hottest NVMe sensor, if the kernel exposes one
    temps = []
    for d in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if open(os.path.join(d, "name")).read().strip() == "nvme":
                temps.append(int(open(os.path.join(d, "temp1_input")).read()) // 1000)
        except OSError:
            pass
    return f"{max(temps)} C" if temps else "n/a"


def corpus():
    files = sorted(f for f in glob.glob(os.path.join(repo, "**", "*.md"), recursive=True)
                   if "/.git/" not in f and os.path.relpath(f, repo) != "README.md")
    return [(os.path.relpath(f, repo), open(f, encoding="utf-8", errors="replace").read()) for f in files]


def chat(url, messages, n_gen):
    t0 = time.time()
    r = post(url, "/v1/chat/completions", {
        "messages": messages, "max_tokens": n_gen, "ignore_eos": True, "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    t = r["timings"]
    return r["choices"][0]["message"]["content"], t, time.time() - t0


def line(name, t, wall):
    return (f"{name:<10} prompt {t['prompt_n']:>6} tok {t['prompt_per_second']:>7.1f} t/s | "
            f"gen {t['predicted_n']:>5} tok {t['predicted_per_second']:>6.2f} t/s | wall {wall:6.1f} s | SSD {ssd_temp()}")


def doc_of(url, docs, n_tok, start=0):
    # whole files from docs[start:] until n_tok tokens; returns the text and the index of the next file
    text, n, i = "", 0, start
    while n < n_tok and i < len(docs):
        part = f"\n\n===== {docs[i][0]} =====\n{docs[i][1]}"
        k = len(post(url, "/tokenize", {"content": part})["tokens"])
        if n + k > n_tok:
            toks = post(url, "/tokenize", {"content": part})["tokens"][: n_tok - n]
            part = post(url, "/detokenize", {"tokens": toks})["content"]
            k = len(toks)
        text, n, i = text + part, n + k, i + 1
    return text, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--only", default="short,8k,32k,session")
    a = ap.parse_args()
    only = a.only.split(",")
    docs = corpus()
    print(f"server {a.url}, corpus {len(docs)} files, SSD {ssd_temp()}")
    if "short" in only:
        _, t, w = chat(a.url, [{"role": "user", "content": "Write a long story about a lighthouse keeper who finds a message in a bottle."}], 1024)
        print(line("short", t, w), flush=True)
    nxt = 0
    for name, n_tok in (("8k", 8192), ("32k", 32768), ("100k", 100000)):
        if name in only:
            text, nxt = doc_of(a.url, docs, n_tok, nxt)
            _, t, w = chat(a.url, [{"role": "user", "content": text + "\n\nSummarize the documents above, file by file."}], 512)
            print(line(name, t, w), flush=True)
    if "session" in only:
        msgs = [{"role": "system", "content": "You are a helpful assistant that reviews documentation."}]
        gen_n = gen_s = pp_n = pp_s = 0.0
        t_start = time.time()
        for turn in range(12):
            text, nxt = doc_of(a.url, docs, 2500, nxt)
            msgs.append({"role": "user", "content": text + "\n\nWhat is new in this part, and what is unclear?"})
            answer, t, w = chat(a.url, msgs, 400)
            msgs.append({"role": "assistant", "content": answer})
            print(line(f"turn {turn + 1}", t, w), flush=True)
            gen_n += t["predicted_n"]; gen_s += t["predicted_ms"] / 1000
            pp_n += t["prompt_n"]; pp_s += t["prompt_ms"] / 1000
        print(f"session    prompt {pp_n:.0f} tok {pp_n / pp_s:.1f} t/s | gen {gen_n:.0f} tok {gen_n / gen_s:.2f} t/s | "
              f"wall {time.time() - t_start:.0f} s | SSD {ssd_temp()}")


if __name__ == "__main__":
    main()
