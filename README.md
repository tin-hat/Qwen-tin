# Run Qwen3.8-Flash-Next on 8 GB VRAM and 32 GB RAM at 20 t/s

This is Qwen3.8-Flash-Next, specifically [ISTA-DASLab's GSQ-RCO IQ3_S quant](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (which according to authors "matches or exceeds the base model on every task"), adapted for 8 GB VRAM and 32 GB RAM.

It runs at ~20 tokens/second and has 150k context window - pretty usable.

On long sessions it may slow down toward ~14 t/s.

| | Qwen-tin |
|---|---|
| **Model** | Qwen3.8-Flash-Next: 125B MoE with 6B active per token, plus a 51B n-gram table |
| **Quant** | [ISTA-DASLab GSQ-RCO IQ3_S](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), 3.50 bits per weight, 83.6 GB |
| **Changed in the quant** | Gated Residual matrices BF16 -> Q8_0; vision encoder BF16 -> Q5_0/F16/Q8_0 mix |
| **KV cache** | K q5_0, V q4_0 |
| **Context** | 150k (153,600 tokens) |
| **Quality** | KL vs BF16: Qwen-tin 0.0760, original ISTA-DASLab GSQ-RCO IQ3_S 0.0756 |
| **Generation** | 18.2-23.9 tok/s (16.5 with a hot SSD) |
| **Reading** | 461-489 tok/s for 8k-100k documents, 199 tok/s for 2.5k-token chat turns |
| **Hardware** | RTX 5060 Laptop GPU (8 GB), 32 GB DDR4, PCIe 4.0 NVMe SSD |

## The quant: ISTA-DASLab GSQ-RCO IQ3_S

Qwen-tin runs **[ISTA-DASLab's GSQ-RCO IQ3_S GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)**. GSQ learns the grid assignments and scales of each tensor, and RCO searches for the quant type of each tensor under a size budget. IQ3_S averages 3.50 bits per weight: 54.8 GB of transformer weights in shard 1 and the n-gram table at IQ4_NL (28.8 GB) in shard 2.

### ISTA's benchmarks for IQ3_S

From ISTA's model card:

| | bpw | size | AIME25 | GPQA-Diamond | LiveCodeBench v6 | task average |
|---|---|---|---|---|---|---|
| BF16 | 16.00 | 354 GB | 100.00 | 91.92 | 87.43 | 93.12 |
| **GSQ-RCO IQ3_S** | **3.50** | **83.6 GB** | **100.00** | **92.93** | **86.86** | **93.26** |

ISTA recommends IQ3_S: it matches BF16 on AIME25, beats it on GPQA-Diamond and is within 0.57 points on LiveCodeBench v6, at under a quarter of the size. ISTA reads the scores above BF16 as benchmark variance, so this is parity, not a better model.

## Architecture: how it fits in 8 GB VRAM and 32 GB RAM

ISTA's card plans for all of shard 1 (54.8 GB) to be resident in VRAM. Qwen-tin keeps only the non-expert weights there. The 46.8 GiB of routed experts stay on the SSD with a cache in RAM, because each token uses only 10 of the 512 experts in each layer: about 0.9 GiB per token, 96% of which is already in RAM or VRAM.

| | VRAM (8 GB card) | RAM (32 GB) | SSD |
|---|---|---|---|
| **routed experts** (46.8 GiB) | 2-3.3 GiB: the most used ones, in banks that take whatever VRAM is left | 24 GiB: a cache of the most recently used ones | all 46.8 GiB; misses are read with `O_DIRECT`, ~20 MB per generated token |
| **other weights** (3.6 GiB) | 3.3 GiB: attention and indexer, DeltaNet, Gated Residual, shared experts, routers, output head | 0.3 GiB: the token embedding | - |
| **KV cache** (12 sparse-attention layers, K q5_0 / V q4_0) | 940 MiB at 100k, ~1.4 GiB at 150k; backed only for the cells in use | - | - |
| **n-gram table** (26.8 GiB) | - | ~1.2 GiB: the rows in use, as page cache | all 26.8 GiB, memory-mapped; one row per token |
| **vision encoder** (0.48 GiB) | - | only while an image is encoded; it runs on the CPU | 0.48 GiB, memory-mapped |
| **the rest** | ~0.6 GiB: DeltaNet state (0.1 GiB), compute buffers (~40 MiB in generation), CUDA runtime | ~2 GiB: SSD staging buffers, CPU compute buffers, context checkpoints | - |
| **total** | **~7.25 GiB: all of the ~7.5 GiB the driver makes available, except ~0.3 GiB kept free** | **~27.5 GiB of the 30.3 GiB Linux sees; at least 2 GiB stays available to other programs** | **77.8 GiB (83.5 GB) of model files** |

Prompt batches of 8192 tokens borrow most of the banks' VRAM for their compute buffer, and the experts are copied back afterwards.

### What changes in ISTA's files

| | ISTA's file / llama.cpp default | **Qwen-tin** | why |
|---|---|---|---|
| **Gated Residual** matrices (194 `hc_*` tensors) | BF16 | **Q8_0** | 568 MiB more VRAM for experts; shard 1 goes from 54.8 to 54.2 GB |
| **Vision encoder** (mmproj) | BF16, 0.91 GB | **Q5_0 / F16 / Q8_0 mix, 0.51 GB** | 0.4 GB less RAM while an image is encoded |
| **KV cache** | F16 | **K q5_0, V q4_0** | 2.4 GiB less VRAM than F16 at 150k |
| **Context** | 262,144 (native) | **153,600 (150k)** | the KV cache and the expert banks share the 8 GB; `CTX` changes it |

**Nothing else changes.** All 512 experts are kept and all 10 run for every token: no pruning, no expert skipping, no lower-bit swaps. The rest of shard 1 is copied byte for byte, and the n-gram table (shard 2) is ISTA's file as published. ISTA's GGUF has no MTP head, so there is no multi-token prediction. `setup.sh` makes both converted files on your machine and checks them against the checksums of the files measured here.

### What the engine does

- **Expert streaming** (`ggml/src/ggml-cpu/moe-stream.cpp`). Missing experts are read into the RAM cache with `O_DIRECT` by 16 threads on the E-cores. While a layer computes, the next layer's router is applied to its input, and the experts it predicts are read ahead (the top guess is used ~95% of the time). A memory governor shrinks the cache to keep 2 GiB of MemAvailable on the machine (and under a cgroup memory limit, if there is one).
- **Hybrid MoE in generation** (`src/llama-moe-vram.cpp`). For batches under 256 tokens, the GPU computes the experts in its banks while the CPU computes the rest, and the two results are added. Between steps, the most used experts are copied into the banks from the RAM cache. The GPU covers about half of the expert work at short context and a third at 100k.
- **Prompts on the GPU** (`ggml/src/ggml-cuda/mmid-stream.cu`). Prompt batches of 8192 tokens (4096 past 128k of context) compute all experts on the GPU. Each layer's experts stream from the RAM cache and the SSD through a 144 MiB ring, about 45 GiB per batch over PCIe.
- **VRAM manager.** The KV cache and the compute buffers are backed (CUDA VMM) only for what the next step uses. Everything else holds experts.
- **Gathered sparse attention** (`src/models/qwen4exp.cpp`). In generation, each sparse-attention layer reads only the ~2k cells its indexer selects instead of masking the whole cache. At 109k context that took generation from 9.5 to 15.9 tok/s.
- **Smaller compute buffers.** Chunked attention and n-gram blocks, in-place GLU, and per-chunk expert sums let 8192-token batches fit next to everything else.

## Quality: Qwen-tin vs original IQ3_S vs BF16

To check that the changes and the streaming engine cost nothing, I compared token probabilities against the BF16 model itself, served by Alibaba Cloud's API: 44 prompts (code, math, reasoning, knowledge, writing, other languages, JSON, five long documents of 3k-30k tokens) and 14,828 answer tokens. The local model is teacher-forced on the API's answers. At each position it gets the KL divergence over the API's top-5 tokens (plus one bucket for the rest), whether its top token is the same, and the extra negative log-likelihood of the token the API chose.

| | KL vs BF16 | same top-1 token as BF16 | NLL excess |
|---|---|---|---|
| BF16 (the reference) | 0 | 100% | 0 |
| BF16 against itself: the same temperature-0 request sent to the API twice | 0.0126 | 96.9% | |
| Original ISTA-DASLab GSQ-RCO IQ3_S (unmodified file: BF16 Gated Residual, F16 KV cache, all experts on the CPU) | 0.0756 | 91.88% | +0.078 |
| **Qwen-tin** (Q8_0 Gated Residual, q5_0/q4_0 KV cache, streamed hybrid experts) | **0.0760** | **91.91%** | **+0.079** |

**Qwen-tin's changes cost nothing measurable.** The last two rows differ by less than the run-to-run noise of the local side (0.0014 KL). All of the gap to BF16 comes from the 3.5-bit quant itself, and ISTA's benchmarks above show that this gap doesn't cost task accuracy.

Every engine change was also checked step by step with `llama-replay`, which teacher-forces a fixed token sequence and compares the logits. Results were bit-identical or at the floating-point reordering level of splitting a prompt into two batches (mean KL ~0.005). With BF16 Gated Residual matrices as the reference, the Q8_0 requant measured a perplexity ratio of 0.997 +- 0.006.

## Speed: reading and generation

Hardware: Intel i9-13900H, RTX 5060 Laptop GPU (8 GB), 2x16 GB DDR4-3200, Kingston OM8PGP41024Q 1 TB (PCIe 4.0 x4), Ubuntu 26.04, CUDA 13.1. Measured with [`qwen-tin/bench.py`](qwen-tin/bench.py) from a fresh clone, on public text (this repo's Markdown files), thinking off, generation forced to full length. Reading is prompt processing.

| scenario | context | reading | prompt read in | generation | SSD |
|---|---|---|---|---|---|
| short chat: 29-token prompt, 1,024 tokens generated | 1k | - | - | **23.9 tok/s** | 33 -> 46 C |
| 8k-token document + question, 512 generated | 8.7k | **464 tok/s** | 18 s | **20.6 tok/s** | -> 52 C |
| 32k-token document + question, 512 generated | 33k | **489 tok/s** | 67 s | **19.1 tok/s** | -> 58 C |
| 100k-token document + question, 512 generated (separate run) | 100k | **461 tok/s** | 3.6 min | **18.2 tok/s** | 42 -> 61 C |
| 12-turn chat: each turn adds a 2.5k-token document and generates 400 | up to 35k | **199 tok/s** | 13 s per turn | **19.0 tok/s** (16.5-20.3 per turn) | -> 70 C |

The rows other than the 100k one ran in one session, one after the other. Reading speed depends on batch size more than on context. Each 8192-token batch has to stream nearly all 46.8 GiB of experts to the GPU once, from the RAM cache and the SSD over PCIe 4.0 x4. A 2.5k-token chat turn pays almost the same transfer for a third of the tokens.

A generated token takes about 52 ms at 8-11k context: ~20 ms for the dense layers on the GPU, ~15 ms for the CPU's share of the experts, ~11 ms waiting for the SSD (30+ ms once it is hot) and ~5 ms of overhead.

**Heat is the catch.** Generation reads about 20 MB per token from the SSD, and this SSD slows down above ~65-70 C. In the chat above that took generation from ~20 to 16.5-17 tok/s. An earlier build averaged 14 tok/s over a 30-minute agent session that read 2.4 TB from the SSD. A heatsink or airflow over the M.2 slot helps more than any setting. The reads don't wear the SSD; wear comes from writes. After more than 200 TB of reads, this one's SMART data shows 0% wear and no media errors.

**With 16 GB of RAM** (emulated with a cgroup limit that leaves what a 16 GB machine would), the RAM cache settles at 6.3 GiB. The hit rate falls from 96% to 71%, and the SSD reads four times as much per token. Generation ran at ~12 tok/s for the first two minutes and ~5 tok/s once the SSD was hot. Reading stayed at 250-475 tok/s.

## Try it

You need Linux, an NVIDIA GPU with 8 GB or more, 32 GB of RAM (16 GB works, slowly), an NVMe SSD with 84 GB free (140 GB during setup), the CUDA toolkit, cmake, a C++ compiler, curl, and python3 with numpy and PyYAML.

```sh
git clone https://github.com/tin-hat/Qwen-tin
cd Qwen-tin
qwen-tin/setup.sh          # builds, downloads 84 GB from Hugging Face (checksums verified), converts 2 files
qwen-tin/qwen-tin-server   # OpenAI-compatible API and web UI on http://127.0.0.1:8080
qwen-tin/bench.py          # optional: the benchmark above
```

It is the full llama-server: OpenAI-compatible API, tool calls, images and the web UI. `setup.sh [dir]` puts the model in `models/Qwen3.8-Flash-Next`, or in `dir` (then set `QWEN_TIN_MODELS=dir` for the server). A second run resumes downloads and skips finished steps. `NO_HCQ8=1` uses ISTA's shard 1 as is, which needs no extra 54 GB during setup but leaves 0.6 GB less VRAM for experts.

`qwen-tin-server` sizes itself from the machine. These are all optional:

| variable | default | what it sets |
|---|---|---|
| `CACHE_MIB` | RAM - 6.25 GiB | RAM expert cache at most; the governor shrinks it when the machine needs memory |
| `VRAM_EXPERTS_MIB` | VRAM - 4 GiB | VRAM expert banks at most; they get what the KV cache and compute buffers leave |
| `CTX` | 153600 | context size in tokens |
| `THREADS`, `CPU_MASK`, `IO_CPUS` | from the CPU | compute threads (busy-waiting, pinned) and the cores of the SSD readers. Hybrid Intel: P-core threads + half the E-cores compute, the E-cores read |
| `HOST`, `PORT` | 127.0.0.1, 8080 | |

Anything after `qwen-tin-server` goes to llama-server.

## Limits

- Linux and NVIDIA only (`O_DIRECT`, CUDA VMM). Tested on the one machine above; other GPUs should work but haven't been tried.
- One request at a time (`--parallel 1`).
- Built for this model's architecture (`qwen4exp`). Expert streaming may work for other MoE GGUFs, but that is untested.
- None of this is in upstream llama.cpp, and the knobs are environment variables. The base is llama.cpp [`46baf1f1`](https://github.com/ggml-org/llama.cpp/commit/46baf1f1fec5a06d1e52122a9207ca978b720f95) (2026-10-07), and the changes are one commit on top of it.

If you have more memory, look at [Strata](https://github.com/Niko1221/Strata), a purpose-built engine for this model with a doorbell-driven CPU/GPU expert split and multi-token prediction. It is much faster when the experts fit in RAM, and its community has published results for dozens of machines. Qwen-tin is for the machines below that: an 8 GB GPU, 32 GB of RAM, and an SSD doing the rest.

## Tools

- `llama-replay` (`examples/replay`): teacher-forced generation benchmark. It saves and compares the logits of every step (KL, top-1, bit-identical count) and can replay a recorded session.
- `llama-kldapi` (`examples/kldapi`): KL against reference answers with top-k logprobs, such as an API serving the BF16 model. This is how the quality table was measured.

## Credits and licenses

- Code: MIT, like [llama.cpp](https://github.com/ggml-org/llama.cpp) ([LICENSE](LICENSE)), which this is a fork of; its README is in [README-llama.cpp.md](README-llama.cpp.md).
- Model: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team, under the Qwen Community License 1.0. Quantization: [ISTA-DASLab's GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) ([GSQ](https://arxiv.org/abs/2604.18556), [RCO](https://arxiv.org/abs/2605.00649)), whose weights inherit Qwen's license. This repo contains no weights. `setup.sh` downloads them from ISTA's repository and converts two files on your machine. Using the model means accepting Qwen's license.
- The vision encoder mix follows HermiHg's Q5_K-MIX recipe.
- Written with the help of Claude (Anthropic).

Contact: superairesearcher@proton.me
