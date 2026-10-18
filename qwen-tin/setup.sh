#!/usr/bin/env bash
# Qwen-tin setup: builds llama-server with CUDA, downloads ISTA-DASLab's GSQ-RCO IQ3_S GGUF of Qwen3.8-Flash-Next (pinned
# revision, checksums verified) and makes the two files Qwen-tin runs from it:
#   - shard 1 with its 194 BF16 hyper-connection matrices requantized to Q8_0 (568 MiB less VRAM, everything else copied)
#   - the vision encoder as a Q5_0/F16/Q8_0 mix (0.5 GB instead of 0.9 GB)
# Usage: qwen-tin/setup.sh [models dir]   (default: $QWEN_TIN_MODELS, else models/Qwen3.8-Flash-Next in this repo)
# Env:   CUDA_ARCH=120 (default: native = the GPU in this machine), NO_BUILD=1, NO_VISION=1,
#        NO_HCQ8=1 (run shard 1 as downloaded: no 54 GB conversion, 0.6 GB less VRAM for experts),
#        KEEP_ORIGINAL=1 (keep the downloaded shard 1 and BF16 vision encoder after converting them)
# Disk: 84 GB at the end; 140 GB while shard 1 is converted (84 GB with NO_HCQ8=1). Re-running resumes and skips finished steps.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(dirname "$here")
M=${1:-${QWEN_TIN_MODELS:-$repo/models/Qwen3.8-Flash-Next}}
REV=ed59f92082b1e93c0e96d60a8b11aab089b52f09
HF=https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/$REV
S1=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
S2=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
MMB=mmproj-Qwen3.8-Flash-Next-BF16.gguf
MMQ=mmproj-Qwen3.8-Flash-Next-Q5mix.gguf

die() { echo "ERROR: $*" >&2; exit 1; }
want() { awk -v f="$1" '$2 == f { print $1 }' "$here/SHA256SUMS"; }
# ok <file> <name in SHA256SUMS>: whether the file has the expected checksum
ok() { echo "  checking $2 ..."; [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$(want "$2")" ]; }

fetch() {  # fetch <path in the ISTA repo>: download into $M unless it is there with the right checksum
    local f="$M/$1"
    if [ -f "$f" ]; then
        ok "$f" "$1" && return 0
        echo "  $1: checksum mismatch, downloading again"; rm -f "$f"
    fi
    mkdir -p "$(dirname "$f")"
    echo "downloading $1"
    curl -L --fail --retry 10 --retry-delay 5 -C - -o "$f.part" "$HF/$1" || die "download of $1 failed (run setup again to resume)"
    ok "$f.part" "$1" || { rm -f "$f.part"; die "$1: checksum mismatch after download"; }
    mv "$f.part" "$f"
}

# convert <script> <src> <dst>: the result is checked against SHA256SUMS; a mismatch (other numpy version) is a warning
convert() {
    local name=${3#"$M/"}
    echo "converting ${2#"$M/"} -> $name"
    python3 "$here/$1" "$2" "$3.part"
    mv "$3.part" "$3"
    ok "$3" "$name" || echo "  WARNING: $name differs from the published checksum (the file should still work; please report it)"
}

command -v curl >/dev/null || die "curl is missing"
command -v sha256sum >/dev/null || die "sha256sum is missing"
python3 -c 'import numpy, yaml' 2>/dev/null || die "python3 with numpy and PyYAML is needed for the conversion (pip install numpy pyyaml, or apt install python3-numpy python3-yaml)"

if [ -z "${NO_BUILD:-}" ]; then
    command -v cmake >/dev/null || die "cmake is missing"
    nvcc=$(command -v nvcc || ls -d /usr/local/cuda/bin/nvcc 2>/dev/null || true)
    [ -n "$nvcc" ] || die "nvcc (the CUDA toolkit) is missing; add its bin directory to PATH"
    echo "building llama-server ($nvcc, CUDA arch ${CUDA_ARCH:-native})"
    cmake -S "$repo" -B "$repo/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-native}" -DCMAKE_CUDA_COMPILER="$nvcc" -DLLAMA_BUILD_TESTS=OFF >/dev/null
    cmake --build "$repo/build" -j"$(nproc)" --target llama-server
fi

mkdir -p "$M"
avail_gb=$(( $(df -Pk "$M" | awk 'NR == 2 { print $4 }') / 1024 / 1024 ))
echo "models: $M ($avail_gb GB free)"

if [ -n "${NO_HCQ8:-}" ]; then
    fetch "IQ3_S/$S1"
    fetch "IQ3_S/$S2"
else
    H="$M/IQ3_S-hcq8"
    mkdir -p "$H"
    if [ -f "$H/$S2" ] && ok "$H/$S2" "IQ3_S-hcq8/$S2"; then :; else
        fetch "IQ3_S/$S2"
        ln -f "$M/IQ3_S/$S2" "$H/$S2" 2>/dev/null || cp "$M/IQ3_S/$S2" "$H/$S2"
    fi
    if [ ! -f "$H/$S1" ]; then
        fetch "IQ3_S/$S1"
        avail_gb=$(( $(df -Pk "$M" | awk 'NR == 2 { print $4 }') / 1024 / 1024 ))
        [ "$avail_gb" -ge 55 ] || die "the conversion of shard 1 needs 55 GB free, $avail_gb GB left (or run with NO_HCQ8=1)"
        convert requant_hc.py "$M/IQ3_S/$S1" "$H/$S1"
    fi
    if [ -z "${KEEP_ORIGINAL:-}" ]; then
        rm -f "$M/IQ3_S/$S1" "$M/IQ3_S/$S2"
        rmdir "$M/IQ3_S" 2>/dev/null || true
    fi
fi

if [ -z "${NO_VISION:-}" ] && [ ! -f "$M/$MMQ" ]; then
    fetch "$MMB"
    convert requant_mmproj.py "$M/$MMB" "$M/$MMQ"
    [ -n "${KEEP_ORIGINAL:-}" ] || rm -f "$M/$MMB"
fi

echo
echo "done. Start the server with:  $here/qwen-tin-server   (OpenAI-compatible API on http://127.0.0.1:8080)"
[ "$M" = "$repo/models/Qwen3.8-Flash-Next" ] || echo "  (set QWEN_TIN_MODELS=$M first)"
