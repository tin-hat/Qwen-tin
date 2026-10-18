#!/usr/bin/env python3
# mmproj mix in the style of HermiHg's Q5_K-MIX: Q5_0 attention/ffn_up, F16 ffn_down (4304-wide rows fit no block quant),
# Q8_0 projector, F32 norms/biases/embeddings
import os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gguf-py"))
import gguf
from gguf import GGMLQuantizationType as T

if len(sys.argv) != 3:
    sys.exit("usage: requant_mmproj.py <BF16 mmproj> <output>")
src, dst = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(src)
w = gguf.GGUFWriter(dst, arch=r.fields["general.architecture"].contents(), use_temp_file=False)
for name, field in r.fields.items():
    if name.startswith("GGUF.") or name == "general.architecture":
        continue
    vt = field.types[0]
    if vt == gguf.GGUFValueType.ARRAY:
        w.add_key_value(name, field.contents(), vt, sub_type=field.types[-1])
    else:
        w.add_key_value(name, field.contents(), vt)

def as_f32(t):
    shape = [int(x) for x in reversed(t.shape.tolist())]
    if t.tensor_type == T.BF16:
        return (t.data.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if t.tensor_type == T.F16:
        return t.data.view(np.float16).astype(np.float32).reshape(shape)
    if t.tensor_type == T.F32:
        return t.data.reshape(shape)
    raise ValueError(t.tensor_type)

total = 0
for t in r.tensors:
    n = t.name
    target = None
    if re.search(r"\.(attn_qkv|attn_out|ffn_up)\.weight$", n):
        target = T.Q5_0
    elif re.search(r"\.ffn_down\.weight$", n):
        target = T.F16
    elif re.match(r"mm\.\d+\.weight$", n):
        target = T.Q8_0
    if target is None or t.tensor_type == target:
        w.add_tensor(n, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
        total += t.n_bytes
        continue
    f = as_f32(t)
    if target == T.F16:
        q = f.astype(np.float16)
        w.add_tensor(n, q, raw_shape=q.shape, raw_dtype=T.F16)
    else:
        q = gguf.quants.quantize(f, target)
        w.add_tensor(n, q, raw_shape=q.shape, raw_dtype=target)
    total += q.nbytes
print(f"{total/2**20:.1f} MiB of tensor data", flush=True)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(progress=False); w.close()
