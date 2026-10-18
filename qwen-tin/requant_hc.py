#!/usr/bin/env python3
# Rewrite a GGUF shard with the BF16 hyper-connection down/up matrices requantized to Q8_0.
# Every other tensor and every metadata key is copied as is.
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gguf-py"))
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as T  # noqa: E402

if len(sys.argv) != 3:
    sys.exit("usage: requant_hc.py <shard 1 of the IQ3_S GGUF> <output>")
src, dst = sys.argv[1], sys.argv[2]
pat = re.compile(r"(^|\.)(hc_attn_down|hc_attn_up|hc_ffn_down|hc_ffn_up|output_hc_down|output_hc_up)\.weight$")

r = gguf.GGUFReader(src)
arch = r.fields["general.architecture"].contents()
w = gguf.GGUFWriter(dst, arch=arch, use_temp_file=False)

for name, field in r.fields.items():
    if name.startswith("GGUF.") or name == "general.architecture":
        continue
    vtype = field.types[0]
    if vtype == gguf.GGUFValueType.ARRAY:
        w.add_key_value(name, field.contents(), vtype, sub_type=field.types[-1])
    else:
        w.add_key_value(name, field.contents(), vtype)

n_q = 0
saved = 0
for t in r.tensors:
    shape = [int(x) for x in reversed(t.shape.tolist())]  # numpy order
    if pat.search(t.name) and t.tensor_type == T.BF16:
        f32 = (t.data.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        q = gguf.quants.quantize(f32, T.Q8_0)
        w.add_tensor(t.name, q, raw_shape=q.shape, raw_dtype=T.Q8_0)
        n_q += 1
        saved += t.n_bytes - q.nbytes
    else:
        w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)

print(f"requantized {n_q} tensors to Q8_0, {saved / 2**20:.1f} MiB smaller", flush=True)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file(progress=False)
w.close()
print("done", flush=True)
