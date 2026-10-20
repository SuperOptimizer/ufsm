"""A minimal safetensors reader (numpy) for the parity tests, so they need no safetensors package."""
import json, struct
import numpy as np

_DT = {"F32": np.float32, "F16": np.float16, "F64": np.float64, "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in head.items():
            if k == "__metadata__":
                continue
            b0, b1 = v["data_offsets"]
            f.seek(base + b0)
            a = np.frombuffer(f.read(b1 - b0), dtype=_DT[v["dtype"]]).reshape(v["shape"])
            out[k] = a.copy()
        return out
