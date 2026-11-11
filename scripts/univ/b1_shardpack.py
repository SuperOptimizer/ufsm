"""shardpack.py: per 9.6 um shard (512^3 at 9.6 = 2048^3 native) of the Paris 4 June phaseA, the share of labelled cells with sheet
spacing < 150 um (spacing = 9.6 um / |grad phase| in turns per voxel, wrapped central differences), from 4 sampled 32 x 96 x 96 blocks
per shard (ufsm read). Writes p4_shardpack.tsv: z y x (shard), labelled cells, packed share, median spacing, bytes of the 9.6 shard."""
import os, subprocess, sys, numpy as np
from multiprocessing import Pool
U = "/mnt/scratch/univ/b1/ufsm.labelfast"; R = "/vesuvius/ufsm/gt/paris4-wind-20261019/phaseA"; C = f"{R}/9.6/c"
def one(sh):
    z, y, x = sh; tmp = f"/dev/shm/sp_{os.getpid()}.raw"; sp = []
    for oz, oy, ox in ((128, 128, 128), (128, 320, 320), (320, 128, 320), (320, 320, 128)):
        a = [z * 512 + oz, y * 512 + oy, x * 512 + ox, 32, 96, 96]
        r = subprocess.run([U, "read", R, "9.6"] + [str(v) for v in a] + [tmp, "--threads", "1"], capture_output=True)
        if r.returncode: continue
        v = np.fromfile(tmp, np.uint8).reshape(32, 96, 96).astype(float); ph = np.where(v < 252, v / 252, np.nan)
        g2 = np.zeros_like(ph)
        for ax in range(3):
            d = (np.roll(ph, -1, ax) - np.roll(ph, 1, ax)) / 2; d = (d + 0.5) % 1 - 0.5; g2 += d * d
        g = np.sqrt(g2)[1:-1, 1:-1, 1:-1]; g = g[~np.isnan(g) & (g > 1e-4)]; sp.append(9.6 / g)
    try: os.unlink(tmp)
    except OSError: pass
    s = np.concatenate(sp) if sp else np.zeros(0); b = os.path.getsize(f"{C}/{z}/{y}/{x}")
    return (z, y, x, len(s), float((s < 150).mean()) if len(s) else 0.0, float(np.median(s)) if len(s) else 0.0, b)
sh = [tuple(int(p) for p in os.path.relpath(os.path.join(d, f), C).split("/")) for d, _, fs in os.walk(C) for f in fs]
with Pool(12) as P: rows = P.map(one, sorted(sh), chunksize=8)
with open("/mnt/scratch/univ/b1/p4_shardpack.tsv", "w") as f:
    for r in rows: f.write("\t".join(str(v) for v in r) + "\n")
print(len(rows), "shards")
