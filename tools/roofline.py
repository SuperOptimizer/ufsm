#!/usr/bin/env python3
"""Per-layer roofline of a `UFSM_PROF=layers ufsm train` log: measured ms per conv pass against the hardware floor
max(FLOPs / MMA peak, bytes / DRAM bandwidth) of the same pass, ranked by the time above the floor.

usage: roofline.py TRAIN_LOG [--P 704] [--split 1] [--steps-per-report 20] [--gpus 2] [--widths 16,32,64,80] [--cout 2]
Defaults (RTX 5060 Ti, the --fp4 2 recipe): MMA peak 425 TFLOP/s for fp4 operands (MXFP4, fp32 accumulate),
213 for fp8, 54 for 16-bit and fp32 accumulate, 23 for fp32 CUDA-core kernels; DRAM 448 GB/s (--peak-fp4 / --peak-fp8 /
--peak-16 / --peak-fp32 / --bw for other cards); activations stored MX-fp4 (4.25 bits / element), activation gradients
MX-fp8 (8.25 bits); each pass reads its operands once and writes its result once (no halo re-reads), so the floor is
optimistic. The precision of each pass is the first one the log's executed_compute manifest lists for it (fp4 for 3^3
convs, fp16 for the head, when the log has none). Layer slots as in src/unet.c: slot = 2 * layer + conv with, for L
levels, enc0..enc(L-1) = 0..L-1, down0..down(L-2) = L..2L-2, dec(L-2)..dec0 = 2L-1..3L-3, head = 3L-2.
Shapes: enc l conv1 W[l-1] (4 inputs at l = 0) -> W[l], conv2 W[l] -> W[l]; down l W[l] -> W[l] at stride 2;
dec l conv1 W[l + 1] + W[l] -> W[l], conv2 W[l] -> W[l]; head W[0] -> cout (1^3)."""
import argparse, re

B_ACT, B_GRAD = 4.25 / 8, 8.25 / 8


def convs(W, cout):
    """slot -> (name, level of the input, cin, cout, k, stride)"""
    L, c = len(W), {}
    for l in range(L):
        c[2 * l] = ('enc%d.c1' % l, l, 4 if l == 0 else W[l - 1], W[l], 3, 1)
        c[2 * l + 1] = ('enc%d.c2' % l, l, W[l], W[l], 3, 1)
    for l in range(L - 1):
        c[2 * (L + l)] = ('down%d' % l, l, W[l], W[l], 3, 2)
        lid = 3 * L - 3 - l
        c[2 * lid] = ('dec%d.c1' % l, l, W[l + 1] + W[l], W[l], 3, 1)
        c[2 * lid + 1] = ('dec%d.c2' % l, l, W[l], W[l], 3, 1)
    c[2 * (3 * L - 2)] = ('head', 0, W[0], cout, 1, 1)
    return c


def executed(text):
    """conv name -> [fwd, bwd_data, wgrad] precision (first listed) from the executed_compute manifest"""
    m = re.search(r'executed_compute \([^)]*\):([^\n]*)', text)
    out = {}
    for tok in (m.group(1).split() if m else []):
        name, _, v = tok.partition('=')
        out[name] = [p.split('|')[0] for p in v.split(':')]
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('log'); a.add_argument('--P', type=int, default=704); a.add_argument('--split', type=int, default=1)
    a.add_argument('--halo', type=int, default=None, help='split halo planes at level 0 (default 2^(L-1))')
    a.add_argument('--steps-per-report', type=int, default=20); a.add_argument('--gpus', type=int, default=2)
    a.add_argument('--widths', default='16,32,64,80'); a.add_argument('--cout', type=int, default=2)
    a.add_argument('--peak-fp4', type=float, default=425); a.add_argument('--peak-fp8', type=float, default=213)
    a.add_argument('--peak-16', type=float, default=54); a.add_argument('--peak-fp32', type=float, default=23)
    a.add_argument('--bw', type=float, default=448, help='DRAM GB/s')
    a.add_argument('--reports', default=None, help='python slice of the profile reports to average (default 1:-1 with three or more '
                   'reports, which drops the warm-up and the final validation interval, else the last one)')
    g = a.parse_args()
    W = [int(w) for w in g.widths.split(',')]
    peak = {'fp4': g.peak_fp4, 'fp8': g.peak_fp8, 'fp16': g.peak_16, 'bf16': g.peak_16, 'fp32': g.peak_fp32}
    halo = g.halo if g.halo is not None else 1 << (len(W) - 1)
    text = open(g.log).read()
    blocks = text.split('per-op GPU ms over the last')
    if len(blocks) < 2: raise SystemExit('no profile reports in the log (run with UFSM_PROF=layers)')
    reps = blocks[1:]
    if g.reports: reps = reps[slice(*[int(v) if v else None for v in g.reports.split(':')])]
    else: reps = reps[1:-1] if len(reps) >= 3 else reps[-1:]
    if not reps: raise SystemExit('no profile reports selected')
    last = reps[-1]
    ex = executed(text)
    slots = {}
    for r in reps:
        for m in re.finditer(r'slot\s+(\d+) fwd ([\d.]+) bwd_data ([\d.]+) bwd_w ([\d.]+) ms', r):
            v = slots.setdefault(int(m.group(1)), [0.0, 0.0, 0.0])
            for j in range(3): v[j] += float(m.group(2 + j)) / len(reps)
    norm = g.steps_per_report * g.gpus   # the report sums all GPUs over steps-per-report steps
    D0 = (g.P // 2 + halo) if g.split else g.P   # level-0 depth computed per GPU
    rows = []
    for s, (name, l, cin, cout, k, st) in convs(W, g.cout).items():
        if s not in slots: continue
        D, H = max(D0 >> l, 1), max(g.P >> l, 1)
        vin = D * H * H
        vout = vin // 8 if st == 2 else vin
        flops = 2 * k ** 3 * cin * cout * vout
        precs = ex.get(name, ['fp16'] * 3 if name == 'head' else ['fp4'] * 3)
        passes = {
            'fwd': (flops, vin * cin * B_ACT + vout * cout * B_ACT),
            'bwd_data': (flops, vout * cout * B_GRAD + vin * cin * B_GRAD),
            'bwd_w': (flops, vin * cin * B_ACT + vout * cout * B_GRAD),
        }
        for j, (p, (f, b)) in enumerate(passes.items()):
            ms = slots[s][j] / norm
            if ms <= 0: continue
            prec = precs[j] if precs[j] in peak else 'fp4'
            t_mma, t_mem = f / (peak[prec] * 1e12) * 1e3, b / (g.bw * 1e9) * 1e3
            floor = max(t_mma, t_mem)
            rows.append((ms - floor, name, p, prec, ms, floor, 'mma' if t_mma > t_mem else 'mem', f / (ms * 1e-3) / 1e12))
    rows.sort(reverse=True)
    tot = sum(r[4] for r in rows); tfl = sum(r[5] for r in rows)
    print('per GPU per step: convs %.1f ms measured, %.1f ms floor (%.0f%% of floor speed)' % (tot, tfl, 100 * tfl / tot if tot else 0))
    print('%-10s %-8s %-5s %9s %9s %6s %5s %8s' % ('conv', 'pass', 'prec', 'ms', 'floor ms', 'x', 'bound', 'TFLOP/s'))
    for gap, name, p, prec, ms, floor, bound, tf in rows:
        print('%-10s %-8s %-5s %9.2f %9.2f %6.1f %5s %8.1f' % (name, p, prec, ms, floor, ms / floor, bound, tf))
    cat = {}
    for r in reps:
        for n, ms, pc in re.findall(r'^\s+(\S[^\n]*?)\s+([\d.]+) ms\s+([\d.]+)%$', r, re.M): cat[n] = cat.get(n, 0.0) + float(ms) / len(reps)
    if cat:
        ct = sum(cat.values())
        print('\nop categories (all GPUs, per %d steps, mean of %d reports):' % (g.steps_per_report, len(reps)))
        for n, ms in cat.items(): print('  %-16s %9.1f ms %5.1f%%' % (n, ms, 100 * ms / ct))


if __name__ == '__main__':
    main()
