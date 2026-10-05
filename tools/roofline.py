#!/usr/bin/env python3
"""Per-layer roofline of a `UFSM_PROF=layers ufsm train` log: measured ms per conv pass against the hardware floor
max(FLOPs / MMA peak, bytes / DRAM bandwidth) of the same pass, ranked by the time above the floor.

usage: roofline.py TRAIN_LOG [--P 704] [--split 1] [--steps-per-report 20] [--gpus 2]
Assumptions (RTX 5060 Ti, the --fp4 2 recipe): MMA peak 425 TFLOP/s for fp4 operands (MXFP4, fp32 accumulate),
213 for fp8, 54 for 16-bit (enc0.c1); DRAM 448 GB/s; activations stored MX-fp4 (4.25 bits / element), activation
gradients MX-fp8 (8.25 bits); each pass reads its operands once and writes its result once (no halo re-reads), so the
floor is optimistic. Layer slots as in src/unet.c: slot = 2 * layer + conv, layers enc0..3 = 0..3, down0..2 = 4..6,
dec2..0 = 7..9, head = 10; 4-level net, widths 16,32,64,80, 4 input channels."""
import argparse, re

W = [16, 32, 64, 80]
PEAK = {'fp4': 425e12, 'fp8': 213e12, 'fp16': 54e12}
BW = 448e9
B_ACT, B_GRAD = 4.25 / 8, 8.25 / 8


def convs():
    """slot -> (name, level, cin, cout, k, stride, precision)"""
    c = {}
    for l in range(4):
        cin = 4 if l == 0 else W[l]   # enc l >= 1 reads the down-conv output (W[l] channels)
        c[2 * l] = ('enc%d.c1' % l, l, cin, W[l], 3, 1, 'fp16' if l == 0 else 'fp4')
        c[2 * l + 1] = ('enc%d.c2' % l, l, W[l], W[l], 3, 1, 'fp4')
    for l in range(3):
        c[2 * (4 + l)] = ('down%d' % l, l, W[l], W[l + 1], 3, 2, 'fp4')
    for l in range(3):   # dec l: concat(up(W[l+1]), skip W[l]) -> W[l]; layer id 9 - l
        lid = 9 - l
        c[2 * lid] = ('dec%d.c1' % l, l, W[l + 1] + W[l], W[l], 3, 1, 'fp4')
        c[2 * lid + 1] = ('dec%d.c2' % l, l, W[l], W[l], 3, 1, 'fp4')
    c[20] = ('head', 0, W[0], 2, 1, 1, 'fp16')
    return c


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('log'); a.add_argument('--P', type=int, default=704); a.add_argument('--split', type=int, default=1)
    a.add_argument('--halo', type=int, default=8); a.add_argument('--steps-per-report', type=int, default=20); a.add_argument('--gpus', type=int, default=2)
    g = a.parse_args()
    text = open(g.log).read()
    blocks = text.split('per-op GPU ms over the last')
    if len(blocks) < 2: raise SystemExit('no profile reports in the log (run with UFSM_PROF=layers)')
    last = blocks[-1]
    slots = {int(m.group(1)): tuple(map(float, m.group(2, 3, 4)))
             for m in re.finditer(r'slot\s+(\d+) fwd ([\d.]+) bwd_data ([\d.]+) bwd_w ([\d.]+) ms', last)}
    norm = g.steps_per_report * g.gpus   # the report sums all GPUs over steps-per-report steps
    D0 = (g.P // 2 + g.halo) if g.split else g.P   # level-0 depth computed per GPU
    rows = []
    for s, (name, l, cin, cout, k, st, prec) in convs().items():
        if s not in slots: continue
        D, H = D0 >> l, g.P >> l
        vin = D * H * H
        vout = vin // 8 if st == 2 else vin
        flops = 2 * k ** 3 * cin * cout * vout
        peak = PEAK[prec]
        passes = {
            'fwd': (flops, vin * cin * B_ACT + vout * cout * B_ACT),
            'bwd_data': (flops, vout * cout * B_GRAD + vin * cin * B_GRAD),
            'bwd_w': (flops, vin * cin * B_ACT + vout * cout * B_GRAD),
        }
        for j, (p, (f, b)) in enumerate(passes.items()):
            ms = slots[s][j] / norm
            if ms <= 0: continue
            t_mma, t_mem = f / peak * 1e3, b / BW * 1e3
            floor = max(t_mma, t_mem)
            rows.append((ms - floor, name, p, prec, ms, floor, 'mma' if t_mma > t_mem else 'mem', f / (ms * 1e-3) / 1e12))
    rows.sort(reverse=True)
    tot = sum(r[4] for r in rows); tfl = sum(r[5] for r in rows)
    print('per GPU per step: convs %.1f ms measured, %.1f ms floor (%.0f%% of floor speed)' % (tot, tfl, 100 * tfl / tot if tot else 0))
    print('%-10s %-8s %-5s %9s %9s %6s %5s %8s' % ('conv', 'pass', 'prec', 'ms', 'floor ms', 'x', 'bound', 'TFLOP/s'))
    for gap, name, p, prec, ms, floor, bound, tf in rows:
        print('%-10s %-8s %-5s %9.2f %9.2f %6.1f %5s %8.1f' % (name, p, prec, ms, floor, ms / floor, bound, tf))
    cats = re.findall(r'^\s+(\S[^\n]*?)\s+([\d.]+) ms\s+([\d.]+)%$', last, re.M)
    if cats:
        print('\nop categories (all GPUs, %d steps):' % g.steps_per_report)
        for n, ms, pc in cats: print('  %-16s %9.1f ms %5.1f%%' % (n, float(ms), float(pc)))


if __name__ == '__main__':
    main()
