#!/bin/bash
# sweep_psnr.sh [ARM...]: every sweep arm's CT as the readers see it (ufsm read of the arm's tree, no cache: local roots have none) against
# the anchor's on one 384^3 box per scan: differing bytes, rms error, PSNR. -> sw/psnr.tsv. An arm equal to the anchor is void.
L=/workspace/latstore; U=$L/ufsm/build/ufsm; SW=$L/sw; T=$SW/psnr; mkdir -p $T
B="PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2:12352,4160,3648 PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr/2:1280,1152,896 PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/0:5440,3136,3904"
for a in ${*:-$(ls $SW | grep -E '^(orig|vq|lv)')}; do
  i=0; for kb in $B; do i=$((i + 1)); k=${kb%%:*}; o=${kb##*:}; [ -e $SW/$a/${k%/*}/.done ] || continue
    [ -e $T/orig.$i.raw ] || $U read $SW/orig $k ${o//,/ } 384 384 384 $T/orig.$i.raw > /dev/null 2>&1
    grep -qP "^$a\t$i\t" $SW/psnr.tsv 2>/dev/null && continue
    $U read $SW/$a $k ${o//,/ } 384 384 384 $T/a.raw > /dev/null 2>&1 || { echo "read failed $a $k"; continue; }
    python3 -I -c "
import numpy as np, sys, hashlib
a = np.fromfile('$T/orig.$i.raw', np.uint8); b = np.fromfile('$T/a.raw', np.uint8); d = a.astype(np.float64) - b; m = a > 0
mse = (d[m] ** 2).mean()
print('$a\t$i\t%d\t%.3f\t%.2f\t%s' % ((a != b).sum(), mse ** 0.5, 10 * np.log10(255 ** 2 / mse) if mse > 0 else 99, hashlib.md5(b.tobytes()).hexdigest()[:12]))" >> $SW/psnr.tsv
  done
done
sort -k2,2n -k4,4n $SW/psnr.tsv
