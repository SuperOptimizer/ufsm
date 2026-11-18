#!/bin/bash
# r3ink.sh SURFDIR OUT.tif CT.tif: the unroll's production ink on one rendered surface (INKMODE=r3, the default since 2026-11-17).
# ink-ext with a frozen 2-D teacher (reader3; R3W=a,b: an ensemble, e.g. reader3 + argus2, the mean probability) on the surface SURFDIR
# (a tifxyz dir: a strip stripe or an up/ wrap), written on the same render grid as ufsm render --scale 1 (ink-ext's canvas TIF is that grid)
# over OUT.tif (the render's ink, from the CT given as render's --volume), then papyrus-gated: OUT x [CT.tif >= R3GATE] (img-gate --min,
# CT.tif = the render's filled CT, so the gate follows the render's own fill / recentring onto papyrus).
# Why: the latent ink head (inkpod 15000) reached held-out AUC 0.582 on 1667 per wrap vs reader3 0.858 on the same samples (DESIGN
# "2-D ink teachers on the try9-6250 pod unrolls"); the user switched production ink to reader3 (2026-11-17). INKMODE=lat keeps latink.
# R3PRE=DIR: the map was made beforehand by ONE ink-ext call over all the run's surfaces (unroll.sh r3pre: decoded CT chunks stay resident
# across surfaces and a shared --cache keeps the fetched ones; neighbouring 1667 wraps share ~83 % of their chunks, unrollperf 11-17):
# DIR/<name>_<model>.tif is taken (moved) instead of running ink-ext here.
#   env: U (ufsm), R3CTR R3CTK (CT root + scan key, as ink-ext --ct), R3W (weights, default reader-v3), R3MODEL (reader3), R3GATE (80; 0 = no
#        gate), R3GPU (CUDA device, default the caller's), R3X (more ink-ext args, e.g. --ct-local ROOT --local-levels 1,2 --cache DIR), R3TH (8)
set -u
SD=${1:?surface dir}; OUT=${2:?out tif}; CT=${3:?ct tif}
U=${U:?ufsm binary}; R3CTR=${R3CTR:?CT root}; R3CTK=${R3CTK:?CT scan key}
R3W=${R3W:-/vesuvius/ufsm/inksdk/weights/ink_9um@reader-v3/reader-v3.pth}; R3MODEL=${R3MODEL:-reader3}; R3GATE=${R3GATE:-80}
for w in ${R3W//,/ }; do [ -s "$w" ] || { echo "r3ink: no weights $w (R3W)" >&2; exit 1; }; done
n=$(basename "$SD"); T=$(dirname "$OUT")/.r3ink_$n.$$
case "$T" in */.r3ink_*) ;; *) echo "r3ink: bad temp dir $T" >&2; exit 1;; esac
if [ -n "${R3PRE:-}" ] && [ -s "$R3PRE/${n}_$R3MODEL.tif" ]; then mv -f "$R3PRE/${n}_$R3MODEL.tif" "$OUT" || exit 1
else
mkdir -p "$T" || exit 1
trap 'rm -rf -- "$T"' EXIT
${R3GPU:+env CUDA_VISIBLE_DEVICES=$R3GPU} nice -n 10 $U ink-ext --model $R3MODEL --weights $R3W --surface "$SD" --ct $R3CTR $R3CTK --out "$T" \
  --threads ${R3TH:-8} ${R3X:-} > "$T/log" 2>&1 || { echo "r3ink: ink-ext failed on $SD:" >&2; tail -5 "$T/log" >&2; exit 1; }
[ -s "$T/${n}_$R3MODEL.tif" ] || { echo "r3ink: ink-ext left no $T/${n}_$R3MODEL.tif" >&2; exit 1; }
grep -E "tiles/s|^ink-ext: $n " "$T/log" | tail -2 >&2
mv -f "$T/${n}_$R3MODEL.tif" "$OUT" || exit 1
fi
[ "$R3GATE" = 0 ] && exit 0
$U img-gate "$OUT" "$CT" --min "$R3GATE" || { echo "r3ink: img-gate failed" >&2; exit 1; }
