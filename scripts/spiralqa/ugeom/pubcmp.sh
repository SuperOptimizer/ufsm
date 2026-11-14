#!/usr/bin/env bash
# pubcmp.sh RUN OUT [WRAPS (wraps_snap)] [CT_L4_RAW]: an unroll's wraps vs the published PHerc1667 unwrap, the audit ugeom budget (CPU, ~15 min,
# ~12 GB): wrap-dump -> surfcmp.py (pub -> ours d, sheet offset; ours -> pub) -> per-vertex radius / pitch / hole / angle (budget1667.py)
# -> the binned tables with the handedness ramp removed (bins1667.py) and the area budget (budget.py, needs ctval.npy from onct.py).
# Verify the dir fix: same_raw ~= same_ramp and a flat 64-bin ramp; the masked-CT fix: "B1 on zero (masked) CT" near 0.
set -euo pipefail
RUN=$1 OUT=$2 WR=${3:-wraps_snap} CT=${4:-/vesuvius/ufsm/gt/pub1667-cmp/ct_l4.raw}
U=${U:-ufsm}; H=$(cd "$(dirname "$0")" && pwd)
PUB=${PUB:-/vesuvius/ufsm/gt/pub1667-unwrap/20260612121456-on-20251217075048-2.399um.tifxyz}
mkdir -p "$OUT"; cd "$OUT"
[ -s ours_l0.f32 ] || nice -n 10 "$U" wrap-dump "$RUN/$WR" ours_l0.f32 --z -10,1e9 --every 1 --grid-um 2.399
[ -s full/pub_dist.npz ] || python3 -I "$H/surfcmp.py" "$PUB" ours_l0.f32 "$RUN/axis.json" full --um 2.399 | tee surfcmp.log
[ -s b1667_aux.npz ] || python3 -I "$H/budget1667.py" "$PUB" full/pub_dist.npz "$RUN/axis.json" b1667
python3 -I "$H/bins1667.py" full/pub_dist.npz b1667_aux.npz > bins1667.txt
[ -s ctval.npy ] || (CT_RAW=$CT python3 -I "$H/onct.py" > onct.txt)
python3 -I "$H/budget.py" > budget.txt; cat budget.txt
rm -f ours_l0.f32   # 2 GB; pub_dist.npz holds everything the tables read
