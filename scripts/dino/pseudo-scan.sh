#!/usr/bin/env bash
# pseudo-scan.sh OUTROOT SCROLL SCAN UM [NBOX]: winding pseudo labels over NBOX (default 6) papyrus-rich 512^3 boxes of one scan at its
# ~9 um level LEVEL (default 0: the ESRF scans; UM = the level-0 um), for dino-train GT SupCon on volumes without labels.
#   axis: the scan's published prod96 fit fields.json (mirror; [y, x] per native z) -> umbilicus control points
#   boxes: ufsm dino-boxes --mask (the scan's scanprep cell mask PREP/<S>-<SC>.mask.raw): NB random non-overlapping 512^3 boxes with an
#          object-cell fraction >= MINFRAC (0.8), away from the scan's z ends
#   per box: pseudo-box.sh (wind96 predict -> fit -> gated band codes); OUTROOT/SCROLL/sources.json collects the box entries
#   whose fit agrees with the network phase (pseudo.agree_cos >= MINCOS, default 0.8) and the scan's norm (from sources-u0.json if NORMS given)
# Holdouts: PHerc1667 and PHerc1447 are refused (eval scans). Env: R (CT root), PREP (scanprep dir, default /vesuvius/ufsm/u0/prep), MINFRAC, SEED, FIELDS (fit-fields zarr name), everything of pseudo-box.sh.
set -u
O=$1 S=$2 SC=$3 UM=$4 NB=${5:-6}; R=${R:-https://dl.ash2txt.org/community-uploads/forrest/volcomp}
case $S in PHerc1667*|PHerc1447*|PHercParis4*) echo "pseudo-scan: $S is an eval scan (or has real GT: Paris 4), refused"; exit 2;; esac
D=$(dirname $(readlink -f $0)); U=${U:-$HOME/ufsm-data/dino/bin/ufsm-fs24}; export U
o=$O/$S${LEVEL:+-$SC}; mkdir -p $o   # (other volumes of the scroll: <S>-<scan>)
G=$(curl -sf $R/$S/volumes/ 2>/dev/null | grep -oE "$SC-[^\"/]*masked\.zarr" | head -1); [ -n "$G" ] && G=$S/volumes/$G || G=${GROUP:?cannot list $R/$S/volumes; set GROUP}
FZ=${FIELDS:-$SC-fitfields-20261016-ufsm-prod96-h2.zarr}
M=${PREP:-/vesuvius/ufsm/u0/prep}/$S-$SC.mask.raw; [ -s $M ] || { echo "pseudo-scan: no scanprep mask $M"; exit 1; }
if [ ! -s $o/axis.json ]; then   # the published fit fields' axis, else the mask's per-z object centroid (dino-boxes --mask-axis; ~0.5 mm off on 0139)
  if [ "${MASKAXIS:-0}" = 0 ] && curl -sf --retry 3 $R/$S/representations/predictions/fit-fields/$FZ/fields.json -o $o/fields.json; then
    jq -c '.origin as $o | {control_points: [.axis | to_entries[] | select(.key % 32 == 0) | {z: (.key + $o[0]), y: (.value[0] + $o[1]), x: (.value[1] + $o[2]), score: 1}]}' $o/fields.json > $o/axis.json
  else rm -f $o/fields.json; $U dino-boxes - --mask-axis $M > $o/axis.json || { echo "pseudo-scan: mask axis failed"; exit 1; }; echo "$S: axis from the scanprep mask"; fi
fi
[ -s $o/boxes.json ] || $U dino-boxes - --mask $M --out-level ${LEVEL:-0} --block 512 --top $NB --min-frac ${MINFRAC:-0.8} --seed ${SEED:-1} > $o/boxes.json 2> $o/boxes.log || { echo "pseudo-scan: boxes failed"; exit 1; }
nb=$(jq length $o/boxes.json); echo "$S: $nb boxes"
for i in $(seq 0 $((nb - 1))); do
  b=$(jq -r ".[$i] | map(tostring) | join(\",\")" $o/boxes.json)
  PNAME=$S-$SC-pb$i $D/pseudo-box.sh $o/b$i $R $G $UM $o/axis.json $b
done
jq -s --argjson mc ${MINCOS:-0.8} '[.[] | select(.pseudo.agree_cos >= $mc)]' $o/b*/source.json > $o/sources.json 2> /dev/null
ls $o/b*/source2.json > /dev/null 2>&1 && jq -s --argjson mc ${MINCOS:-0.8} '[.[] | select(.pseudo.agree_cos >= $mc)]' $o/b*/source2.json > $o/sources2.json 2> /dev/null
echo "$S: $(jq length $o/sources.json) of $nb boxes kept (agree_cos >= ${MINCOS:-0.8}); labelled % $(jq -r '[.[] | .pseudo.labelled_pct] | map(tostring) | join(" ")' $o/sources.json); agree_cos $(jq -r '[.[] | .pseudo.agree_cos] | map(tostring) | join(" ")' $o/sources.json)"
