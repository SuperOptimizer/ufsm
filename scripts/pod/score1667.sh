#!/bin/bash
# scripts/pod/score1667.sh DIR [TAG]: GT-score a PHerc1667 unroll synced to the desk by scripts/pod/podq2.sh (DIR = its DESKDIR), desk CPU only,
# 4 threads, lowprio. fix0332's recipe (/vesuvius/ufsm/fix0332/gt1667/score.sh, 11-17/18), made portable:
#   counting   wrap-check --codes (GT turns per fitted turn by radius band, CT peaks per turn, verdicts; the GT, not the CT peaks, is the
#              judge: DESIGN "The curve on three scrolls") and gtboxes.sh (GT ids right on the 1667 wbench boxes not tagged label-noise) on
#              ref-cf and joined, fields DIR/fields/rsp
#   ink        scripts/inkgt/inkscore.sh on the per-wrap surfaces DIR/up with their CT-gated reader3 maps DIR/ink/wraps (KEEPINK=1): the
#              HELD-OUT AUC at 20 / 40 / 80 um (ink-once desk reference, DESIGN "Ink the papyrus once": 0.8664 / 0.8622 / 0.8500)
# -> DIR/score/{score.txt, wc-*.log, gtboxes-*/, inkscore-TAG.*}. Run it after DIR/.synced exists:
#   ~/bin/memcap ~/bin/lowprio bash scripts/pod/score1667.sh /vesuvius/ufsm/podun/PHerc1667-q2
set -u
D=${1:?usage: score1667.sh DIR [TAG]}; T=${2:-$(basename $D)}; O=$D/score; mkdir -p $O
R=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd); U=${U:-$R/build/ufsm}
[ -e $D/.synced ] || echo "score1667: $D/.synced missing: the pod's final push has not finished (scoring what is there)" >&2
F=$D/fields/rsp; [ -e $F/fields.json ] || { echo "score1667: no fields at $F" >&2; exit 2; }
CR=/vesuvius/ufsm/gt/PHerc1667-wind-20261016/codes; CK=4.798
M=https://dl.ash2txt.org/community-uploads/forrest/volcomp; CTK=PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr/2
BX=$O/boxes.tsv; awk -F'\t' '$2 == "1667" && $10 !~ /label-noise/' $R/scripts/wbench/boxes.tsv > $BX
for m in ref-cf joined; do [ -e $D/$m ] || continue
  [ -s $O/wc-$m.log ] || CUDA_VISIBLE_DEVICES= nice -n 10 $U wrap-check $D/$m $F --ct $M $CTK --ct-um 9.596 --codes $CR $CK --threads 4 --bands $O/bands-$m.tsv > $O/wc-$m.log 2>&1
  [ -s $O/gtboxes-$m/summary.tsv ] || CUDA_VISIBLE_DEVICES= bash $R/scripts/spiralqa/ugeom/gtboxes.sh $F $D/$m $BX $CR $CK $O/gtboxes-$m $U > $O/gtboxes-$m.log 2>&1
  echo "$T $m: $(grep -a 'GT turns per fitted turn' $O/wc-$m.log | head -1 | sed 's/^ *//') | $(grep -a 'peaks per fitted' $O/wc-$m.log | head -1 | awk '{print "CT/turn", $7}') | $(grep -a '^wrap-check: verdicts' $O/wc-$m.log) | $(tail -1 $O/gtboxes-$m.log)"
done | tee $O/score.txt
if [ -d $D/ink/wraps ] && [ -d $D/up ]; then
  [ -s $O/inkscore-$T.tsv ] || OUTD=$O bash $R/scripts/inkgt/inkscore.sh $T $D/up $D/ink/wraps > $O/inkscore-$T.out 2>&1
  echo "$T ink (held out): $(grep -aiE 'held.?out' $O/inkscore-$T.log 2>/dev/null | tail -3 | tr '\n' ' ')" | tee -a $O/score.txt
else echo "$T ink: no $D/up + $D/ink/wraps (KEEPINK=1 not set on the pod?)" | tee -a $O/score.txt; fi
