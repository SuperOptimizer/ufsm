#!/usr/bin/env bash
# inkev.sh CKPT TAG: ink screen of one checkpoint on the held-out human ink labels (rebuilt 11-01 from the lost
# ~/ufsm-data/inkft/scripts/ev.sh: the holdout-F1 part; the w058 corr part is gone, its control labels were lost and it was not
# informative, corr <= 0 for every checkpoint). Per box: predict (halo 0, window 512 = shard, EMA, --channel from PX) and ufsm eval
# (F1 at p 0.5, tolerance 2, unknown ignored) against the "tight" ink-splat human labels. One row TAG<TAB>F1... -> $E/eval.tsv.
# Boxes: 0139 w039 (9.362 um scan, level 0), 0139 w039 (2.399 um scan, level 2), 1667 w028 (level 2), Paris 4 segment 20231031143852
# (level 2). PX: extra predict args (a v2 universal checkpoint: PX="--channel ink"; ift3 / ink96: none).
# Env: E (laptop ~/ufsm-data/v2eval/ink; desk /vesuvius/ufsm/v2/eval/ink), U (pinned binary $E/../bin/ufsm), LAB (labels dir: the
# desk's /vesuvius/ufsm/v2/ink/human/tight or its copy $E/labels), CA (HTTP chunk cache), LOCK (GPU flock file), GPU (0).
set -u
CK=$(readlink -f "${1:?usage: inkev.sh CKPT TAG}"); TAG=${2:?TAG}
if [ -d /vesuvius/ufsm/v2 ]; then E=${E:-/vesuvius/ufsm/v2/eval/ink}; LAB=${LAB:-/vesuvius/ufsm/v2/ink/human/tight}; LOCK=${LOCK:-/vesuvius/ufsm/v2/eval/gpu${GPU:-1}.lock}
else E=${E:-$HOME/ufsm-data/v2eval/ink}; LAB=${LAB:-$E/labels}; LOCK=${LOCK:-$HOME/ufsm-data/gpu.lock}; fi
U=${U:-$(dirname $E)/bin/ufsm}; CA=${CA:-$(dirname $E)/cache}; W=$E/work/$TAG; mkdir -p $W $CA
R=https://dl.ash2txt.org/community-uploads/forrest/volcomp
CAP=(); [ -x $HOME/ufsm-data/capped.sh ] && CAP=(env MEM=${MEMJ:-12G} $HOME/ufsm-data/capped.sh)
PR=(--halo 0 --ema 1 --q 8 --threads 8 --gpu ${GPU:-0} ${PX:-})
row="$TAG"
while read -r nm k um lv box lab; do
  o=$W/ho_$nm
  [ -d $o ] || flock $LOCK "${CAP[@]}" $U predict $CK $R $k $o --um $um --level $lv --box $box --window 512 --shard 512 --levels 1 --cache $CA "${PR[@]}" > $o.log 2>&1 ||
    { echo "inkev: $nm predict failed ($o.log)"; tail -n 3 $o.log; }
  IFS=, read -r bz by bx _ <<< "$box"; f=$((1 << lv))
  "${CAP[@]}" $U eval $o . $LAB/$lab . --um $um --level $lv --pred-origin $((bz * f)),$((by * f)),$((bx * f)) --thr 0.5 --tol 2 > $o.eval 2>&1
  f1=$(grep -E '^0\.50 ' $o.eval | awk '{print $4}'); row="$row	${f1:-na}"
  echo "$nm: $(grep -E '^0\.50' $o.eval)"
done <<'EOF'
0139w039 PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr 9.362 0 4000,3317,2700,768,768,1024 PHerc0139-human.zarr
0139w039h24 PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr 2.399 2 3197,2477,2600,768,768,1024 PHerc0139-human24.zarr
1667w028 PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr 2.399 2 4096,1024,512,768,768,768 PHerc1667-human.zarr
P4seg PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr 2.4 2 7424,3072,4096,768,768,768 PHercParis4-human.zarr
EOF
[ -s $E/eval.tsv ] || printf 'tag\t0139w039\t0139w039h24\t1667w028\tP4seg\n' > $E/eval.tsv
echo "$row" >> $E/eval.tsv
echo "$row"
