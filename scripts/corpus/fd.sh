#!/usr/bin/env bash
# fd.sh: the forlindesk side of the corpus unroll (see common.sh). Loop, in list.tsv order: the next scroll with state/fields.ready (the
# desk pushed fields + ref) and no wraps yet -> slabs -> join -> wraps -> state/wraps.ready (the desk pulls joined + wraps and does
# surf-stats, flatten and renders); its fields and slab fits are deleted here.
#   slabs   fit --ref-model REF --ref-snap 1 --uw-repair 0 --dir (the ref's: count-field finds the handedness, 1447 is -1) per 64-cell slab (+16 overlap), full cross-section, one lane per GPU; air
#           slabs at the box ends (fit: no data for the prior) count as done and are left out of the join
#   join    fit-join --ref REF (prints the overlap integer agreement)
#   wraps   ufsm wraps (.sfc, step 4)
# Ends when the desk has finished (C/desk.done). Start:
#   cd /vesuvius/ufsm/corpus && mkdir -p run && cp src/scripts/corpus/{common,fd}.sh run/ && setsid nohup bash run/fd.sh > fd.out 2>&1 < /dev/null &
#   (a copy: editing the repo never touches the running loop; fd.sh finds mkcrop.sh in src/scripts/wbench)
# Env: TH (20), MEMGB (12: MemAvailable GB before a fit), CLEAN (1), STOPF.
H=$(cd "$(dirname "$0")" && pwd); . $H/common.sh
TH=${TH:-20}; MEMGB=${MEMGB:-12}

lane() {   # lane GPU: unclaimed slabs in z order
  local g=$1 k s d c0 c1 a b t0 rc ze=$(jq -r '.shape[0]' $F/fields.json) ny=$(jq -r '.shape[1]' $F/fields.json) nx=$(jq -r '.shape[2]' $F/fields.json)
  for k in $(seq 0 $(( (ze + I - 1) / I - 1 ))); do
    stopped && return 0
    s=$(printf 's%03d' $k); d=$O/slabs/$s; mkdir -p $d
    exec 9> $d/claim; flock -n 9 || continue; [ -e $d/done ] && continue
    c0=$((k * I)); c1=$((c0 + I)); [ $c1 -gt $ze ] && c1=$ze; a=$((c0 - OV)); [ $a -lt 0 ] && a=0; b=$((c1 + OV)); [ $b -gt $ze ] && b=$ze
    rm -rf $d/f $d/fit; bash $C/src/scripts/wbench/mkcrop.sh $F $d/f $a 0 0 $((b - a)) $ny $nx > /dev/null || continue
    echo "$c0 $c1 $a $b" > $d/cells; memok; t0=$(date +%s)
    env UFSM_FIT_THREADS=10 nice -n 5 $U fit $d/f --gpu $g --out $d/fit --dir $(jq -r .dir $REF/model.json) --ref-model $REF --ref-snap 1 --uw-repair 0 > $d/fit.log 2>&1; rc=$?
    echo "$s $c0..$c1 rc $rc gpu $g $(( $(date +%s) - t0 ))s" >> $O/slabs.txt
    if [ $rc = 0 ]; then touch $d/done; elif grep -q "no data for the prior" $d/fit.log; then touch $d/empty $d/done; fi   # (air slabs: the box ends)
  done
}
slabs() {
  lane 0 & lane 1 & wait
  local ns=$(( ($(jq -r '.shape[0]' $F/fields.json) + I - 1) / I )) nd=$(ls -d $O/slabs/s???/done 2>/dev/null | wc -l)
  log "slabs $nd of $ns done, $(awk '{s += $NF + 0} END {printf "%.2f", s / 3600}' $O/slabs.txt) GPU-h"; [ $nd = $ns ]
}
joinst() {
  local args=() d c0 c1 a b
  for d in $(ls -d $O/slabs/s??? | sort); do [ -e $d/done ] || return 1; [ -e $d/empty ] && continue; read -r c0 c1 a b < $d/cells; args+=($d/fit $((2 * c0)) $((2 * c1))); done
  rm -rf $O/joined; N $U fit-join $O/joined $F --blend 16 --ref $REF "${args[@]}" > $O/logs/join.log 2>&1 || return 1
  log "join: $(grep -oE 'integer agreement[^:]*: [0-9.]+|mean share\(mode\)[^:]*: [0-9.]+' $O/logs/join.log | tr '\n' ' ')"
}
wraps() {
  rm -rf $O/wraps; N $U wraps $O/joined $O/wraps --step 4 --out-um $LUM --sfc 0.25 --fields $F --threads $TH > $O/logs/wraps.log 2>&1 || return 1
  echo "$(date '+%F %T')" > $O/state/wraps.ready
  log "wraps $(ls $O/wraps | grep -c '^w') -> wraps.ready"
  [ "${CLEAN:-1}" = 1 ] && rm -rf $F $O/slabs/s???/f $O/slabs/s???/fit; return 0   # (the joined model is standalone)
}

while ! stopped; do
  fs=
  for S in $(scrolls); do
    scan $S || continue; failed && continue; done_ wraps && continue
    [ -e $O/state/fields.ready ] && { fs=$S; break; }
  done
  if [ -n "$fs" ]; then S=$fs; scan $S; run slabs slabs && run join joinst && run wraps wraps; continue; fi
  [ -e $C/desk.done ] && break
  sleep 300
done
echo "$(date '+%m-%d %H:%M:%S') fd.sh done" >> $C/corpus.log
