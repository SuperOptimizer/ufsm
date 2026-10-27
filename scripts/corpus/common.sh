# scripts/corpus/common.sh: shared settings of the corpus unroll (sourced by desk.sh and fd.sh).
# Two hosts, pipelined (user 10-27):
#   desk (forlindesk2, 2 x 5060 Ti): per scroll of list.tsv prep (mask box, axis) -> wind96 fit fields (GPU 0) || ift3-24000 ink (GPU 1)
#        -> push fields + ink + the CT level + box/axis to forlindesk (rsync, byte-checked) -> forlindesk:<S>/state/ready -> delete the
#        desk fields; the next scroll's predicts start right away (the push runs beside them).
#   forlindesk (2 x 3090, 60 GB RAM; it cannot ssh to the desk): every ready scroll in list order: count-field ref -> snapped slab fits
#        (one per GPU) -> fit-join --ref -> wraps + surf-stats -> SLIM flatten -> upright CT | ink renders + sheets.tsv -> delete fields.
# Root C=/vesuvius/ufsm/corpus on both hosts (src = master clone; the desk build is build/, forlindesk's ARCH=86 build is build/ too).
# Per scroll C/<S>: state/<stage>.done, state/failed (a stage failed RETRY+1 times: the loops skip it; rm it to retry), logs/, run.log.
# Stop: touch C/STOP on that host (checked between stages and slabs).
set -u
C=${C:-/vesuvius/ufsm/corpus}; U=${U:-$C/src/build/ufsm}
CK=/vesuvius/ufsm/paris4/ckpt/wind96-wk1-best.ckpt; INKCK=/vesuvius/ufsm/ink-ft/runs/ift3/snap/ift3-24000.ckpt
FD=${FD:-forrest@192.168.0.242}   # forlindesk as seen from the desk
I=64; OV=16                       # slab interiors / overlap (field cells), as Paris 4 run3-cf
scan() {   # scan S: R K UM LV AXS LUM of list.tsv (LUM: the level's voxel um)
  IFS=$'\t' read -r _ R K UM LV AXS < <(awk -F'\t' -v s=$1 '$1 == s' $C/list.tsv); [ -n "${K:-}" ] || return 1
  LUM=$(awk -v u=$UM -v l=$LV 'BEGIN { printf "%.4g", u * 2 ^ l }'); O=$C/$1; F=$O/fields; REF=$O/ref; mkdir -p $O/state $O/logs; }
log() { echo "$(date '+%m-%d %H:%M:%S') $S $*" | tee -a $O/run.log $C/corpus.log; }
done_() { [ -e $O/state/$1.done ]; }
mark() { echo "$(date '+%F %T') ${2:-}" > $O/state/$1.done; }
stopped() { [ -e $C/STOP ]; }
memok() { while [ $(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) -lt ${MEMGB:-40} ]; do sleep 30; done; }
space() { local f=$(df -BG --output=avail $C | tail -n 1 | tr -dc 0-9); [ $f -ge ${SPACEGB:-120} ] || { log "only $f GB free (SPACEGB ${SPACEGB:-120})"; return 1; }; }
N() { nice -n 5 "$@"; }
run() {   # run STAGE FN: the stage unless done, with RETRY (2) retries; returns 1 (and marks the scroll failed) when they are used up
  local st=$1 fn=$2 k t0; done_ $st && return 0; stopped && return 1
  for k in $(seq 0 ${RETRY:-2}); do
    t0=$(date +%s); log "$st start (try $k)"
    if $fn; then mark $st "$(( $(date +%s) - t0 )) s"; log "$st done $(( $(date +%s) - t0 )) s"; return 0; fi
    log "$st FAILED (try $k, logs/$st*)"; stopped && return 1; sleep 30
  done
  touch $O/state/failed; log "giving up at $st"; return 1
}
