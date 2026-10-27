# scripts/corpus/common.sh: shared settings of the corpus unroll (sourced by desk.sh, laptop.sh, fd.sh). Three hosts, pipelined (user 10-27):
#   desk (forlindesk2: 2 x 5060 Ti, 32 threads, 182 GB): prep of every scroll (mask box, axis) -> wind96 fit fields, one scroll per GPU
#        (two at once) -> lossless repack to 512^3 shards -> count-field ref on the CPU (REFPAR at once) -> push fields + ref + box/axis to
#        forlindesk (rsync, byte-checked) -> forlindesk:<S>/state/fields.ready, desk fields deleted.
#   laptop (RTX 5080): ift3-24000 ink of every scroll's box, CT read over HTTP from the public mirror (no CT copy, no chunk cache), the
#        ink zarr pushed to forlindesk (byte-checked) -> forlindesk:<S>/state/ink.ready, deleted locally.
#   forlindesk (2 x 3090, 60 GB; cannot ssh to the desk): scrolls with fields.ready in list order: snapped slab fits (one per GPU) ->
#        fit-join --ref -> wraps + surf-stats -> fields deleted -> SLIM flatten; then, once ink.ready: upright CT (mirror) | ink renders +
#        sheets.tsv -> clean.
# Root C on each host: desk + forlindesk /vesuvius/ufsm/corpus, laptop ~/ufsm-data/corpus (scripts from the repo checkout).
# Per scroll C/<S>: state/<stage>.done, state/failed (RETRY+1 failures: skipped; rm it to retry), logs/, run.log; corpus.log per host.
# Stop: touch C/STOP on that host (checked between stages / scrolls / slabs).
set -u
C=${C:-/vesuvius/ufsm/corpus}; U=${U:-$C/src/build/ufsm}
CK=/vesuvius/ufsm/paris4/ckpt/wind96-wk1-best.ckpt; INKCK=/vesuvius/ufsm/ink-ft/runs/ift3/snap/ift3-24000.ckpt
DESK=${DESK:-forrest@192.168.0.175}; FD=${FD:-forrest@192.168.0.242}; FDC=/vesuvius/ufsm/corpus; DC=/vesuvius/ufsm/corpus
MIRROR=https://dl.ash2txt.org/community-uploads/forrest/volcomp
I=64; OV=16                       # slab interiors / overlap (field cells), as Paris 4 run3-cf
scan() {   # scan S: R K UM LV AXS of list.tsv, LUM (the level's voxel um), MK (the scan's key on the mirror), O F REF
  IFS=$'\t' read -r _ R K UM LV AXS < <(awk -F'\t' -v s=$1 '$1 == s' $C/list.tsv); [ -n "${K:-}" ] || return 1
  LUM=$(awk -v u=$UM -v l=$LV 'BEGIN { printf "%.4g", u * 2 ^ l }'); MK=$(dirname $K)/volumes/$(basename $K)
  O=$C/$1; F=$O/fields; REF=$O/ref; mkdir -p $O/state $O/logs; }
scrolls() { tail -n +2 $C/list.tsv | cut -f1; }
log() { echo "$(date '+%m-%d %H:%M:%S') $S $*" | tee -a $O/run.log $C/corpus.log; }
done_() { [ -e $O/state/$1.done ]; }
failed() { [ -e $O/state/failed ]; }
mark() { echo "$(date '+%F %T') ${2:-}" > $O/state/$1.done; }
stopped() { [ -e $C/STOP ]; }
memok() { while [ $(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) -lt ${MEMGB:-40} ]; do sleep 30; done; }
freegb() { df -BG --output=avail $C | tail -n 1 | tr -dc 0-9; }
N() { nice -n 5 "$@"; }
bytes() { du -sb --apparent-size "$@" | awk '{s += $1} END {print s + 0}'; }
run() {   # run STAGE FN: the stage unless done, with RETRY (2) retries; returns 1 (and marks the scroll failed) when they are used up
  local st=$1 fn=$2 k t0; done_ $st && return 0; stopped && return 1
  for k in $(seq 0 ${RETRY:-2}); do
    t0=$(date +%s); log "$st start (try $k)"
    if $fn; then mark $st "$(( $(date +%s) - t0 )) s"; log "$st done $(( $(date +%s) - t0 )) s"; return 0; fi
    log "$st FAILED (try $k, logs/$st*)"; stopped && return 1; sleep 30
  done
  touch $O/state/failed; log "giving up at $st"; return 1
}
claim() { exec 8> $O/state/$1.claim; flock -n 8; }   # claim STAGE: this process owns the scroll's stage (fd 8, released with unclaim / exit)
unclaim() { exec 8>&-; }
