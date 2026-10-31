#!/usr/bin/env bash
# scripts/v2/scrolls.sh [CFG]: the v2 scroll table. Per scroll of CFG (scripts/v2/runs.cfg: S|split|cutoff|run dirs|strip dirs|extra dirs)
# builds a symlink run tree V2/wraps_src/<S>/ that scripts/unroll/snap_existing.sh reads as RUN:
#   wraps/wNNN.sfc  newest copy per name among the run dirs (the tiered pod pulls split wraps over /vesuvius and /mnt/scratch), newer than cutoff
#   wrapcheck.tsv   newest among the run dirs (newer than cutoff)
#   axis.json, fields  first run dir (then extra dir) holding them; fields must have fields.json + a readable pc plane
# and writes V2/scrolls.tsv (one row per scroll):
#   scroll split nwraps wraps wrapcheck strip_json strip_log axis ct_root ct_key ct_um ct_level fields fields_packed
#   ink_src ink_key ink_model ink_mode ink_seam (scripts/v2/inksrc.sh: the pseudo-ink source, its model and window-seam handling)
# strip_json = the strip's strip.json ("wraps": [first, last] = the trim), strip_log = its logs/strip.log; ct_root = V2/ct when the
# fetched scan is complete there (fetch state done or a symlinked local scan), else "-" (pending). Rerun after the fetch to fill ct_root.
# Env: V2 (/vesuvius/ufsm/v2), U (V2/src/build/ufsm). Never touches the source runs.
set -u
H=$(cd "$(dirname "$0")" && pwd); REPO=$(cd $H/../.. && pwd)
CFG=${1:-$H/runs.cfg}; V2=${V2:-/vesuvius/ufsm/v2}; U=${U:-$V2/src/build/ufsm}
OUT=$V2/scrolls.tsv; TMP=$OUT.tmp; PACKLOG=${PACKLOG:-/vesuvius/ufsm/clean/pack-result.tsv}
printf 'scroll\tsplit\tnwraps\twraps\twrapcheck\tstrip_json\tstrip_log\taxis\tct_root\tct_key\tct_um\tct_level\tfields\tfields_packed\tink_src\tink_key\tink_model\tink_mode\tink_seam\n' > $TMP
newest() { local cut=$1; shift; find "$@" -maxdepth 0 -newermt "$cut" -size +0 -printf '%T@\t%p\n' 2>/dev/null | sort -gr | head -1 | cut -f2; }
grep -v '^#' $CFG | while IFS='|' read -r S SPLIT CUT RUNS STRIPS EXTRA FCFG; do
  [ -n "$S" ] || continue
  # (the tree is built once: a snap may be reading it; REBUILD=1 rebuilds; the fields link is refreshed each run)
  W=$V2/wraps_src/$S; [ "${REBUILD:-0}" = 1 ] && rm -rf $W; NEW=0; [ -d $W/wraps ] || { mkdir -p $W/wraps; NEW=1; }
  # wraps: newest per name, newer than the cutoff
  [ $NEW = 1 ] &&   for R in ${RUNS//,/ }; do [ -d $R/wraps ] && find $R/wraps -maxdepth 1 -name 'w*.sfc' -newermt "$CUT" -size +0 -printf '%T@\t%f\t%p\n'; done |
    sort -t$'\t' -k2,2 -k1,1gr | awk -F'\t' '!seen[$2]++ { print $3 }' | while read -r p; do ln -s $p $W/wraps/; done
  nw=$(ls $W/wraps | wc -l)
  wc=$(newest "$CUT" $(for R in ${RUNS//,/ }; do echo $R/wrapcheck.tsv; done)); [ $NEW = 1 ] && [ -n "$wc" ] && ln -s $wc $W/wrapcheck.tsv
  ax=; for R in ${RUNS//,/ } ${EXTRA//,/ }; do [ -z "$ax" ] && [ -s $R/axis.json ] && ax=$(readlink -f $R/axis.json); done
  # fields: the cfg store; "asis:" = usable now, else only once CLEAN (PACKLOG) lists it PACKED (no half-merged store mid-pack)
  fl=${FCFG#asis:}; [ "$fl" = - ] && fl=
  if [ -n "$fl" ] && [ "$fl" = "$FCFG" ]; then rel=${fl#/mnt/scratch/}; rel=${rel#/vesuvius/ufsm/}
    awk -F"\t" -v r=$rel '$1 == r && $2 == "PACKED" { f = 1 } END { exit !f }' $PACKLOG 2>/dev/null || fl=; fi
  [ -n "$fl" ] && { [ -s $fl/fields.json ] && [ -e $fl/pc/zarr.json ] || fl=; }
  [ $NEW = 1 ] && [ -n "$ax" ] && ln -s $ax $W/axis.json
  if [ -e $V2/wraps_snap/$S/claim ]; then fl=$(readlink $W/fields 2>/dev/null); FIX=1; else rm -f $W/fields; FIX=0; fi   # (started or done: keep its fields)
  pk=-; if [ -n "$fl" ]; then
    read -r fz fy fx < <(jq -r '.shape | map(tostring) | join(" ")' $fl/fields.json)
    if $U read $fl pc $((fz / 2)) $((fy / 2)) $((fx / 2)) 32 32 32 /dev/null > /dev/null 2>&1; then
      [ $FIX = 0 ] && ln -sfn $fl $W/fields; pk=raw; grep -q fields_pack $fl/pc/zarr.json && pk=q4
    else echo "scrolls: $S fields $fl unreadable: not used" >&2; fl=; fi
  fi
  sj=$(newest "$CUT" $(for T in ${STRIPS//,/ }; do echo $T/strip.json $T/strip/strip.json; done))
  sl=$(newest "$CUT" $(for T in ${STRIPS//,/ } ${RUNS//,/ }; do echo $T/logs/strip.log; done))
  row=$(awk -F'\t' -v s=$S '$1 == s' $REPO/scripts/corpus/list.tsv $REPO/scripts/unroll/scrolls.tsv | head -1)
  IFS=$'\t' read -r _ R0 K UM LV _ <<< "$row"
  cr=-; vol=$(basename $K); tag=$S-${vol%%-*}
  if [ -L $V2/ct/$K ] || [ -e $V2/fetch/state/$tag.copy.done ]; then cr=$V2/ct; fi
  ink=$(bash $H/inksrc.sh $S)
  printf '%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $S $SPLIT $nw $W/wraps ${wc:--} ${sj:--} ${sl:--} ${ax:--} $cr $K $UM $LV ${fl:--} $pk "$ink" >> $TMP
done
mv $TMP $OUT
column -t $OUT | cut -c1-200
