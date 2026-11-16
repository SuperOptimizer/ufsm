# scripts/wbench/ckid.sh (sourced): ckid_check DIR CKPT: the checkpoint's identity (sha256 prefix + header step) in DIR/ckid; a DIR that
# already holds another checkpoint's identity is refused (return 1), because reruns under one tag reuse its fields, predictions and rows
# (eval audit 11-14 #21: wbench / wgate / ueval / eval rows carried no checkpoint hash, and $O/ckpt was overwritten by a rerun with
# another checkpoint while the old rows stayed). FORCE_TAG=1 accepts the new checkpoint (and says so). Sets CKID.
ckid_of() { local s st; s=$(sha256sum "$1" 2> /dev/null | cut -c1-16); st=$(head -c 400000 "$1" 2> /dev/null | head -1 | grep -ao '"step":[0-9]*' | head -1 | cut -d: -f2)
  echo "${s:-nosha} step ${st:-?}"; }
ckid_check() {
  local d=$1 ck=$2 old; CKID=$(ckid_of "$ck"); mkdir -p "$d"
  if [ -s "$d/ckid" ] && old=$(cat "$d/ckid") && [ "$old" != "$CKID" ]; then
    if [ "${FORCE_TAG:-0}" = 1 ]; then echo "ckid: $d was $old, now $CKID (FORCE_TAG=1): earlier outputs of this tag are mixed in" >&2
    else echo "ckid: $d holds results of checkpoint $old, not $CKID ($ck): use another tag (FORCE_TAG=1 mixes them)" >&2; return 1; fi
  fi
  echo "$CKID" > "$d/ckid"; return 0; }
