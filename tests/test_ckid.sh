#!/usr/bin/env bash
# tests/test_ckid.sh: scripts/wbench/ckid.sh, the checkpoint-identity guard of wbench / wgate / ueval (eval audit 11-14 #21): a tag dir
# records sha256 prefix + header step; the same checkpoint again is accepted, another one refused (return 1, the old id kept), and
# FORCE_TAG=1 accepts it and records the new id. CPU only, ~0.1 s.
set -u
H=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
D=$(mktemp -d "${TMPDIR:-/tmp}/test_ckid_XXXXXX")
fail() { echo "FAIL: $*"; echo "kept $D"; exit 1; }
. "$H/scripts/wbench/ckid.sh"
printf 'UFSM{"nlev":6,"step":1234,"widths":[1]}\n' > "$D/a.ckpt"; head -c 4096 /dev/urandom >> "$D/a.ckpt"
printf 'UFSM{"nlev":6,"step":5678,"widths":[1]}\n' > "$D/b.ckpt"; head -c 4096 /dev/urandom >> "$D/b.ckpt"
ckid_check "$D/tag" "$D/a.ckpt" || fail "a fresh tag dir refused"
A=$CKID; [[ "$A" == *" step 1234" ]] || fail "id '$A' lacks the header step"
[ "$(cat "$D/tag/ckid")" = "$A" ] || fail "the id was not recorded"
ckid_check "$D/tag" "$D/a.ckpt" || fail "the same checkpoint refused on a rerun"
FORCE_TAG=0 ckid_check "$D/tag" "$D/b.ckpt" 2> "$D/err" && fail "another checkpoint accepted under the same tag"
grep -q 'holds results of checkpoint' "$D/err" || fail "no refusal message: $(cat "$D/err")"
[ "$(cat "$D/tag/ckid")" = "$A" ] || fail "a refused checkpoint overwrote the recorded id"
FORCE_TAG=1 ckid_check "$D/tag" "$D/b.ckpt" 2> "$D/err" || fail "FORCE_TAG=1 refused"
grep -q 'FORCE_TAG=1' "$D/err" || fail "FORCE_TAG=1 did not say the outputs mix"
[[ "$(cat "$D/tag/ckid")" == *" step 5678" ]] || fail "FORCE_TAG=1 did not record the new id"
[[ "$(ckid_of "$D/missing")" == "nosha step ?" ]] || fail "a missing file: '$(ckid_of "$D/missing")'"
case "$D" in "${TMPDIR:-/tmp}"/test_ckid_*) rm -rf -- "$D" ;; esac
echo "test_ckid: ok"
