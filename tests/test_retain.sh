#!/usr/bin/env bash
# test_retain.sh (2026-11-14, ops audit #19): scripts/run/retain.py on a temp tree. Old entries go; a new one, one with a symlink into
# it from outside, one pinned by KEEP and one a process has as its cwd stay; --apply deletes only the dropped ones. CPU, ~1 s.
set -u
cd "$(dirname "$0")/.." || exit 1
T=$(mktemp -d "${TMPDIR:-/tmp}/tretain.XXXXXX") || exit 1
pid=; trap '[ -n "$pid" ] && kill $pid 2>/dev/null; case "$T" in */tretain.*) rm -rf -- "$T";; esac' EXIT
fails=0; ck() { if eval "$2"; then echo "ok: $1"; else echo "FAIL: $1"; fails=$((fails + 1)); fi; }
D=$T/scratch; mkdir -p $D/old1/sub $D/old2 $D/new $D/linked $D/pinned $D/busy $T/live
echo x > $D/old1/sub/a; echo y > $D/old2/b; echo z > $D/oldfile; echo n > $D/new/c; echo l > $D/linked/d; echo p > $D/pinned/e; touch $D/pinned/KEEP
echo q > $D/busy/f
touch -h -d '10 days ago' $D/old1/sub/a $D/old1/sub $D/old1 $D/old2/b $D/old2 $D/oldfile $D/linked/d $D/linked $D/pinned/e $D/pinned/KEEP $D/pinned \
  $D/busy/f $D/busy
touch -d '10 days ago' $D/new; echo fresh > $D/new/c   # (an old dir with one fresh file: kept)
ln -s $D/linked/d $T/live/to_d
( cd $D/busy && exec sleep 60 ) & pid=$!
sleep 0.3
R=(python3 scripts/run/retain.py --only-extra --rule $D:3 --link-roots $T)
out=$("${R[@]}"); echo "$out"
ck "old1, old2, oldfile would go" '[ $(echo "$out" | grep -c "^drop") = 3 ] && echo "$out" | grep -q "drop.*/old1$" && echo "$out" | grep -q "drop.*/oldfile$"'
ck "new kept (a fresh file inside)" '! echo "$out" | grep -q "/new"'
ck "linked kept (symlink named)" 'echo "$out" | grep -q "keep.*/linked  (symlink $T/live/to_d"'
ck "pinned kept (KEEP)" 'echo "$out" | grep -q "keep.*/pinned  (pinned"'
ck "busy kept (in use)" 'echo "$out" | grep -q "keep.*/busy  (in use"'
ck "dry run deleted nothing" '[ -e $D/old1/sub/a ] && [ -e $D/oldfile ]'
ck "json plan parses" '"${R[@]}" --json | python3 -c "import json,sys; d=json.load(sys.stdin); assert len([e for e in d[\"entries\"] if not e[\"keep\"]]) == 3"'
"${R[@]}" --apply > /dev/null; rc=$?
ck "apply rc 0" '[ $rc = 0 ]'
ck "apply removed old1 / old2 / oldfile" '[ ! -e $D/old1 ] && [ ! -e $D/old2 ] && [ ! -e $D/oldfile ]'
ck "apply kept new / linked / pinned / busy" '[ -e $D/new/c ] && [ -e $D/linked/d ] && [ -e $D/pinned/e ] && [ -e $D/busy/f ]'
[ $fails = 0 ] && echo "test_retain: ok" || echo "test_retain: $fails failure(s)"
exit $((fails != 0))
