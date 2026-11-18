#!/usr/bin/env bash
# test_wbench_fail.sh: wbench exits 1 with a FAILED count when a box gets no row (lead 11-18: every predict failed, rc was 0), and 0 when
# every box has its row. A fake ufsm whose predict fails; no GPU, no data (the box is never predicted for real).
set -u; T=$(mktemp -d); trap 'rm -rf -- "$T"' EXIT; f=0
printf '#!/bin/sh\necho "fake ufsm: $1 fails" >&2; exit 1\n' > $T/ufsm; chmod +x $T/ufsm; printf 'UFSM{"step":1}\n' > $T/ck.ckpt
run() { env WB=$T SITE=desk U=$T/ufsm CPU=1 SLABS=0 PBATCH=0 DLOCKS= NOLOCK=1 WZARGS= BOXES="PH1 PH2" bash scripts/wbench/wbench.sh $T/ck.ckpt $1 > $T/$1.out 2>&1; }
run tfail; rc=$?
[ $rc = 1 ] || { echo "FAIL: rc $rc with every predict failing (want 1)"; f=1; }
grep -q 'FAILED: 2 boxes / sites without a row (PH1 PH2)' $T/tfail.out || { echo "FAIL: no FAILED count"; tail -5 $T/tfail.out; f=1; }
mkdir -p $T/res; printf 'tag\tbox\n' > $T/res/tok.tsv; for b in PH1 PH2; do printf 'tok\t%s\tparis4\n' $b >> $T/res/tok.tsv; done
run tok; rc=$?
[ $rc = 0 ] || { echo "FAIL: rc $rc with every box done (want 0)"; tail -5 $T/tok.out; f=1; }
[ $f = 0 ] && echo "test_wbench_fail: ok"; exit $f
