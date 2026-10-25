#!/usr/bin/env bash
# ufsm wbench-stats: a constant paired shift gives that difference with a zero-width CI and P(A>B) 1; the clean set drops ceil < 0.9;
# replicate rows of a box are averaged; stratified resampling keeps the mean of equal boxes.
set -u; U=${1:-build/ufsm}; T=$(mktemp -d); trap 'rm -rf $T' EXIT; f=0
printf 'tag\tbox\tscroll\tids\tceil\n' > $T/a.tsv; printf 'tag\tbox\tscroll\tids\tceil\n' > $T/b.tsv
for i in 1 2 3 4 5 6; do s=s$((i % 2)); v=0.$((3 + i)); echo -e "a\tb$i\t$s\t$v\t0.95" >> $T/a.tsv; echo -e "b\tb$i\t$s\t$(echo "$v - 0.1" | bc -l)\t0.95" >> $T/b.tsv; done
echo -e "a\tbad\ts0\t0.1\t0.5" >> $T/a.tsv; echo -e "b\tbad\ts0\t0.9\t0.5" >> $T/b.tsv
echo -e "a\tb1\ts1\t0.6\t0.95" >> $T/a.tsv; echo -e "b\tb1\ts1\t0.4\t0.95" >> $T/b.tsv   # replicates: b1 A (0.4 + 0.6) / 2, B (0.3 + 0.4) / 2
o=$($U wbench-stats $T/a.tsv $T/b.tsv --metric ids --n 2000)
echo "$o"
echo "$o" | grep -qE '^ids +n +6 ' || { echo "FAIL clean set size"; f=1; }
echo "$o" | grep -qE 'A-B \+0\.1083 ' || { echo "FAIL paired difference"; f=1; }
echo "$o" | grep -qE 'P\(A>B\) 1\.000' || { echo "FAIL P(A>B)"; f=1; }
o=$($U wbench-stats $T/a.tsv --metric ids --set all --n 500); echo "$o" | grep -qE '^ids +n +7 ' || { echo "FAIL --set all"; f=1; }
[ $f = 0 ] && echo "wbench-stats ok"; exit $f
