#!/usr/bin/env bash
# scripts/pod/packfields.sh FIELDS [fields-pack options]: shrink a winding fit-field store in place before copying it home (pods: run it
# once the fields stage is done, e.g. on stores written before predict's lossy phase default). FIELDS.pk is written next to it, then
# swapped in; the old store is removed only after the pack succeeded. Readers take the packed store unchanged.
# Default: phase planes at volcomp q FITQP (4, predict's default), the other planes' chunks copied (re-sharded to 512^3). Measured:
# PHerc0490B 21.6 -> 2.85 GB (7.6x), PHerc0483B 30.4 -> 3.47 GB (8.8x); the phase planes 28-31x smaller. Threads: TH (the cgroup CPUs).
set -eu
F=${1:?usage: packfields.sh FIELDS [fields-pack options]}; shift
U=${U:-ufsm}; TH=${TH:-$($U sysinfo --ncpu 2> /dev/null || nproc)}
[ -e "$F/fields.json" ] || { echo "packfields: no $F/fields.json"; exit 1; }
rm -rf "$F.pk"
$U fields-pack "$F" "$F.pk" --q-phase ${FITQP:-4} --threads $TH "$@"
[ -e "$F.pk/fields.json" ] || { echo "packfields: pack failed"; exit 1; }
mv "$F" "$F.old" && mv "$F.pk" "$F" && rm -rf "$F.old"
du -sh "$F"
