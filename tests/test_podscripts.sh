#!/usr/bin/env bash
# test_podscripts.sh (2026-11-14, mini audit podscripts): static checks of the pod-facing scripts for the rules the audit found broken.
#  1. no desk-to-pod label staging: pod_pack.sh copies no label store (gt/, inkgt/ data dirs) and refuses an OUT holding a zarr;
#  2. no hardcoded pod endpoint as a default (root@<ip> or a -p port in ${VAR:-...}): pods change daily, the endpoint must be passed;
#  3. trainer launches on pods detach with setsid (nohup alone dies with the caller's process group);
#  4. the pod GPU smoke test runs under flock /root/gpu.lock with a timeout.
# CPU, < 1 s.
set -u
cd "$(dirname "$0")/.." || exit 1
fails=0; ck() { if eval "$2"; then echo "ok: $1"; else echo "FAIL: $1"; fails=$((fails + 1)); fi; }
P=scripts/univ/pod_pack.sh
ck "pod_pack stages no label sets" '! grep -nE "cp -a[lL]? [^#]*(codesA|codesB|/phase|qA|inkgt/\\\$d|\\\$G/\\\$d)" $P'
ck "pod_pack refuses a zarr in OUT" 'grep -q "label data must not go to a pod" $P'
hits=$(grep -rnE '\$\{[A-Z_]+:-[a-z]+@[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' scripts --include='*.sh' | grep -v '^scripts/test/' | grep -vE '@(192\.168|10)\.' || true)   # (LAN hosts: the desk / fd, stable)
ck "no hardcoded pod endpoint defaults${hits:+: $hits}" '[ -z "$hits" ]'
nos=$(grep -rnE '^[^#]*[^d] nohup bash [^ ]*(train|u1\.train|train_hv)[^ ]*\.sh' scripts/univ/pod_univ.sh scripts/fibers/*.launch.sh scripts/pod 2>/dev/null | grep -v setsid || true)
ck "pod trainer launches use setsid${nos:+: $nos}" '[ -z "$nos" ]'
ck "pod smoke test under the GPU lock + timeout" 'grep -qE "flock( -w [0-9]+)? /root/gpu.lock timeout -k [0-9]+ [0-9]+ \./\\\$BD/test_mx_sw" scripts/pod/setup.sh'
ck "evald marks done only on success" 'grep -q "if \[ \$NF = 0 \]; then rm -f \$o/failed; touch \$o/done" scripts/fibers/evald.sh'
Q2=scripts/pod/podq2.sh   # (11-18: the next pod session's queue)
ck "podq2 / deskq2 / score1667 parse" 'bash -n $Q2 && bash -n scripts/pod/deskq2.sh && bash -n scripts/pod/q2pull.sh && bash -n scripts/pod/score1667.sh'
ck "deskq2 pins card 1 for every stage" 'grep -q "export CUDA_VISIBLE_DEVICES=1 FIELDGPUS=0 FITGPUS=0 INKGPU=0; unset SGPU" scripts/pod/deskq2.sh'
ck "podq2 probes the desk before the unroll" '[ "$(grep -n "^if ! done_ desk" $Q2 | tail -1 | cut -d: -f1)" -lt "$(grep -n "^if ! done_ unroll" $Q2 | cut -d: -f1)" ]'
ck "podq2 only pushes from the pod (no rsync from the desk)" '! grep -nE "push [^#]*\\\$DESK:[^ ]* +\\\$(RU|Q|K)" $Q2'
ck "podq2 keeps the per-wrap ink for the GT score" 'grep -q "KEEPINK=1" $Q2 && grep -q "KEEPINK" scripts/unroll/unroll.sh'
t=$(printf '%s\n' "2026-11-18 03:20:05 rectosp start" "2026-11-18 03:20:11 rectosp done: 6 s wall, 32 CPU s, 5.6 GB max RSS" | awk "$(sed -n "s/^  awk '\(.*\)' \\\\$/\1/p" $Q2)")
ck "podq2 stage table row${t:+ ($t)}" '[ "$t" = "$(printf "rectosp\t2026-11-18 03:20:05\t2026-11-18 03:20:11\t6\t32\t5.6")" ]'
[ $fails = 0 ] && echo "test_podscripts: ok" || echo "test_podscripts: $fails failure(s)"
exit $((fails != 0))
