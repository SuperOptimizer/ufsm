#!/usr/bin/env bash
# test_unroll_cards.sh (2026-11-18): unroll.sh's card placement. A desk run with the caller's CUDA_VISIBLE_DEVICES=1 and FITGPUS=0 ran
# perwrap's flatten / render / reader3 on card 0 (the training card): the per-job lines set CUDA_VISIBLE_DEVICES=<id> as given. Now every
# per-job id goes through pcard (an index into the caller's list; a physical card under GVIS). Checks:
#  1. pcard / lg / sgpu from unroll.sh itself under the caller's list (1; 0,1; an index out of range fails) and under GVIS;
#  2. no per-job CUDA_VISIBLE_DEVICES=$var in unroll.sh / strip_refill.sh that skips pcard (static);
#  3. with CARD_UUID=<uuid prefix> (by hand, GPU): `ufsm sysinfo` under the card pcard picks for CUDA_VISIBLE_DEVICES=$CARD reports it.
# CPU, < 1 s (3: one CUDA context).
set -u
cd "$(dirname "$0")/.." || exit 1
F=scripts/unroll/unroll.sh; fails=0; ck() { if eval "$2"; then echo "ok: $1"; else echo "FAIL: $1"; fails=$((fails + 1)); fi; }
defs=$(awk '/^pcard\(\) \{/,/^  else echo \$1; fi; \}$/ {print} /^lg\(\) \{/ {print} /^sgpu\(\) \{/,/^  return 0; \}$/ {print}' $F)
ck "the helpers are found" '[ $(printf "%s\n" "$defs" | grep -cE "^(pcard|lg|sgpu)\(\)") = 3 ]'
run() { env -i PATH="$PATH" bash -c "$defs
$1" 2>/dev/null; }
ck "caller CVD=1: per-job 0 -> card 1" '[ "$(run "GVIS=; CUDA_VISIBLE_DEVICES=1; pcard 0")" = 1 ]'
ck "caller CVD=0,1: per-job 1 -> card 1" '[ "$(run "GVIS=; CUDA_VISIBLE_DEVICES=0,1; pcard 1")" = 1 ]'
ck "caller CVD=1: per-job 1 fails (not in the list)" '! run "GVIS=; CUDA_VISIBLE_DEVICES=1; pcard 1" > /dev/null'
ck "GVIS 1: per-job 1 stays card 1" '[ "$(run "GVIS=1; CUDA_VISIBLE_DEVICES=1; pcard 1")" = 1 ]'
ck "no caller CVD, no GVIS: unchanged" '[ "$(run "GVIS=; unset CUDA_VISIBLE_DEVICES; pcard 0")" = 0 ]'
ck "caller CVD=1 SGPU=0: card 1" '[ "$(run "GVIS=; CUDA_VISIBLE_DEVICES=1; SGPU=0; sgpu")" = CUDA_VISIBLE_DEVICES=1 ]'
# (gpu / g are mapped where they are set, pw_one / r3pre; any other variable, RGPU included, must call pcard on the line itself)
bad=$(grep -nE 'CUDA_VISIBLE_DEVICES=\$\{?[A-Za-z_]+' $F scripts/strip/strip_refill.sh | grep -vE 'CUDA_VISIBLE_DEVICES=\$(gpu|g)\b|pcard|CUDA_VISIBLE_DEVICES=\$\{?(GVIS|SGPU|sg|CUDA_VISIBLE_DEVICES|a\[)' || true)
ck "every per-job CUDA_VISIBLE_DEVICES goes through pcard${bad:+: $bad}" '[ -z "$bad" ] && grep -q "gpu=\$(pcard \$2)" $F && grep -q "g=\$(pcard \$g)" $F && grep -q "CUDA_VISIBLE_DEVICES=\$(pcard \$RGPU)" $F'
if [ -n "${CARD_UUID:-}" ]; then
  c=$(run "GVIS=; CUDA_VISIBLE_DEVICES=${CARD:-1}; pcard 0"); u=$(CUDA_VISIBLE_DEVICES=$c ${U:-build/ufsm} sysinfo 2>&1 | grep -oE 'uuid [0-9a-f-]+' | head -1)
  ck "a real GPU call on card $c: $u (want $CARD_UUID)" '[ "${u#uuid }" != "" ] && case "$CARD_UUID" in "${u#uuid }"*) true;; *) false;; esac'
fi
[ $fails = 0 ] && echo "test_unroll_cards: ok" || echo "test_unroll_cards: $fails failure(s)"
exit $((fails != 0))
