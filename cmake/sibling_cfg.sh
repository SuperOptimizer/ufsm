#!/bin/bash
# sibling_cfg.sh CMAKE DIR CMAKE-ARGS...: configure the sibling build dir (test-prep: the release dir of a verify dir, or the reverse) with
# this dir's settings, unless DIR was already configured with exactly these arguments (DIR/.ufsm-sibling-cfg). A re-configure of an
# existing dir re-generates every ninja rule: 7.2 s on the loaded desk (2026-11-20), paid by every `--target test`. When the arguments
# are the same, DIR's own build.ninja re-runs cmake on a CMakeLists / *.cmake change (ninja's regeneration rule), so skipping is safe.
c=$1 d=$2; shift 2
s=$(printf '%q ' "$@")
if [ -f "$d/build.ninja" ] && [ -f "$d/CMakeCache.txt" ] && [ "$(cat "$d/.ufsm-sibling-cfg" 2> /dev/null)" = "$s" ]; then exit 0; fi
"$c" "$@" -B "$d" || exit
printf '%s' "$s" > "$d/.ufsm-sibling-cfg"
