#!/usr/bin/env bash
# Worktree-aware version of ~/ufsm-data/fulltest.sh. Run under capped.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/ufsm-venv/bin:/usr/local/cuda/bin:$PATH"
run=${1:-$HOME/ufsm-data/perf-int/fulltest-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$run/tmp"
run=$(realpath "$run")
export TMPDIR="$run/tmp"
printf '%s\n' "$PWD" > "$run/worktree.txt"
git rev-parse HEAD > "$run/head.txt"
git diff HEAD --binary > "$run/change.patch"
: > "$run/testres.txt"
make -j"${J:-8}" VERIFY=0 build/ufsm build/test_sheet build/make_pipeline_fixture \
    build/test_http_reader build/test_sampler_safety build/check_surface_samples build/test_formats \
    > "$run/build.log" 2>&1
# make -qp prints the combined prerequisites of all test-all declarations.
make -qp VERIFY=1 > "$run/make.db" 2>/dev/null || [[ $? == 1 ]]
mapfile -t targets < <(python3 - "$run/make.db" <<'PY'
import pathlib, sys
targets = set()
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if line.startswith('test-all:'):
        targets.update(x for x in line.split()[1:] if x.startswith('build-verify/'))
print('\n'.join(sorted(targets)))
PY
)
make -j"${J:-8}" VERIFY=1 "${targets[@]}" build-verify/test_spatial_augment >> "$run/build.log" 2>&1
make VERIFY=1 -n test-all > "$run/make.commands" 2>&1
python3 - "$run" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
commands = [s.strip() for s in (p / 'make.commands').read_text().splitlines()
            if s.startswith(('./', 'python3 ', 'UFSM'))]
assert commands, 'no test commands found'
(p / 'testcmds.txt').write_text('\n'.join(commands) + '\n')
PY
index=0
while IFS= read -r command; do
    index=$((index + 1))
    start=$(date +%s)
    rc=0
    # Release the cooperative GPU lock between tests so fit validation can run.
    flock "$HOME/ufsm-data/gpu.lock" bash -c "$command" > "$run/test-$index.log" 2>&1 || rc=$?
    printf '%s\t%s\t%s\t%s\n' "$index" "$rc" "$(( $(date +%s) - start ))" "$command" >> "$run/results.tsv"
    printf '%s %ss %s\n' "$rc" "$(( $(date +%s) - start ))" "$command" | tee -a "$run/testres.txt"
done < "$run/testcmds.txt"
python3 - "$run" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
known = {'./build-verify/test_sheet_gpu', 'python3 tests/test_sheet_cli.py',
         'python3 tests/test_pipeline_cli.py', 'python3 tests/test_cover_training.py',
         'python3 tests/test_production.py'}
rows = [s.split('\t', 3) for s in (p / 'results.tsv').read_text().splitlines()]
failures = {r[3] for r in rows if int(r[1])}
result = {'tests': len(rows), 'failures': sorted(failures),
          'unexpected_failures': sorted(failures - known),
          'unexpected_passes': sorted(known - failures)}
(p / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
(p / 'testres.txt').open('a').write('DONE\n')
sys.exit(bool(result['unexpected_failures'] or result['unexpected_passes']))
PY
