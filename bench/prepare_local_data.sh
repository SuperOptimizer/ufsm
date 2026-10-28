#!/usr/bin/env bash
# A private mirror of the existing Paris 4 box. Never fill/overwrite the shared CT.
# Level 5 is absent from the local box but used by level-2 sampler occupancy probes.
set -euo pipefail
out=${1:-$HOME/ufsm-data/perf-int}
source_box=$HOME/ufsm-data/fitkit/p4box/ct
group=PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr
remote=https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
mkdir -p "$out"
out=$(realpath "$out")
if [[ ! -d $out/ct ]]; then
    mkdir -p "$out/ct"
    cp -as "$source_box/." "$out/ct/"
fi
base="$out/ct/$group/5"
if [[ -L $base/zarr.json || ! -f $base/zarr.json ]]; then
    curl --fail --silent --show-error --retry 2 --max-time 60 "$remote/5/zarr.json" -o "$base/zarr.download.json"
    mv "$base/zarr.download.json" "$base/zarr.json"
fi
mkdir -p "$base/c/1/0"
if [[ ! -s $base/c/1/0/0 ]]; then
    curl --fail --silent --show-error --retry 2 --max-time 180 "$remote/5/c/1/0/0" -o "$base/c/1/0/0.download"
    mv "$base/c/1/0/0.download" "$base/c/1/0/0"
fi
python3 - "$HOME/ufsm-data/w24/sources-box.json" "$out" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[2])
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
for source in d['sources']:
    source['root'] = str(out / 'ct')
(out / 'sources-box.json').write_text(json.dumps(d, indent=2) + '\n')
PY
