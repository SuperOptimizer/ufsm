#!/usr/bin/env bash
# Execute under ~/ufsm-data/capped.sh. Every case uses the real train/predict CLI.
# NETS='wind96 wind2-7level' WINDOWS='512x512 640x640 768x768 256x768 256x1024 192x1024'
# SCALES='9.6 2.4' POLICIES='auto fast' WORKERS=6 SLOTS=13 STEPS=40 GPU=0
# SOURCES_96=... SOURCES_24=... override the local sources; CKPT=... for inference.
# First run bench/prepare_local_data.sh. Defaults replay one real CT box (hot-cache
# upper bound); set SAMPLE_AT_96='' SAMPLE_AT_24='' with complete corpus sources
# to measure random-sampling throughput. Fixed boxes are performance fixtures only.
# PRINT_ONLY=1 writes exact commands without running them. No build or desk launch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/ufsm-venv/bin:/usr/local/cuda/bin:$PATH"
mode=${1:-dry}
case "$mode" in dry|train|infer) ;; *) echo 'usage: bench/perf_baseline.sh dry|train|infer [disk output directory]' >&2; exit 2;; esac
out=${2:-$HOME/ufsm-data/perf-int/baseline-$mode-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$out/tmp"
out=$(realpath "$out")
export TMPDIR="$out/tmp"
git rev-parse HEAD > "$out/head.txt"
git diff HEAD --binary > "$out/change.patch"
uname -a > "$out/system.txt"
[[ ${PRINT_ONLY:-0} == 1 ]] || nvidia-smi > "$out/gpu.txt"
binary=${UFSM_BIN:-$PWD/build/ufsm}
run_case() {
    local tag=$1; shift
    local case_dir="$out/$tag"
    mkdir -p "$case_dir"
    printf '%q ' "$@" > "$case_dir/command.sh"
    printf '\n' >> "$case_dir/command.sh"
    printf '%s\n' "$tag"
    [[ ${PRINT_ONLY:-0} == 1 ]] && return
    local rc=0
    (
        flock 9
        nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv > "$case_dir/gpu-before.csv"
        # Board memory is sampled: the result includes any pre-existing contexts.
        nvidia-smi -i "${GPU:-0}" --query-gpu=timestamp,memory.used --format=csv,noheader,nounits \
            --loop-ms=100 > "$case_dir/gpu-sampled.csv" &
        local monitor=$!
        trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
        /usr/bin/time -v -o "$case_dir/time.txt" timeout --signal=TERM --kill-after=20s "${CASE_TIMEOUT:-1800}" \
            "$@" > "$case_dir/stdout.log" 2> "$case_dir/stderr.log"
    ) 9>"$HOME/ufsm-data/gpu.lock" || rc=$?
    printf '%s\n' "$rc" > "$case_dir/exit.txt"
    python3 bench/perf_summary.py "$out"
}
if [[ $mode == infer ]]; then
    : "${CKPT:=$HOME/ufsm-data/fitkit/ckpt/wind2-ft3-6000.ckpt}"
    : "${CT_ROOT:=$HOME/ufsm-data/fitkit/p4box/ct}"
    : "${CT_GROUP:=PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr}"
    # Coordinates are selected-level CT voxels: a fixed 512^3 box at 9.6 um.
    for window in ${INFER_WINDOWS:-512x512 384x640 640x640}; do
        z=${window%x*}; xy=${window#*x}
        hz=$((z / 8)); halo=$((xy / 8))
        tag="infer-$window"
        run_case "$tag" env UFSM_MEM_REPORT=1 UFSM_PRED_PROF=1 \
            "$binary" predict "$CKPT" "$CT_ROOT" "$CT_GROUP" "$out/$tag/pred" \
            --fit-zarr "$out/$tag/fields.zarr" --fit-h 2 --um 2.4 --level 2 --levels 1 \
            --box "${INFER_BOX:-12288,4096,3584,512,512,512}" \
            --axis "${AXIS:-$HOME/ufsm-data/fitkit/p4box/axis-0.json}" --gpus "${GPU:-0}" \
            --window "$xy" --window-z "$z" --halo "$halo" --halo-z "$hz" --shard 512 \
            --threads "${READ_THREADS:-6}" --cache "${CACHE:-$HOME/ufsm-data/cache}"
    done
else
    for scale in ${SCALES:-9.6 2.4}; do
        case "$scale" in
            9.6) source_cfg=${SOURCES_96:-$HOME/ufsm-data/perf-int/sources-box.json}; levels=0,0,1; at=${SAMPLE_AT_96-12288,4096,3584};;
            2.4) source_cfg=${SOURCES_24:-$HOME/ufsm-data/perf-int/sources-box.json}; levels=1; at=${SAMPLE_AT_24-49152,16384,14336};;
            *) echo "unsupported physical scale: $scale" >&2; exit 2;;
        esac
        # The production unwrap2 loader carries band/phase labels, not unused recto/ink.
        if [[ ${TASK:-unwrap2} == unwrap2 ]]; then
            python3 - "$source_cfg" "$out/sources-$scale.json" "$scale" <<'PY'
import json, pathlib, sys
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
for source in d['sources']:
    source['targets'] = {k: v for k, v in source.get('targets', {}).items() if k == 'band'}
    # The local box fixture caps native-level tests at 0; benchmark one exact level.
    source['min_level'] = source['max_level'] = 2 if sys.argv[3] == '9.6' else 0
assert all('band' in s['targets'] for s in d['sources']), 'unwrap2 baseline requires band sources'
pathlib.Path(sys.argv[2]).write_text(json.dumps(d, indent=2) + '\n')
PY
            source_cfg="$out/sources-$scale.json"
        fi
        for net in ${NETS:-wind96 wind2-7level}; do
            case "$net" in
                wind96) widths=32,64,96,128,160,192; dec=0,32;
                    weights=(--resume "${TRAIN_CKPT_96:-$HOME/ufsm-data/fitkit/ckpt/wind2-ft3-6000.ckpt}" --finetune 1);;
                wind2-7level) widths=16,16,32,64,128,192,256; dec=0;
                    # This old checkpoint has a label-convention error. It is a
                    # numerical performance fixture, never an accuracy control.
                    weights=(--resume "${TRAIN_CKPT_7:-$HOME/ufsm-data/wind2/ck-16000.ckpt}" --finetune 1);;
                *) echo "unknown net: $net" >&2; exit 2;;
            esac
            for window in ${WINDOWS:-512x512 640x640 768x768 256x768 256x1024 192x1024}; do
                z=${window%x*}; xy=${window#*x}
                for policy in ${POLICIES:-auto fast}; do
                    tag="$mode-$net-$scale-$window-$policy"
                    dry=0; [[ $mode == dry ]] && dry=1
                    run_case "$tag" env -u UFSM_MEM_PICK UFSM_MEM_DRY="$dry" UFSM_MEM_REPORT=1 UFSM_SYNC_CHECK=1 UFSM_SAMPLE_AT="$at" \
                        "$binary" train "$source_cfg" --out "$out/$tag/run" --task "${TASK:-unwrap2}" \
                        "${weights[@]}" \
                        --widths "$widths" --dec-widths "$dec" --down-norm 1 --scale-cond 1 \
                        --P "$xy" --Pz "$z" --B 1 --gpus "${GPU:-0}" --mem "$policy" \
                        --fp4 2 --input-prec 8 --gpu-input 1 --pack-targets 1 --val-store disk \
                        --workers "${WORKERS:-6}" --sampler-slots "${SLOTS:-13}" --steps "${STEPS:-40}" \
                        --val-batches 0 --val-every 1000000 --ckpt-every 1000000 --log-every 5 \
                        --lr 0 --warmup 0 --opt muon --muon-lr 0 --gscale "${GSCALE:-1}" --seed 73 --det 1 \
                        --levels "$levels" --zfix 1
                done
            done
        done
    done
fi
[[ ${PRINT_ONLY:-0} == 1 ]] || python3 bench/perf_summary.py "$out"
