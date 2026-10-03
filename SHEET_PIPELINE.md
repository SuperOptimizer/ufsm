# Paris4 continuous winding experiment

The native backbone stays at widths `16,32,64,80` (1,172,050 parameters).
The two existing outputs become surface confidence and an **unsigmoided winding
residual in turns**. `q = q0 + residual`; the gauge of q is arbitrary. No integer
wrap IDs or recto/verso classification are added. This is an experimental recipe;
the current production run continues with its frozen binary and labels.

## Data and geometry

Install the optional Python tools with `pip install -r requirements-sheet.txt`.
Build native code with `make`. All data, checkpoints and exported bundles belong
outside Git. The following paths match the current machine; change them for a
different installation.

```sh
python3 tools/build_surface_store.py \
  --segments-file configs/paris4-segments-20260623.txt \
  --download-cache /vesuvius/ufsm/gt/paris4-all-surfaces \
  --work /vesuvius/ufsm/gt/paris4-sheet-20261003 \
  --out /vesuvius/ufsm/gt/paris4-sheet-20261003/labels.zarr \
  --sources-out /vesuvius/ufsm/gt/paris4-sheet-20261003/sources.json \
  --band-chamfer 0 --level 1 --threads 8
python3 tools/prepare_surface_training.py \
  --work /vesuvius/ufsm/gt/paris4-sheet-20261003 \
  --out /vesuvius/ufsm/gt/paris4-sheet-20261003/training-4p8.zarr \
  --sources-out /vesuvius/ufsm/gt/paris4-sheet-20261003/training-sources.json \
  --source-template /vesuvius/ufsm/gt/paris4-sheet-20261003/sources.json
python3 tools/build_sheet_geometry.py \
  --meshes-root /vesuvius/ufsm/gt/paris4-all-surfaces/aws \
  --segments configs/paris4-segments-20260623.txt \
  --axis /vesuvius/usrm/umbilicus/PHercParis4/umbilicus-full-resolution.json \
  --splits configs/paris4-sheet-splits.json \
  --out /vesuvius/ufsm/gt/paris4-sheet-20261003/geometry
python3 tools/sheet_pipeline.py audit \
  --geometry /vesuvius/ufsm/gt/paris4-sheet-20261003/geometry/geometry.json \
  --out /tmp/paris4-winding-audit.png
```

The binary mask is an unexpanded surface at 4.8 µm, upsampled at native training
resolution. Runtime soft targets use sigma 2 native voxels. That is a conservative
uniform cap, bounded by a quarter of every audited resolvable gap; it is not yet
a spatially varying width field. Verified gaps narrower than 8 native voxels get
8-voxel contact exclusions. Gap negatives require clearance from sampled mesh
geometry and are discarded at sampling time if the dense surface target is
positive there. Unlabelled physical sheets remain a limitation of any binary
union of the released surfaces.

Mesh grid adjacency supplies normals, angular lifting and continuity. Invalid
vertices, long edges and inconsistent cycles stay excluded. Integer gauge
alignment uses consensus among nearby outward-normal matches and rejects
contradictory anchored components; disconnected components receive no global
supervision. Segment names select revisions only, never winding labels. Matches
use the nearest registered triangle intersected by a calibrated mesh normal;
ambiguous first hits are rejected rather than skipped. Normal signs use known
within-mesh winding adjacency, and the widest connected lift establishes the
global direction. Review `audit.json` and the preview before accepting a dataset.
The coarse radial prior preserves a positive global pitch when a folded axial
region has an invalid local fit. Every such fallback is recorded; mesh winding
targets remain unchanged.

Sparse, spatially indexed records contain coordinates, continuity pairs, ordering
pairs, 32-point paths and gap negatives. Path points follow the original mesh grid
at up to two native voxels between samples. Evaluation keeps the original grid
density independently of training decimation. No dense floating teacher is
written. Record/index/reference/contact/audit hashes are bound in `geometry.json`.

Development and final test occupy separate parts of the existing withheld Paris4
cube, with a 704-voxel training guard. Use a donor whose training withheld that
original cube. Full-mesh lifting supplies evaluation correspondence only; withheld
geometry does not fit the training reference or determine training gauges.

## Training and matched ablations

The FP16 stem receives CT, normalized q0 and two radial direction channels. The
quantized body keeps the FP4 policy. Geometry reductions use FP64 on the CPU;
logits and exported q are FP32, and activation gradients use the 16-bit storage
contract. `--mem auto16` chooses a compatible memory mode. Large-window capacity
and throughput need fresh measurements for this task; the old 704-cube FP8-gradient
fit does not establish that this new precision contract fits at 704.

A legacy warm-start zeroes the new input weights, residual-head weights/bias and
their optimizer/EMA state. Surface weights remain reusable. Baseline controls use
the same FP16 input contract. The loss is BCE + 0.5 Dice plus normalized Huber
coordinate, continuity and ordering losses (0.25 each), path support and gap BCE
(0.1 each), ramped over 500 updates. Per-term losses go to `geometry.csv`.

```sh
python3 tools/sheet_pipeline.py sweep \
  --geometry /vesuvius/ufsm/gt/paris4-sheet-20261003/geometry/geometry.json \
  --sources /vesuvius/ufsm/gt/paris4-sheet-20261003/training-sources.json \
  --original-sources /vesuvius/ufsm/gt/paris4-selected-20260623/training-sources.json \
  --cover /vesuvius/ufsm/gt/paris4-selected-20260623/cover704.json \
  --resume /path/to/donor.ckpt --updates 2000 --out runs/sheet-ablation
# These explicitly start GPU work; run sequentially when the GPUs are available.
python3 tools/sheet_pipeline.py resume --run runs/sheet-ablation/expanded
python3 tools/sheet_pipeline.py resume --run runs/sheet-ablation/thin
python3 tools/sheet_pipeline.py resume --run runs/sheet-ablation/winding
python3 tools/sheet_pipeline.py resume --run runs/sheet-ablation/full
```

`sweep` prepares four candidates with the same donor, ordered safe tiles and update
budget: existing expanded targets, thin targets, thin + winding/ordering, and all
losses. It starts no training. The existing GPU leases prevent collisions with
production. `resume` starts a prepared candidate or resumes its committed cursor;
it checks frozen inputs and external store metadata. UFSM environment overrides
are cleared so they cannot silently change the recipe. External CT and label
payloads must remain immutable; the runner binds metadata/provenance without
hashing every CT shard.

Native use is also available: `ufsm train SOURCES --geometry geometry.json
--sheet-init 1 --resume DONOR ...`. A winding resume omits `--sheet-init`; changes
to geometry, reference, loss variant or ramp origin are rejected. B=1, native level
0, one GPU or a spatial Z split, and Z-preserving symmetries are supported.

## Prediction, reconstruction and acceptance

```sh
python3 tools/sheet_pipeline.py predict --run runs/sheet-ablation/full \
  --box 48128,17408,16384,384,1024,1024 --out runs/sheet-dev --gpu 0
python3 tools/sheet_pipeline.py extract --prediction runs/sheet-dev \
  --out runs/sheet-dev-evidence.npz --threshold 0.3
python3 tools/sheet_pipeline.py evaluate \
  --truth /vesuvius/ufsm/gt/paris4-sheet-20261003/geometry/evaluation-mesh.npz \
  --evidence runs/sheet-dev-evidence.npz --split development --out runs/sheet-dev.json
```

Prediction writes the usual uint8 probability store plus compressed FP32 q shards.
NaN marks air or surface probability below 0.1. A reference hash is mandatory for
winding models. Export uses one GPU per store currently. Evidence extraction
measures probability along nearby point connections, including across shard
boundaries; the surface-only controls use these observed affinities, while winding
models also constrain their coordinate difference. Both use the same threshold.
Track edges require these measured affinities; supported endpoints alone do not
count as continuous predictions. Reconstructed observed coverage additionally
requires nearby original network evidence, so inferred completion cannot inflate it.

Evaluate all four on development, then call `sheet_pipeline.py select --baseline
BASELINE.json --candidates CANDIDATE.json ... --out SELECTION.json`. Acceptance is
at least 30% fewer switches and false bridges, 25% longer median correct tracks,
and at most 5% supported-coverage loss. If the baseline exposes no failures,
prepare a fresh matched 5,000-update sweep. Track length is a two-sweep geodesic
diameter lower bound, and the connectivity metrics are local affinity measures;
they do not certify a whole-scroll tracing solution.

```sh
python3 tools/sheet_reconstruct.py --evidence runs/sheet-dev-evidence.npz \
  --reference /vesuvius/ufsm/gt/paris4-sheet-20261003/geometry/reference.json \
  --out runs/sheet-dev-mesh --device cpu
```

Reconstruction fits a canonical spiral through a bounded injective ambient flow.
The initializer lowers its radial offsets when needed to keep the entire selected
winding domain away from the axis; its offset cap is recorded. This leaves the
model reference and observed winding labels unchanged.
Each Euler map has a Lipschitz displacement bound below one; the exported triangle
mesh additionally passes intersection, edge and vertex manifold checks. Local
stretch and reliable normal agreement are regularized. OBJ/NPZ provide q/Z UVs;
these are material-coordinate initialization, not a metric flattening. Support
labels distinguish observed, uncertain and inferred completion. A valid manifold
can still follow the wrong sheet: geometric validity is not an accuracy claim.
The default deformation grids (8,16,32) are a coarse initial fit. For finer geometry
use `--grids 8 16 32 64 128` or higher and inspect `field_spacing_native` in the
report. Whole-scroll reconstruction needs a measured resolution/memory budget;
the coarse default does not establish voxel-level accuracy over the entire scan.

Evaluate the selected candidate once on final test and reconstruct that prediction.
Pass `--reconstruction MESH_DIRECTORY` to the final `evaluate` call for certified
nearest-triangle winding correspondence. `export --run RUN --report TEST.json
--reconstruction-report MESH/report.json --selection SELECTION.json --out BUNDLE`
requires matching checkpoint/reference/evidence hashes and this mesh evaluation.
The portable bundle contains code, model, reference, axis and reports. Only after
the development and final-test review should a new full coverage pass replace the
current production recipe.

## Verification

`make test-sheet` checks lifting, exclusions, loss derivatives, reconstruction and
file contracts on CPU. `make test-sheet-gpu` checks interpolation/scatter across
both GPUs plus small real training, resume and floating-output prediction. The
CUDA integration fixture uses 32-cubes and widths 8,8, so it verifies behavior,
not production capacity, accuracy or speed. Existing raster, sampler and production
runner regression tests remain applicable.

## Local qualification (2026-10-03)

The complete 25-mesh build is published under
`/vesuvius/ufsm/gt/paris4-sheet-20261003`. All mesh unwrap cycle checks passed.
After contact exclusions, 3,977,322 points carry coordinate supervision. The
dataset contains 14,095,914 sparse records, including 111,971 ordering pairs,
and 663 contact exclusions. Its development/test artifacts contain 19,178 / 18,151
mesh points. Native loading, file hashes and sparse sampling passed on this dataset.
The winding audit preview is `/tmp/paris4-winding-audit.png`.

`runs/paris4-sheet-ablation-20261003` contains four prepared 2,000-update controls
with one frozen donor and identical safe coverage order. They have not started
GPU training. The existing selected-surface production job still owns both GPUs.
CPU and small single-/split-GPU integration tests pass; the complete GPU regression
suite and the new task's large-window capacity, speed and quality gates remain
to be measured when the production GPUs are available.
