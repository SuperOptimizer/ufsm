#!/bin/sh
# Export every label zarr in the HF labels.zip to volcomp pyramids under /vesuvius/ufsm/gt/labels/<name>/.
# Voxel sizes from the matching open-data volumes (the three w0xx zarrs have no identified volume yet:
# exported at 2.4 um as a placeholder, flagged in the log).
set -u
B=/home/forrest/ufsm/build/ufsm
ZIP=${1:-/vesuvius/ufsm/hf/labels.zip}
OUT=${2:-/vesuvius/ufsm/gt/labels}
T=${THREADS:-32}
run() { # name um
  d="$OUT/$1"
  if [ -f "$d/zarr.json" ]; then echo "skip $1 (done)"; return; fi
  echo "== $1 @ $2 um"
  $B ingest-zip "$ZIP" "$1" "$d" --um "$2" --levels 6 --threads "$T" || echo "FAILED $1"
}
run PHercMANBp-ct-2um_surface.zarr 2.399
run 0500p2_5217.zarr 2.215
run 2.215um_0.4m_111keV_PHerc0343P_surface.zarr 2.215
run SCROLLS_HEL_2.399um_78keV_0.22m_PHerc_1667_TA_0001_masked_surface.zarr 2.399
run s1_2.4um_gp.zarr 2.4
run w023-w032.zarr 2.4
run w037-w041.zarr 2.4
run w047-w059.zarr 2.4
echo "labels done"
