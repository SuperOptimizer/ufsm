#!/bin/bash
# run.sh SEG U0 V0 N OUT [extra flags]: ufsm sheet-pbr on a Paris 4 segment crop (2.4 um level 0 from the volcomp mirror)
S=$1; shift; u=$1; v=$2; n=$3; o=$4; shift 4
exec ~/bin/memcap -m 20G nice /vesuvius/ufsm/pbr/ufsm.pbr2 sheet-pbr /vesuvius/ufsm/pbr/seg/$S/mesh/tifxyz $o --ct https://dl.ash2txt.org/community-uploads/forrest/volcomp \
  PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0 --threads 8 --cache /vesuvius/ufsm/cache --crop $u,$v,$n "$@"
