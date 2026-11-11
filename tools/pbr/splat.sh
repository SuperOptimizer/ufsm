#!/bin/bash
# splat.sh: the human Paris 4 ink labels (HF ink/phercparis4, segments on the 2.4 um scan) splatted into one sparse 3D label zarr
# (prob254, band -4..4 voxels around each label surface, 2.4 um level 0) over the label windows that reach the sheet-pbr crops
cd /vesuvius/ufsm/pbr; U=/vesuvius/ufsm/pbr/ufsm.pbr3
grep 'label window' hlab_windows.txt | while read w nm _ nv _ _ _ c0 _ r0 _ ww _ hh lab msk; do
  echo "== $w $nm $c0,$r0,$ww,$hh"; t0=$(date +%s)
  ~/bin/memcap -m 24G nice $U ink-splat hlab --surface hfink/$w --labels hfink/$w/$lab --mask hfink/$w/$msk --ct https://dl.ash2txt.org/community-uploads/forrest/volcomp \
    PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr --ct-um 2.4 --level 0 --iso 1 --band -4,4 --crop $c0,$r0,$ww,$hh --merge --coarse 0 \
    --name ${w}_$nm --threads 8 --cache /vesuvius/ufsm/cache 2>&1 | tail -2 | cut -c1-250
  echo "   $(( $(date +%s) - t0 )) s"
done; echo splat done
