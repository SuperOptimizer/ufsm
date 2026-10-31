#!/usr/bin/env bash
# scripts/v2/inksrc.sh S: the ift3-24000 ink prediction the v2 pseudo ink of scroll S reads, as "src<TAB>key<TAB>model<TAB>mode<TAB>seam".
# Sources (checked 10-31; every lane that wrote them ran INKCK = ift3-24000: corpus common.sh / laptop.sh, pod unroll.sh / setup.sh):
#   corpus <S>/ink.zarr named "ink-blend" (state/ink.2pass) = 2 offset passes + ink-blend: seam-free (ink-seam --block 16 jump ~0.5);
#   corpus inks named "ufsm-recto" (0211, 1447) = ONE 640/64 pass: window seams every 512 voxels (jump 17-18) -> seam 512,0,SEAMW;
#   pod unroll inks (0800, 0268 cal) = predict --feather 32 single pass: seam-free; Paris 4 = ift3-24000-ov8 (8-pass blend): seam-free.
# seam: the ink-gate --seam value ("-" = none). Env: INKSRC_<S> (override path), SEAMW (6 ink voxels each side).
S=${1:?scroll}
declare -A POD=([PHercParis4]=/vesuvius/ufsm/paris4/ink/ift3-24000-ov8.zarr [PHerc0800]=/mnt/scratch/pods/PHerc0800/unroll/ink/ift3-24000.zarr
                [PHerc0268]=/mnt/scratch/pods/PHerc0268/cal/run/ink.zarr)
v=INKSRC_$S; src=${!v:-${POD[$S]:-}}
for c in /vesuvius/ufsm/corpus/$S/ink.zarr /mnt/scratch/vesuvius-offload/corpus/$S/ink.zarr; do [ -z "$src" ] && [ -e $c/zarr.json ] && src=$c; done
[ -n "$src" ] && [ -e $src/zarr.json ] || { printf -- '-\t-\t-\t-\t-\n'; exit 1; }
key=$(ls $src | grep -v '\.\(json\|txt\)$' | sort -g | head -1)
name=$(jq -r '.attributes.ome.multiscales[0].name // "?"' $src/zarr.json)
case "$src:$name" in
  *-ov8.zarr:*) mode=ov8-blend; seam=-;;
  /mnt/scratch/pods/*|/vesuvius/ufsm/pods/*) mode=feather32; seam=-;;
  *:ink-blend) mode=2pass-blend; seam=-;;
  *) mode=single-pass; seam=512,0,${SEAMW:-6};;
esac
printf '%s\t%s\tift3-24000\t%s\t%s\n' $src $key $mode $seam
