#!/bin/bash
# scripts/inkgt/splat_all.sh [SCROLL ...]: human ink labels -> one sparse prob254 3-D target per scan (2026-11-11, branch inkgt-20261111).
# segs.tsv (scroll name tifxyz labels mask ct-key um), winall.tsv (name c0 r0 w h sup: 16k label windows with supervision), holdboxes.tsv
# (scroll  z,y,x,nz,ny,nx (level-0 of the scan)  note): every window of every segment of SCROLL splatted with --face -6,0 (the label on the
# outermost ~14 um of papyrus at the sheet's open face on +N, the written recto face) at CT level 1 (4.8 um), --merge into
# $O/<scroll>.zarr, held-out boxes --exclude'd; then the known-max pyramid (--pyramid-only, levels 9.6 .. ).
set -u
D=${D:-/vesuvius/ufsm/inkgt}; O=${O:-$D/gt}; U=${U:-/home/forrest/ufsm-inkgt/build/ufsm}; M=https://dl.ash2txt.org/community-uploads/forrest/volcomp
T=${T:-6}; ISO=${ISO:-64}; LV=${LV:-1}; FACE=${FACE:--6,0}; mkdir -p $O/logs
SC=${*:-$(cut -f1 $D/segs.tsv | sort -u)}
for sc in $SC; do (
  ex=(); while IFS=$'\t' read -r s b n; do [ "$s" = "$sc" ] && ex+=(--exclude $b); done < $D/holdboxes.tsv
  out=$O/$sc.zarr; log=$O/logs/$sc.log; echo "== $sc: ${#ex[@]} exclude args $(date)" >> $log
  awk -F'\t' -v s=$sc '$1==s' $D/segs.tsv | while IFS=$'\t' read -r s nm tx lab msk key um; do
    awk -F'\t' -v n="$nm" '$1==n' $D/winall.tsv | while IFS=$'\t' read -r n c0 r0 w h sup; do
      tag="$nm@$c0,$r0"; grep -qF "DONE $tag" $log && continue
      ~/bin/memcap -m 14G nice -n 10 $U ink-splat $out --surface $tx --labels $lab --mask $msk --ct $M $key --ct-um $um --level $LV --iso $ISO --face $FACE \
        --crop $c0,$r0,$w,$h --merge --coarse 0 --name "${nm}_${c0}_${r0}" --threads $T --cache /vesuvius/ufsm/cache "${ex[@]}" >> $log 2>&1 \
        && echo "DONE $tag" >> $log || echo "FAIL $tag" >> $log
    done
  done
  ~/bin/memcap -m 14G nice -n 10 $U ink-splat $out --pyramid-only --surface x --labels x --ct $M $(awk -F'\t' -v s=$sc '$1==s {print $6; exit}' $D/segs.tsv) \
     --ct-um $(awk -F'\t' -v s=$sc '$1==s {print $7; exit}' $D/segs.tsv) --level $LV --threads $T --cache /vesuvius/ufsm/cache >> $log 2>&1 && echo "PYRAMID $sc" >> $log
  echo "== $sc done $(date)" >> $log ) &
done; wait
