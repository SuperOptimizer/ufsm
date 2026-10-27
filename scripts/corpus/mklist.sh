#!/usr/bin/env bash
# mklist.sh > list.tsv: the corpus order for corpus.sh: one scan per scroll at ~8.6-9.6 um (the 2025/2026 ESRF 8.64 / 9.36 um scan where it
# exists, else the 2.4 um scan at level 2), from the desk's u0 copies (u0/final.tsv). 1667 first (eval boxes), then scrollprize.org/prizes
# order, the 2.4-um-only scrolls last (Paris 3: 102 GB, last). Paris 4 is done (p4u/run3-cf). Fragments (0343P, 0500P2, MANBp) skipped.
# axis: an umbilicus in the scan's own level-0 frame (only 1667, 0332, MANB; the desk umbilici of 0139 0175A 0343 0813 1447 are in a
# 2.4 um frame: never use them) or "auto" (ufsm axis: per-z centroid of the masked CT).
V=/vesuvius/ufsm/u0/vol; UM=/vesuvius/ufsm/umbilicus
row() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6"; }
printf 'scroll\troot\tkey\tum\tlevel\taxis\n'
row PHerc1667 /vesuvius/ufsm/volcomp PHerc1667/20251217075048-2.399um-0.2m-78keV-masked.zarr 2.399 2 $UM/PHerc1667/umbilicus-full-resolution.json
for s in 1447:20250521151220-8.640 0125:20250821151825-9.362 0191:20250821151635-9.362 0211:20250821151803-9.362 0257:20250821151750-9.362 \
         0268:20251110183117-8.640 0358:20250821151737-9.362 0800:20250521135224-8.640 0813:20250821151723-9.362 0826:20250821151701-9.362 \
         1203:20250820131727-9.362 1218:20250521120456-8.640 1545:20250821151648-9.362 0175A:20250521115057-8.640 0175B:20250521125822-8.640 \
         0306B:20250521133212-8.640 0483A:20250521140913-8.640 0483B:20251124083638-8.640 0490A:20250521151210-8.640 0490B:20250521151215-8.640 \
         0846A:20250728152254-9.362 0846B:20250804142305-9.362 0139:20250728140407-9.362 0814:20250804134230-9.362 0841:20250821151531-9.366 \
         1451:20250521151225-8.640 0343:20250521140437-8.640 0009B:20250521125136-8.640; do
  sc=${s%%:*}; id=${s#*:}; um=${id#*-}; kev=113keV; [ "$um" = 8.640 ] && kev=116keV
  row PHerc$sc $V "PHerc$sc/${id}um-1.2m-$kev-masked.zarr" $um 0 auto
done
row PHerc0332 $V PHerc0332/20251211183505-2.399um-0.2m-78keV-masked.zarr 2.399 2 $UM/PHerc0332/umbilicus-full-resolution.json
row PHercMANB $V PHercMANB/20260323091048-2.399um-0.2m-78keV-masked.zarr 2.399 2 $UM/PHercMANB/umbilicus-full-resolution.json
row PHercMAN5 $V PHercMAN5/20260311104824-2.399um-0.2m-78keV-masked.zarr 2.399 2 auto
row PHerc1299 $V PHerc1299/20260309130042-2.399um-0.2m-78keV-masked.zarr 2.399 2 auto
row PHercParis3 $V PHercParis3/20260427095331-2.400um-0.2m-78keV-masked.zarr 2.400 2 auto
