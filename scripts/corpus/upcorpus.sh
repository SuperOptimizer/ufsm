#!/bin/bash
# upcorpus.sh S...: upload a non-prize scroll's corpus products to dl.ash2txt volcomp/<S>/representations/ufsm-c2f-20261028/
# (fields, ref, joined, wraps, ink.zarr, axis/box): whatever exists locally. Holds the desk push lock so the fields are not
# stripped (after their push to forlindesk) mid-upload. Re-running resumes (mirror --continue skips finished files).
C=/vesuvius/ufsm/corpus; R=ufsm-c2f-20261028; cd ~/reg-upload || exit 1
for S in "$@"; do
  O=$C/$S; [ -d $O ] || continue
  (
    flock 9
    L=up-$S.lftp
    {
      echo "mkdir -p volcomp/$S/representations/$R"
      echo "cd volcomp/$S/representations/$R"
      for f in axis.json box.txt; do [ -e $O/$f ] && echo "put $O/$f"; done
      for d in fields ref joined wraps; do [ -d $O/$d ] && echo "mirror -R --continue --parallel=4 --no-perms $O/$d/ $d"; done
      [ -d $O/ink.zarr ] && [ -e $O/state/ink.ready ] && echo "mirror -R --continue --parallel=4 --no-perms $O/ink.zarr/ ink.zarr"
      echo "du -s ."
    } > $L
    echo "$(date '+%F %T') $S start"
    bash up.sh < $L 2>&1 | tail -3
    echo "$(date '+%F %T') $S done"
  ) 9> $O/state/push.claim
done
echo ALLDONE
