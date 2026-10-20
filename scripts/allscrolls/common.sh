# scripts/allscrolls/common.sh: shared settings of the all-scrolls job (sourced by fit.sh, ink.sh, render.sh, queue.sh).
# Scope (user, 2026-10-20): the 2025/2026 ESRF 8.64 um (116 keV) / 9.36 um (113 keV) scan of every mirror scroll that has one, its published prod96 fit
# fields, 3D ink with ink96-wk1 at the scan's native spacing. Layout (A = the job root on this host):
#   A/scrolls.tsv                    scroll, scan, um, fields zarr name (queue.sh table)
#   A/bin/ufsm                       the pinned binary (A/bin/ufsm.txt: commit)
#   A/<S>/fields/<zarr>              local copy of the published fit fields (laptop)
#   A/<S>/boxes/<b>/                 box fits (fit.sh); A/<S>/stitch, A/<S>/sheets (stitched sheets)
#   A/<S>/surf/                      the surfaces handed to the desk: <b>/tifxyz (per box), stitched/ (sheets), list.tsv
#   A/<S>/ink/<scan>-ink96.zarr      3D ink (desk); A/<S>/renders (JPEG panels + index.tsv)
#   A/<S>/state/                     <step>.done / .failed markers, logs
set -u
R=https://dl.ash2txt.org/community-uploads/forrest/volcomp
case "$(hostname)" in forlindesk2*) HOST=desk; A=${ALLS:-/vesuvius/ufsm/allscrolls};; *) HOST=laptop; A=${ALLS:-$HOME/ufsm-data/allscrolls};; esac
U=${U:-$A/bin/ufsm}
DESK=${DESK:-forlindesk2}; DA=/vesuvius/ufsm/allscrolls   # the desk's job root (seen from the laptop)
# scrollprize.org/prizes order (user 10-20): 2027 Grand Prize volumes (PHerc1447 first: published segments), then the First Letters ones, then the rest
FIRST="PHerc1447 PHerc0125 PHerc0191 PHerc0211 PHerc0257 PHerc0268 PHerc0358 PHerc0800 PHerc0813 PHerc0826 PHerc1203 PHerc1218 PHerc1545 PHerc0175A PHerc0175B PHerc0306B PHerc0483A PHerc0483B PHerc0490A PHerc0490B PHerc0846A PHerc0846B"
# C CMD...: one job under the laptop's memory slice (MEM <= 6G per job), plain nice on the desk
C() { if [ $HOST = laptop ]; then MEM=${MEM:-6G} $HOME/ufsm-data/capped.sh "$@"; else nice -n 5 "$@"; fi; }
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $A/queue.log >&2; }
# scan S: "scan um fields" of scroll S from scrolls.tsv
scan_of() { awk -v s=$1 '$1 == s {print $2, $3, $4}' $A/scrolls.tsv; }
done_() { [ -e $A/$1/state/$2.done ]; }
mark() { mkdir -p $A/$1/state; echo "$(date '+%F %T') ${3:-}" > $A/$1/state/$2.done; }
