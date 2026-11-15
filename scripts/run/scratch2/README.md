# scratch2 run control (merger2 2026-11-14, ops audit #4 / #5 / #6, eval audit #6)

Live copies are in /vesuvius/ufsm/v2/lat/runs (gswap.sh in /vesuvius/ufsm/merge/swap2); these are the reviewed sources. Deploy by
copying a NEW file and `mv` it over the live name (a running bash script must never be edited in place: bash reads it incrementally).

- `s2ctl.sh {status|stop|resume|swap|hold|drystart|replay|fresh|sync}`: the one control path. Run it with S2CTL_OWNER=<agent>.
  - flock -n scratch2.ctl.lock; a second actor is refused with the holder's name.
  - The stop file (owner / reason) is removed by an EXIT trap.
  - scratch2/launch.env is the only config: every KEY= line is exported to scratch2.launch, and gates/settings + gates/binary are generated.
  - resume dry-starts a changed launch.env (2 windows from a copy of last.ckpt) and falls back to launch.env.good if that fails.
  - replay OLD NEW [N]: OLD, OLD, NEW at lr 0 with --det 1 from last.ckpt; PASS if max rel loss diff <= max(10 x noise, 1e-3).
  - fresh NEWENV ARCHIVE: dry start from scratch, archive the old run, launch.
- `scratch2.launch`: the recipe (ARCH / EXTRA). It runs `scratch2.train.sh`, which logs each trainer's UFSM_* env + argv to resume.log,
  writes `.relaunch` as `s2ctl.sh resume`, and implements DRY / FRESH / DRYSTEPS.
- `scratch2.resume.sh`: a wrapper around `s2ctl.sh resume`.
- `gswap.sh SHA`: binary-only gated swap; 12-line per-process medians of acc / gn; auto-rollback.
- `try7.launch.env`: the try7 config (U is filled in at launch).
