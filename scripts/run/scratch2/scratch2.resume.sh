#!/usr/bin/env bash
# scratch2.resume.sh [NEWBIN] (since 2026-11-14 ops audit #5 / #6: a wrapper): s2ctl.sh resume, which takes scratch2.ctl.lock and launches
# with exactly scratch2/launch.env. Set S2CTL_OWNER=<agent>. The pre-audit body is in scratch2.resume.sh.bak5-20261114.
exec env S2CTL_OWNER="${S2CTL_OWNER:-resume.sh}" bash /vesuvius/ufsm/v2/lat/runs/s2ctl.sh resume "${1:-}" "via scratch2.resume.sh"
