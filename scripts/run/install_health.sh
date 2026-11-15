#!/usr/bin/env bash
# scripts/run/install_health.sh RUN [ROOT]: install / update and (re)start the health watcher of training run RUN as the systemd --user
# service ufsm-health@RUN (Restart=always; survives logouts with linger). The script is copied to ~/.local/share/ufsm/health.py, so the
# service does not depend on a worktree. Status: systemctl --user status ufsm-health@RUN; alerts in ROOT/RUN.alerts, state in
# ROOT/RUN.health.state. Stop: systemctl --user disable --now ufsm-health@RUN.
set -eu
RUN=${1:?RUN}; ROOT=${2:-/vesuvius/ufsm/v2/lat/runs}; H=$(cd "$(dirname "$0")" && pwd)
mkdir -p ~/.local/share/ufsm ~/.config/systemd/user ~/.config/ufsm
install -m 755 "$H/health.py" ~/.local/share/ufsm/health.py
install -m 644 "$H/ufsm-health@.service" ~/.config/systemd/user/ufsm-health@.service
[ -f ~/.config/ufsm/health-$RUN.env ] || printf 'UFSM_RUN_ROOT=%s\nHEALTH_ARGS=--root %s\n' "$ROOT" "$ROOT" > ~/.config/ufsm/health-$RUN.env
systemctl --user daemon-reload
systemctl --user enable "ufsm-health@$RUN" > /dev/null 2>&1
systemctl --user restart "ufsm-health@$RUN"
sleep 3; systemctl --user --no-pager status "ufsm-health@$RUN" | head -5
loginctl show-user "$USER" -p Linger 2> /dev/null | grep -q yes || echo "note: linger is off: the service stops at logout (loginctl enable-linger $USER, needs root)"
