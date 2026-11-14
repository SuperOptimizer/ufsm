#!/usr/bin/env python3
# scripts/run/health.py RUN [--root R] [--every S] [--once]: the health watcher of a training run (2026-11-14, ops audit #3; from
# univA's scratch2.health.py). Reads only: R/RUN.log, R/RUN.resume.log, R/RUN.pid, the trainer's cgroup, disks. Writes R/RUN.alerts
# (one line per alert, the same key at most once an hour: the lead's monitor tails it) and R/RUN.health.state (a status line per check).
# Never signals the trainer. Run it as the systemd --user service ufsm-health@RUN (scripts/run/install_health.sh), so it survives
# logouts and restarts itself; the run's files are looked up by name each check, so tryN renames of old files do not blind it.
# Checks:
#   - NaN / inf in step lines; loader wait > 20 % on the last 3 log lines; no new step for 30 min while the trainer lives;
#   - trainer exits (resume.log "exit rc=", not during an RUN.evpause), STOP file / MemAvailable lines;
#   - per-head losses after step 2000: a head's mean over the last 5 logs > 2x the 20 before; a head at 0 for 10 logs (no windows);
#   - memory: limits from the trainer's own cgroup (memory.max of its scope and of the slice above it, not constants): alert when the
#     slice's unreclaimable memory (anon + shmem) passes 90 % of its max, and on every new oom_kill in the slice's memory.events;
#   - disks: /, the run root and the derived cache ($UFSM_DERIVED_CACHE or ~/.cache/ufsm/derived) under 10 % free (or 50 GB), with
#     the growth rate and hours to full when it is shrinking;
#   - a status line every 1000 steps.
import os, re, sys, time, shutil, collections, statistics, argparse

ap = argparse.ArgumentParser()
ap.add_argument('run'); ap.add_argument('--root', default=os.environ.get('UFSM_RUN_ROOT', '/vesuvius/ufsm/v2/lat/runs'))
ap.add_argument('--every', type=float, default=120); ap.add_argument('--once', action='store_true')
ap.add_argument('--disks', default='', help='more paths whose filesystems to watch (comma list)')
A = ap.parse_args()
o = os.path.join(A.root, A.run); LOG = o + '.log'; RES = o + '.resume.log'; AL = o + '.alerts'; ST = o + '.health.state'
HEADS = [('phase', r'phase bce ([0-9.naif]+)'), ('normal', r'normal bce ([0-9.naif]+)'), ('wind4', r'wind4 bce ([0-9.naif]+)'),
         ('rv', r'rv bce ([0-9.naif]+)'), ('ink', r'ink bce ([0-9.naif]+)'), ('fib', r'fib bce ([0-9.naif]+)'),
         ('fibdir', r'fib bce [0-9.naif]+ dir ([0-9.naif]+)'), ('loss', r'loss ([0-9.naif]+)')]
last_alert = {}; hist = collections.defaultdict(list); pos = {}; last_step = 0; last_step_t = time.time(); waits = []
oomk = None; disk_hist = collections.defaultdict(list); quiet = True   # (the first pass replays the logs for the head history: no alerts)

def alert(key, msg, force=False):
    if quiet and not force: return
    if time.time() - last_alert.get(key, 0) < 3600: return
    last_alert[key] = time.time()
    with open(AL, 'a') as f: f.write(time.strftime('%F %T ') + msg + '\n')

def read_new(path):   # the bytes appended since the last call (a file replaced / truncated starts over)
    try: st = os.stat(path)
    except OSError: return ''
    p, ino = pos.get(path, (0, None))
    if ino != st.st_ino or st.st_size < p: p = 0
    with open(path, errors='replace') as f: f.seek(p); new = f.read(); pos[path] = (f.tell(), st.st_ino)
    return new

def pid():
    try: return int(open(o + '.pid').read().strip())
    except Exception: return None

def alive():
    p = pid(); return bool(p) and os.path.exists('/proc/%d' % p)

def rss_gb():
    p = pid()
    try:
        for l in open('/proc/%d/status' % p):
            if l.startswith('VmRSS'): return int(l.split()[1]) / 1048576
    except Exception: pass
    return 0.

def cg_dirs():   # the trainer's cgroup and its parents up to the first one with a memory.max (the scope, then the slice)
    p = pid()
    try: cg = [l.split(':', 2)[2].strip() for l in open('/proc/%d/cgroup' % p) if l.startswith('0::')][0]
    except Exception: return []
    out = []; d = '/sys/fs/cgroup' + cg
    while d.startswith('/sys/fs/cgroup/') and len(out) < 3:
        try:
            if open(d + '/memory.max').read().strip() != 'max': out.append(d)
        except OSError: pass
        d = os.path.dirname(d)
    return out

def kv(path):
    try: return {a: int(b) for a, b in (l.split() for l in open(path))}
    except Exception: return {}

def memory():
    global oomk
    ds = cg_dirs(); s = ''
    if not ds: return 'cgroup ?'
    sl = ds[-1]   # the outermost capped cgroup: the slice (ufsm-train.slice)
    mx = int(open(sl + '/memory.max').read()); stat = kv(sl + '/memory.stat'); ev = kv(sl + '/memory.events')
    unrec = stat.get('anon', 0) + stat.get('shmem', 0); cur = int(open(sl + '/memory.current').read())
    s = '%s %.1f / %.1f GB (anon+shmem %.1f)' % (os.path.basename(sl), cur / 1e9, mx / 1e9, unrec / 1e9)
    if unrec > 0.9 * mx: alert('slice', force=True, msg= 'memory: %s anon + shmem %.1f GB = %.0f %% of its max %.1f GB (OOM risk)' % (os.path.basename(sl), unrec / 1e9, 100 * unrec / mx, mx / 1e9))
    k = ev.get('oom_kill', 0)
    if oomk is not None and k > oomk: alert('oomkill-%d' % k, force=True, msg= 'memory: %d new OOM kill(s) in %s (memory.events oom_kill %d)' % (k - oomk, os.path.basename(sl), k))
    oomk = k
    return s

def disks():
    paths = ['/', A.root, os.environ.get('UFSM_DERIVED_CACHE') or os.path.expanduser('~/.cache/ufsm/derived')] + [p for p in A.disks.split(',') if p]
    seen = {}; out = []
    for p in paths:
        try: st = os.stat(p)
        except OSError: continue
        if st.st_dev in seen: continue
        seen[st.st_dev] = p; u = shutil.disk_usage(p); now = time.time()
        h = disk_hist[st.st_dev]; h.append((now, u.free)); h[:] = [x for x in h if now - x[0] < 3 * 3600]
        rate = (h[0][1] - u.free) / (now - h[0][0]) if now - h[0][0] > 600 else 0.   # bytes / s shrinking
        eta = u.free / rate / 3600 if rate > 0 else float('inf')
        out.append('%s %.0f GB free%s' % (p, u.free / 1e9, ' (%.1f GB/h, full in %.0f h)' % (rate * 3.6e3 / 1e9, eta) if rate > 0 else ''))
        if u.free < 0.1 * u.total or u.free < 50e9 or eta < 12:
            alert('disk-' + p, force=True, msg= 'disk: %s has %.0f GB free of %.0f%s' % (p, u.free / 1e9, u.total / 1e9, '; shrinking %.1f GB/h, full in ~%.0f h' % (rate * 3.6e3 / 1e9, eta) if rate > 0 else ''))
    return '; '.join(out)

while True:
    for l in read_new(LOG).splitlines():
        if 'device memory free fell' in l: alert('gpumem-%d' % last_step, 'GPU memory growth after step %d: %s' % (last_step, l.strip()[:200]))
        m = re.match(r'step +(\d+) ', l)
        if not m: continue
        step = int(m.group(1)); last_step = step; last_step_t = time.time()
        if re.search(r'\b(nan|inf)\b', l, re.I): alert('nan', 'NaN / inf at step %d: %s' % (step, l[:300]))
        w = re.search(r'wait (\d+)%', l); waits = (waits + [int(w.group(1))])[-3:] if w else waits
        if len(waits) == 3 and min(waits) > 20: alert('wait', 'loader wait > 20%% on the last 3 logs (%s) at step %d' % (waits, step))
        for h, rx in HEADS:
            v = re.search(rx, l)
            if v:
                try: hist[h].append(float(v.group(1)))
                except ValueError: pass
        if step >= 2000:
            for h, H in hist.items():
                Hn = [x for x in H if x > 0]   # (a log line is one window's losses: heads without labels in it log 0)
                if len(Hn) >= 25:
                    a, b = statistics.mean(Hn[-5:]), statistics.mean(Hn[-25:-5])
                    if b > 0 and a > 2 * b: alert('rise-' + h, 'head %s: mean %.4f over the last 5 logs vs %.4f over the 20 before (step %d)' % (h, a, b, step))
                if len(H) >= 10 and any(x > 0 for x in H[:-10]) and all(x == 0 for x in H[-10:]) and h not in ('phase', 'normal', 'wind4', 'fib', 'fibdir'):
                    alert('zero-' + h, 'head %s: 0 on the last 10 logs (no windows for it?) at step %d' % (h, step))
        if step % 1000 == 0:
            alert('status-%d' % step, 'status step %d: %s' % (step, ' '.join('%s %.4f' % (h, statistics.mean([x for x in H if x > 0][-20:])) for h, H in hist.items() if any(x > 0 for x in H))))
    for l in read_new(RES).splitlines():
        if 'exit rc=' in l and not os.path.exists(o + '.evpause'): alert('exit-' + l[:40], 'trainer exit: ' + l.strip() + ' (last step %d; tail of %s.log for the reason)' % (last_step, A.run))
        if 'STOP file' in l or 'MemAvailable' in l or 'giving up' in l: alert('res-' + l[:40], l.strip())
    if alive() and time.time() - last_step_t > 1800: alert('stall', 'no new step for %.0f min (last %d) while the trainer lives' % ((time.time() - last_step_t) / 60, last_step))
    ms = memory() if alive() else 'trainer not running'; ds = disks()
    with open(ST, 'w') as f: f.write(time.strftime('%F %T') + ' step %d alive %d rss %.1f GB | %s | %s | waits %s\n' % (last_step, alive(), rss_gb(), ms, ds, waits))
    if A.once: break
    quiet = False
    time.sleep(A.every)
