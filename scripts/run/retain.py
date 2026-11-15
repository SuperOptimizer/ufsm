#!/usr/bin/env python3
"""retain.py (2026-11-14, ops audit #19): the /vesuvius retention policy. /vesuvius grew ~140 GB/day on 11-13/14 (profile dumps
46 GB, per-arm bench binary copies 12 GB, try checkpoints) with no rule for what may go; at 87 % used that is ~2 days to full.

Policy: one rule per scratch tree (RULES below; --rule DIR:DAYS adds more). A rule's unit is each top-level entry of DIR (a file or
a whole directory); an entry goes when nothing inside it was modified for DAYS days. Checkpoints are not here: the trainer's
--ckpt-ring / --snap-keep-* (ops audit #10 / #11) own those. An entry is KEPT (and listed with the reason) when
  - a symlink anywhere under the --link-roots (default /vesuvius/ufsm, /home/forrest) points into it, or out of it into a live tree
    (CLAUDE.md: "before deleting data directories, check for symlinks pointing into them"; the 11-09 cleanup broke CT links);
  - a running process has a file in it open or as its cwd / executable (an arm still running from a bench binary copy);
  - it contains a file named KEEP (a manual pin).
Default: a dry run that lists what would go and the total. --apply deletes (each path is checked to lie strictly inside its rule's DIR
first). --json prints the plan for scripts/run/health.py, whose disk alarm names this script.
"""
import argparse, json, os, shutil, sys, time

RULES = [  # (dir, days): scratch trees only; never run roots, GT, CT or label stores
    ('/vesuvius/ufsm/perf2/prof', 3),     # nsys / ncu / UFSM_PROF dumps
    ('/vesuvius/ufsm/merge/bench', 3),    # per-arm ufsm.<sha> binary copies and their gate dirs
    ('/vesuvius/ufsm/gt/pub1667-cmp', 7), # GT comparison renders
]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--apply', action='store_true', help='delete (default: dry run)')
ap.add_argument('--json', action='store_true', help='print the plan as json')
ap.add_argument('--rule', action='append', default=[], metavar='DIR:DAYS', help='an extra rule (repeatable)')
ap.add_argument('--only-extra', action='store_true', help='use only the --rule rules (tests)')
ap.add_argument('--link-roots', default='/vesuvius/ufsm,/home/forrest', help='where to look for symlinks into the candidates')
A = ap.parse_args()
rules = [] if A.only_extra else list(RULES)
for r in A.rule:
    d, _, n = r.rpartition(':'); rules.append((d, float(n)))
now = time.time()

def newest(p):   # the newest mtime of p or anything under it (links not followed)
    t = os.lstat(p).st_mtime
    if os.path.isdir(p) and not os.path.islink(p):
        for root, dirs, files in os.walk(p):
            for f in dirs + files:
                try: t = max(t, os.lstat(os.path.join(root, f)).st_mtime)
                except OSError: pass
    return t

def size(p):
    if not os.path.isdir(p) or os.path.islink(p): return os.lstat(p).st_size
    s = 0
    for root, _, files in os.walk(p):
        for f in files:
            try: s += os.lstat(os.path.join(root, f)).st_blocks * 512
            except OSError: pass
    return s

def inside(p, d): return p == d or p.startswith(d.rstrip('/') + '/')

cands = []   # (rule dir, path, age days)
for d, days in rules:
    if not os.path.isdir(d): continue
    d = os.path.realpath(d)
    for e in sorted(os.listdir(d)):
        p = os.path.join(d, e)
        try: age = (now - newest(p)) / 86400
        except OSError: continue
        if age >= days: cands.append((d, p, age))

# the symlinks that matter: their target resolves into a candidate, or the candidate holds a link out to elsewhere that is still live
links = []   # (link path, resolved target)
for root in [r for r in A.link_roots.split(',') if r]:
    for dp, dirs, files in os.walk(root):
        for f in dirs + files:
            q = os.path.join(dp, f)
            if os.path.islink(q):
                try: links.append((q, os.path.realpath(q)))
                except OSError: pass
        dirs[:] = [x for x in dirs if not x.startswith('.cache') and x not in ('.git', 'node_modules')]
# files in use: every /proc/<pid>/{cwd,exe,fd/*}
used = set()
for pid in filter(str.isdigit, os.listdir('/proc')):
    for q in ('cwd', 'exe'):
        try: used.add(os.readlink('/proc/%s/%s' % (pid, q)))
        except OSError: pass
    try:
        for fd in os.listdir('/proc/%s/fd' % pid):
            try: used.add(os.readlink('/proc/%s/fd/%s' % (pid, fd)))
            except OSError: pass
    except OSError: pass

plan = []
for d, p, age in cands:
    why = None
    for q, t in links:
        if inside(t, p) and not inside(q, p): why = 'symlink %s -> %s' % (q, t); break
    if not why:
        u = next((x for x in used if inside(x, p)), None)
        if u: why = 'in use: ' + u
    if not why and os.path.isdir(p) and not os.path.islink(p):
        for root, _, files in os.walk(p):
            if 'KEEP' in files: why = 'pinned: ' + os.path.join(root, 'KEEP'); break
    plan.append(dict(path=p, rule=d, age_days=round(age, 1), bytes=size(p), keep=why))

drop = [x for x in plan if not x['keep']]
total = sum(x['bytes'] for x in drop)
if A.json:
    print(json.dumps(dict(apply=A.apply, drop_bytes=total, entries=plan), indent=1))
else:
    for x in plan:
        print('%-5s %8.2f GB  %5.1f d  %s%s' % ('keep' if x['keep'] else 'drop', x['bytes'] / 1e9, x['age_days'], x['path'],
                                                  '  (' + x['keep'] + ')' if x['keep'] else ''))
    print('retain: %d of %d entries, %.1f GB %s' % (len(drop), len(plan), total / 1e9, 'deleted' if A.apply else 'would go (dry run; --apply deletes)'))
if A.apply:
    bad = 0
    for x in drop:
        p, d = x['path'], x['rule']
        if not (p.startswith(d.rstrip('/') + '/') and len(p) > len(d) + 1 and '/..' not in p):   # (guard: strictly inside the rule's dir)
            print('retain: refusing %s (not inside %s)' % (p, d), file=sys.stderr); bad += 1; continue
        try:
            if os.path.isdir(p) and not os.path.islink(p): shutil.rmtree(p)
            else: os.unlink(p)
        except OSError as e: print('retain: %s: %s' % (p, e), file=sys.stderr); bad += 1
    sys.exit(1 if bad else 0)
