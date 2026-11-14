#!/usr/bin/env python3
# tests/lint_mgpu.py [--list] [--update]: process-wide mutable state must say how it behaves under 2-GPU data parallel (ops audit #8).
# Train runs one host thread per GPU in one process; a file-scope mutable variable is shared by both, and that class of bug (a global
# where per-device state was needed) cost several runs on 11-13 / 11-14. Every file-scope mutable variable in src/*.c / *.cu / *.cuh
# (static or not; not const / constexpr / __constant__ / __device__ / extern / functions) needs a class, either
#   - an annotation on its line or the line above: /* mgpu: process */ (one value for the process: set before the GPU threads start,
#     read-only after, or host-side work outside the trainer's GPU threads) | mgpu: device (indexed by device) | mgpu: guarded (a mutex,
#     atomic or thread-local) | mgpu: BUG (needs a fix; listed by --list), or
#   - an entry in tests/mgpu_baseline.txt (FILE NAME CLASS: the variables that existed on 2026-11-14, classified without touching
#     every source line, which would conflict with every open branch).
# A new variable without either fails the test, with the line to annotate. --update appends the unclassified ones as "unknown"
# (then classify them by hand). --list prints every variable with its class.
import os, re, sys
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(H, 'src'); BASE = os.path.join(H, 'tests', 'mgpu_baseline.txt')
CLASSES = ('process', 'device', 'guarded', 'BUG', 'unknown')
SKIP_START = ('static const ', 'static constexpr ', 'const ', 'constexpr ', 'extern ', 'static inline ', 'inline ', 'static __device__',
              '__device__', '__constant__', '__global__', '__shared__', 'static __constant__', 'typedef ', 'static __global__', 'template',
              'using ', 'struct ', 'enum ', 'union ', 'static struct {', 'namespace', 'static_assert', 'static __forceinline__',
              'static __host__', '__host__', 'static constinit const')
TYPE_START = re.compile(r'^(static\s+|thread_local\s+|_Thread_local\s+|__thread\s+)*(volatile\s+)?(unsigned|signed|int|float|double|char|size_t|long|short|bool|'
                        r'u?int\d+_t|uintptr_t|ssize_t|pthread_\w+|atomic_\w+|_Atomic|cuda\w+_t|cudaStream_t|cudaEvent_t|cublas\w+|FILE|time_t|'
                        r'[a-z_][a-z0-9_]*_t|[a-z_][a-z0-9_]*)\b')

def strip_comments(s):
    s = re.sub(r'/\*.*?\*/', lambda m: re.sub(r'[^\n]', ' ', m.group(0)), s, flags=re.S)   # (newlines kept: line numbers)
    return re.sub(r'//[^\n]*', '', s)

def names(seg):
    """declarator names of one 'static T a = 1, *b[4];' segment"""
    seg = seg.strip()
    depth = 0; out = []; cur = ''
    for ch in seg:
        if ch in '({[': depth += 1
        elif ch in ')}]': depth -= 1
        if ch == ',' and depth == 0: out.append(cur); cur = ''
        else: cur += ch
    out.append(cur); res = []
    for k, d in enumerate(out):
        d = d.split('=')[0]; d = re.sub(r'\[[^\]]*\]', '', d)
        ids = re.findall(r'[A-Za-z_]\w*', d)
        if ids: res.append(ids[-1])
    return res

DECL = re.compile(r'^((static|thread_local|_Thread_local|__thread|volatile|unsigned|signed|long|short|struct|enum)\s+)*'
                  r'[A-Za-z_]\w*(\s*<[^;]*>)?(\s*\*+\s*|\s+)(\*\s*)*[A-Za-z_]\w*\s*(\[[^;]*\])*\s*(=|,|$)', re.S)
TLS = re.compile(r'\b(thread_local|_Thread_local|__thread)\b')

def scan(path):
    """file-scope declarations: statements at brace depth 0 ending in ';' (preprocessor lines and comments removed)"""
    raw = open(path, errors='replace').read(); rl = raw.split('\n')
    txt = strip_comments(raw)
    txt = re.sub(r'(?m)^[ \t]*#[^\n]*(\\\n[^\n]*)*', lambda m: re.sub(r'[^\n]', ' ', m.group(0)), txt)   # preprocessor lines, continued
    txt = re.sub(r"'(\\.|[^'\\\n])'", lambda m: "'" + ' ' * (len(m.group(0)) - 2) + "'", txt)   # char literals first ('"', '{')
    txt = re.sub(r'"(\\.|[^"\\\n])*"', lambda m: '"' + ' ' * (len(m.group(0)) - 2) + '"', txt)   # string contents (braces in strings)
    found = []; depth = 0; start = 0; line = 1; sline = 1; pd = 0
    for k, ch in enumerate(txt):
        if ch == '\n': line += 1
        if ch in '([': pd += 1
        elif ch in ')]': pd -= 1
        if ch == '{': depth += 1
        elif ch == '}':
            depth -= 1
            if depth <= 0: depth = 0; start = k + 1; sline = line
        elif ch == ';' and depth == 0 and pd == 0:
            seg = txt[start:k]; p0 = start + (len(seg) - len(seg.lstrip())); st = ' '.join(seg.split()); start = k + 1
            sline = txt.count('\n', 0, p0) + 1
            if p0 > 0 and txt[p0 - 1] != '\n': continue   # file-scope declarations start in column 0 here (an indented one is a local
                                                         # the brace count lost track of: #if / #else branches with braces)
            if st and not st.startswith(SKIP_START) and not re.search(r'\b(extern|typedef|const|constexpr|__constant__|__device__|__shared__|__managed__)\b', st.split('=')[0].replace('*const', '*')) \
               and DECL.match(st) and not re.search(r'\b(return|goto)\b', st):
                head = st.split('=')[0]
                if '(' in head and not re.search(r'\(\s*\*', head): sline = line; continue   # function declarations
                # the line of the declaration: the first source line from sline that holds the first name
                for n in names(st):
                    if n in ('static', 'thread_local', 'int', 'unsigned', 'char', 'float', 'double', 'struct', 'enum'): continue
                    ln = sline
                    for q in range(sline - 1, min(line, len(rl))):
                        if re.search(r'\b%s\b' % re.escape(n), rl[q]): ln = q + 1; break
                    ann = re.search(r'mgpu:\s*(\w+)', rl[ln - 1] + ' ' + (rl[ln - 2] if ln > 1 else ''))
                    cls = ann.group(1) if ann else ('guarded' if TLS.search(head) else None)
                    found.append((n, ln, cls))
            sline = line
        if depth == 0 and ch == '\n' and not txt[start:k].strip(): start = k + 1; sline = line
    return found

def main():
    base = {}
    if os.path.exists(BASE):
        for l in open(BASE):
            l = l.split('#')[0].split()
            if len(l) >= 3: base[(l[0], l[1])] = l[2]
    files = sorted(f for f in os.listdir(SRC) if f.endswith(('.c', '.cu', '.cuh')))
    rows = []; bad = []
    for f in files:
        for n, ln, ann in scan(os.path.join(SRC, f)):
            cls = ann or base.get((f, n))
            rows.append((f, n, ln, cls))
            if cls is None: bad.append((f, n, ln))
            elif cls not in CLASSES: bad.append((f, n, ln))
    if '--update' in sys.argv:
        with open(BASE, 'a') as o:
            for f, n, ln in bad: o.write('%s %s unknown\n' % (f, n))
        print('lint_mgpu: %d entries appended to %s as unknown' % (len(bad), BASE)); return 0
    if '--list' in sys.argv:
        for f, n, ln, cls in rows: print('%-24s %5d %-32s %s' % (f, ln, n, cls))
    from collections import Counter
    c = Counter(r[3] for r in rows)
    print('lint_mgpu: %d file-scope mutable variables in %d files: %s' % (len(rows), len(files), ', '.join('%s %d' % (k, v) for k, v in sorted(c.items(), key=lambda kv: str(kv[0])))))
    stale = [k for k in base if not any((r[0], r[1]) == k for r in rows)]
    if stale: print('lint_mgpu: note: %d baseline entries no longer found (rename / removal; prune with care): %s' % (len(stale), ' '.join('%s:%s' % k for k in stale[:10])))
    if bad:
        print('lint_mgpu: FAIL: %d mutable file-scope variable(s) without an mgpu class; annotate the line, e.g. /* mgpu: process */ '
              '(one value per process, set before the GPU threads), device (indexed by device), guarded (mutex / atomic / thread-local), BUG:' % len(bad))
        for f, n, ln in bad: print('  src/%s:%d  %s' % (f, ln, n))
        return 1
    print('lint_mgpu: ok'); return 0

sys.exit(main())
