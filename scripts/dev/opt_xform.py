#!/usr/bin/env python3
"""opt_xform.py SRC_DIR: route option lookups through src/optcheck.h (mini audit opts 2-3).
 - strcmp(X, "--lit")            -> strcmp(X, opt_n("--lit"))       (any X: argv[i], a helper's k)
 - strcmp(argv[..], ident)       -> strcmp(argv[..], opt_n(ident))  (the opt helpers' name parameter)
 - strncmp(X, "--lit", n)        -> strncmp(X, opt_p("--lit"), n)   (prefix families)
 - ato{i,l,ll,f}(H(a, b, "--n", ..  / H("--n", ..) -> opt_ato{..}("--n", H(..   (strict numbers)
prints per-file counts. Idempotent: re-run it after merging a branch that adds option reads (a conflict in a rewritten line: take
the other side, re-run). Hand fixes it does not do: flags that also take a word (predict --window auto, train --accum auto). 2026-11-14."""
import re, sys, pathlib
A = re.compile(r'\bstrcmp\(([^,()]+?), ("--[A-Za-z0-9][^"]*")\)')
B = re.compile(r'\bstrcmp\(((?:argv|av)\[[^\]]+\]), ([A-Za-z_]\w*)\)')
C = re.compile(r'\bstrncmp\(([^,()]+?), ("--[A-Za-z0-9][^"]*"), ')
N = re.compile(r'\bato(i|f|ll|l)\((\w+)\(((?:\w+, \w+, )?)("--[A-Za-z0-9][^"]*"), ')
tot = [0] * 4
for p in sorted(pathlib.Path(sys.argv[1]).glob('*.c')):
    if p.name == 'optcheck.c': continue
    s = p.read_text(); o = s; c = [0] * 4
    s, c[0] = A.subn(r'strcmp(\1, opt_n(\2))', s)
    s, c[1] = B.subn(r'strcmp(\1, opt_n(\2))', s)
    s, c[2] = C.subn(r'strncmp(\1, opt_p(\2), ', s)
    s, c[3] = N.subn(r'opt_ato\1(\4, \2(\3\4, ', s)
    if s != o:
        if '#include "optcheck.h"' not in s:
            lines = s.split('\n'); first = next(i for i, l in enumerate(lines) if l.startswith('#include'))
            lines.insert(first + 1, '#include "optcheck.h"'); s = '\n'.join(lines)   # (after the first: some files include more after code)
        p.write_text(s); print(f'{p.name}: strcmp-lit {c[0]} strcmp-param {c[1]} prefix {c[2]} numeric {c[3]}')
        tot = [a + b for a, b in zip(tot, c)]
print('total', tot)
