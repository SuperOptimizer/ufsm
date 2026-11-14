#!/usr/bin/env python3
"""DP-safety lint (2026-11-14 audit, DESIGN "DP safety"): every mutable process-wide variable in src/ -- file-scope globals, function-local
statics, __device__ / __constant__ globals -- must say why one host thread per GPU (data parallel) and --split z cannot break it, with a
comment `mgpu: TAG` on its line or the line above:

  per-thread   thread_local, or only touched by one known thread
  per-device   indexed by the current device (one host thread per device)
  guarded      a mutex / atomics
  config       set once before the per-GPU threads start, read-only after
  env          a lazily cached environment variable or constant table (same value from every thread)
  main         only the trainer's main thread touches it (optimizer, all-reduce, planner, checkpoint)
  process      deliberately process-wide; the comment says why that is safe
  debug        tests / profiling / debug dumps only

thread_local / __thread / _Thread_local, const and constexpr variables need nothing. The variables that existed when the lint was added are
listed in tests/mgpu_baseline_net.txt (file name per line) and pass unannotated; a new one fails the suite until it carries a tag (or, with a
reason, is added to the baseline). Today's bugs of this kind: at_set_lmax's pointer, the cuBLASLt handles / plans, lp_mxgemm's packed
buffer, the VMM rings' device.

usage: test_mgpu_lint.py [SRC_DIR] [BASELINE]  |  --baseline SRC_DIR > tests/mgpu_baseline_net.txt  (regenerate)
"""
import os
import re
import sys

TAGS = ("per-thread", "per-device", "guarded", "config", "env", "main", "process", "debug")
SAFE_WORDS = re.compile(r"\b(constexpr|thread_local|__thread|_Thread_local|typedef|extern|using|template|static_assert|return|operator"
                        r"|pthread_mutex_t|pthread_cond_t|pthread_once_t|pthread_rwlock_t|mutex|once_flag|atomic_int|_Atomic)\b")   # (locks are their own guard)
IDENT = re.compile(r"[A-Za-z_]\w*")


def strip(text):
    """comments and string / char literals blanked (newlines kept, so line numbers hold); comment text returned per line"""
    out, com = [], {}
    i, n, line = 0, len(text), 1
    while i < n:
        c = text[i]
        if c == "\n":
            out.append(c); line += 1; i += 1
        elif text.startswith("//", i):
            j = text.find("\n", i); j = n if j < 0 else j
            com[line] = com.get(line, "") + text[i:j]; out.append(" " * (j - i)); i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2); j = n if j < 0 else j + 2
            seg = text[i:j]
            for k, part in enumerate(seg.split("\n")):
                com[line + k] = com.get(line + k, "") + part
            out.append("".join(ch if ch == "\n" else " " for ch in seg)); line += seg.count("\n"); i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            seg = text[i:j]; out.append("".join(ch if ch == "\n" else " " for ch in seg)); line += seg.count("\n"); i = j
        elif c == "#" and (i == 0 or text[i - 1] == "\n" or text[:i].rstrip(" \t").endswith("\n")):
            j = i
            while True:   # a preprocessor line with its continuations
                k = text.find("\n", j); k = n if k < 0 else k
                if k > 0 and text[k - 1] == "\\" and k < n:
                    j = k + 1; continue
                break
            seg = text[i:k]; out.append("".join(ch if ch == "\n" else " " for ch in seg)); line += seg.count("\n"); i = k
        else:
            out.append(c); i += 1
    return "".join(out), com


def names_of(decl):
    """the variable names of a declaration statement (no trailing ';')"""
    s, depth, cur, parts = decl, 0, "", []
    for ch in s:   # split on top-level commas, drop initialisers
        if ch in "([{": depth += 1
        elif ch in ")]}": depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur); cur = ""
        else:
            cur += ch
    parts.append(cur)
    names = []
    for p in parts:
        p = p.split("=")[0]
        p = re.sub(r"\[[^\]]*\]", "", p)           # array bounds
        p = re.sub(r"\([^)]*\)\s*$", "", p)        # constructor-style init
        ids = IDENT.findall(p)
        if ids: names.append(ids[-1])
    return names


def scan(path):
    """(line, names, statement) of every mutable static-storage variable"""
    raw = open(path, encoding="utf-8", errors="replace").read()
    code, _ = strip(raw)
    found = []
    depth, stmt, stmt_line, stmt_depth = 0, "", 0, 0
    agg_open = []   # brace depths of open aggregate initialisers / struct bodies inside a declaration statement
    line = 1
    i, n = 0, len(code)
    while i < n:
        c = code[i]
        if not stmt.strip() and not c.isspace():
            stmt_line, stmt_depth = line, depth
        if c == "\n":
            line += 1; stmt += " "
        elif c == "{":
            head = stmt.strip()
            # a brace inside a declaration (struct body, aggregate initialiser) keeps the statement going; a function / namespace / control
            # body starts a new scope
            if re.search(r"(=|\bstruct\b[^;(]*|\bunion\b[^;(]*|\benum\b[^;(]*|\[\s*\]|\])\s*$", head) and "(" not in head.split("=")[0].replace("__align__(", "").replace("__attribute__((", ""):
                agg_open.append(depth); stmt += c
            else:
                stmt = ""
            depth += 1
        elif c == "}":
            depth -= 1
            if agg_open and agg_open[-1] == depth:
                agg_open.pop(); stmt += c
            else:
                stmt = ""
        elif c == ";" and not agg_open:
            s = " ".join(stmt.split()); stmt = ""
            if not s: i += 1; continue
            at_file = stmt_depth == 0
            if not at_file and not re.match(r"^static\b", s): i += 1; continue
            hd = s.split("=")[0].split("{")[0]
            if SAFE_WORDS.search(hd): i += 1; continue
            if re.search(r"\bconst\b", hd) and ("*" not in hd or re.search(r"\*\s*const\s+\w+\s*(\[[^\]]*\]\s*)*$", hd.strip())):
                i += 1; continue   # a const object (a pointer to const is still mutable)
            if re.match(r"^(struct|union|enum|class|namespace|if|for|while|switch|do|else|case|goto|break|continue)\b", s): i += 1; continue
            if re.match(r"^static\s+(struct|union|enum)\s+\w+\s*$", s): i += 1; continue
            head = s.split("=")[0]
            head_noinit = re.sub(r"\{.*\}", "", head)
            if "(" in head_noinit and not re.search(r"\(\s*\*", head_noinit) and "__align__(" not in head_noinit:
                i += 1; continue   # a prototype or a call
            if "__global__" in s or "__forceinline__" in s or re.search(r"\binline\b", s): i += 1; continue
            if at_file and not re.search(r"\b(static|__device__|__constant__)\b", s):
                if not re.match(r"^[A-Za-z_][\w:<>,\s\*&]*\s[\*&\s]*[A-Za-z_]\w*\s*(\[[^\]]*\]\s*)*(=|,|$)", s): i += 1; continue
            names = names_of(re.sub(r"\{.*\}", "", s) if "=" not in s.split("{")[0] else s)
            names = [x for x in names if x not in ("static", "struct", "unsigned", "int", "float", "double", "char", "void")]
            if names: found.append((stmt_line, line, names, s))
        else:
            stmt += c
        i += 1
    return raw.split("\n"), found


def tag_of(lines, a, b):
    for ln in range(max(1, a - 1), b + 1):
        m = re.search(r"mgpu:\s*([a-z-]+)", lines[ln - 1]) if ln - 1 < len(lines) else None
        if m: return m.group(1)
    return None


SELF = """
static uint8_t *g_pk = nullptr; static size_t g_pkn = 0;
static thread_local float *g_lmax;
static const int k_tab[3] = {1, 2, 3};
static const char *g_name;
static int g_dev[8];   /* mgpu: per-device */
/* mgpu: guarded (g_mu) */
static struct { int a; float b[4]; } g_ctx[2];
int g_plain = 1;
static void f(int x);
__constant__ float c_w[16];
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
void g(void) { static int attr[8]; static int on = -1; const int z = 2; static constexpr int K = 3; for (int i = 0; i < 2; i++) {} }
"""
SELF_WANT = {"g_pk": None, "g_pkn": None, "g_name": None, "g_dev": "per-device", "g_ctx": "guarded", "g_plain": None, "c_w": None, "attr": None, "on": None}


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.cu"); open(p, "w").write(SELF)
        lines, found = scan(p)
        got = {nm: tag_of(lines, a, b) for a, b, names, s in found for nm in names}
    if got != SELF_WANT:
        print(f"FAIL mgpu lint self-test: found {got}, want {SELF_WANT}"); return 1
    return 0


def main():
    if selftest(): return 1
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    regen = "--baseline" in sys.argv
    here = os.path.dirname(os.path.abspath(__file__))
    src = args[0] if args else os.path.join(here, "..", "src")
    base = args[1] if len(args) > 1 else os.path.join(here, "mgpu_baseline_net.txt")
    files = sorted(f for f in os.listdir(src) if f.endswith((".c", ".cu", ".cuh", ".h")))
    known = set()
    if not regen and os.path.exists(base):
        for l in open(base):
            l = l.split("#")[0].strip()
            if l: known.add(tuple(l.split()[:2]))
    bad, badtag, seen, ntag, seen_regen = [], [], set(), 0, set()
    for f in files:
        lines, found = scan(os.path.join(src, f))
        for a, b, names, s in found:
            t = tag_of(lines, a, b)
            for nm in names:
                seen.add((f, nm))
                if t:
                    ntag += 1
                    if t not in TAGS: badtag.append(f"{f}:{a}: {nm}: unknown mgpu tag '{t}' (one of {', '.join(TAGS)})")
                elif regen:
                    seen_regen.add(f"{f} {nm}")
                elif (f, nm) not in known:
                    bad.append(f"{f}:{a}: {nm}: process-wide variable without an mgpu tag   [{s[:110]}]")
    if regen:
        print("# tests/test_mgpu_lint.py baseline: the process-wide variables in src/ that predate the lint (2026-11-14), file name per line.")
        print("# Regenerate: python3 tests/test_mgpu_lint.py --baseline src > tests/mgpu_baseline_net.txt (only to drop entries; new ones get an mgpu tag)")
        for x in sorted(seen_regen): print(x)
        return 0
    stale = sorted(known - seen)
    print(f"mgpu lint: {len(seen)} process-wide variables in {len(files)} files, {ntag} tagged, {len(known & seen)} in the baseline, {len(stale)} stale baseline entries")
    for x in stale[:20]: print(f"  (stale baseline entry, can be dropped: {x[0]} {x[1]})")
    for x in badtag + bad: print("FAIL " + x)
    if bad: print("each new process-wide variable needs a comment 'mgpu: TAG' (tests/test_mgpu_lint.py says which); under data parallel one host thread runs per GPU")
    return 1 if bad or badtag else 0


if __name__ == "__main__":
    sys.exit(main())
