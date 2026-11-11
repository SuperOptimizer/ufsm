"""List a prefix of the public Hugging Face storage bucket scrollprize/datasets (plain curl, nothing personal; the tree API is recursive
and paged through Link: rel=next) into a TSV (path, size, mtime) and print per-folder totals. analysis only.
usage: hfbucket.py PREFIX OUT.tsv [MAXPAGES]"""
import json, subprocess, collections, sys, re
pre, out = sys.argv[1], sys.argv[2]; maxp = int(sys.argv[3]) if len(sys.argv) > 3 else 400
url = "https://huggingface.co/api/buckets/scrollprize/datasets/tree/" + pre; rows = []; pages = 0
while url and pages < maxp:
    r = subprocess.run(["curl", "-s", "-L", "-D", "-", url], capture_output=True, text=True).stdout
    i = r.find("\n["); i = i if i >= 0 else r.find("\n{"); hd, body = r[:i], r[i + 1:]; pages += 1
    try: j = json.loads(body)
    except Exception: print("bad page", url, body[:200]); break
    rows += [(e["path"], e.get("size", 0), e.get("mtime", "")[:10]) for e in j if e["type"] == "file"]
    m = re.search(r'<([^>]+)>;\s*rel="next"', hd); url = m.group(1) if m else None
open(out, "w").write("".join(f"{p}\t{s}\t{t}\n" for p, s, t in rows))
c = collections.defaultdict(lambda: [0, 0, "9", "0"])
for p, s, t in rows:
    k = "/".join(p.split("/")[:3]); c[k][0] += 1; c[k][1] += s; c[k][2] = min(c[k][2], t); c[k][3] = max(c[k][3], t)
print(len(rows), "files,", pages, "pages,", "more pages exist" if url else "complete")
for k in sorted(c): print(f"{k}\t{c[k][0]} files\t{c[k][1] / 1e6:.1f} MB\t{c[k][2]}..{c[k][3]}")
