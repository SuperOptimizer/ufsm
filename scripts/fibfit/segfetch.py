#!/usr/bin/env python3
"""segfetch.py SCROLL SCANID OUTDIR [N]: up to N (4) open-data segment meshes of SCROLL on scan SCANID (tifxyz x / y / z.tif) into
OUTDIR/<segment>/, smallest x.tif first among names with a wrap tag (-wNNN) if any (SEGFETCH_LARGEST=1: the largest first, for scrolls whose small segments cover < 0.8 turn); prints the local tifxyz dirs (for seghand.py)."""
import os, re, sys, urllib.request, urllib.parse, xml.etree.ElementTree as ET
B = 'https://vesuvius-challenge-open-data.s3.amazonaws.com'; NS = '{http://s3.amazonaws.com/doc/2006-03-01/}'
def ls(prefix, delim=True):
    out, tok = [], None
    while True:
        q = {'list-type': '2', 'prefix': prefix}
        if delim: q['delimiter'] = '/'
        if tok: q['continuation-token'] = tok
        t = ET.fromstring(urllib.request.urlopen(f'{B}/?{urllib.parse.urlencode(q)}', timeout=60).read())
        out += [(p.find(NS + 'Prefix').text, 0) for p in t.findall(NS + 'CommonPrefixes')]
        out += [(c.find(NS + 'Key').text, int(c.find(NS + 'Size').text)) for c in t.findall(NS + 'Contents')]
        if t.find(NS + 'IsTruncated').text != 'true': return out
        tok = t.find(NS + 'NextContinuationToken').text
S, ID, OUT = sys.argv[1:4]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 4
segs = [p for p, _ in ls(f'{S}/segments/') if p.endswith('/')]
cand = []
for sg in segs:
    for p, _ in ls(f'{sg}mesh/'):
        if p.endswith('.tifxyz/') and f'on-{ID}' in p:
            sz = [s for k, s in ls(p, False) if k.endswith('/x.tif')]
            if sz: cand.append((0 if re.search(r'-w\d{3}', sg) else 1, sz[0], p))
cand.sort(key=(lambda c: -c[1]) if os.environ.get("SEGFETCH_LARGEST") else None)
print(f'# {S} {ID}: {len(segs)} segments, {len(cand)} meshes on this scan', file=sys.stderr)
for _, sz, p in cand[:N]:
    d = os.path.join(OUT, p.rstrip('/').split('/')[-1]); os.makedirs(d, exist_ok=True)
    for c in 'xyz':
        f = f'{d}/{c}.tif'
        if not os.path.exists(f): urllib.request.urlretrieve(f'{B}/{urllib.parse.quote(p)}{c}.tif', f)
    print(d)
