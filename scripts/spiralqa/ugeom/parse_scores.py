"""audit ugeom: parse_scores.py RUN OUT.tsv: a run's score/*.log GT scores (boxes and z-bands) as one table."""
import re,glob,sys,os,json
import numpy as np
run,out=sys.argv[1],sys.argv[2]
R=re.compile(r'fit score: (\d+) labelled voxels, gauge ([-\d.]+), winding ids right ([\d.]+), \|error\| < 0.08 turn ([\d.]+), \|error\| median ([\d.]+) p90 ([\d.]+)')
P=re.compile(r'0.5 mm ([\d.]+) .*\| 1 mm ([\d.]+) .*\| 2 mm ([\d.]+) .*\| 4 mm ([\d.]+)')
Z=re.compile(r'scored label box z (\d+)\.\.(\d+)')
rows=[]
for f in sorted(glob.glob(f'{run}/score/*.log')):
    nm=os.path.basename(f)[:-4]; t=open(f).read(); m=R.search(t); p=P.search(t); z=Z.search(t)
    if not m: continue
    rows.append([nm,int(z[1]),int(z[2]),int(m[1]),float(m[2]),float(m[3]),float(m[4]),float(m[5]),float(m[6])]+[float(x) for x in p.groups()])
with open(out,'w') as o:
    o.write('name\tlz0\tlz1\tn\tgauge\tids_right\terr08\terr_med\terr_p90\tpair05\tpair1\tpair2\tpair4\n')
    for r in rows: o.write('\t'.join(map(str,r))+'\n')
print(len(rows))
