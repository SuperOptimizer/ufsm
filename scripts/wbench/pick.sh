#!/usr/bin/env bash
# pick.sh MAP B N MINLAB SEEDS: greedy disjoint B^3-chunk boxes over a phase-packing map (chunk grid), most labelled chunks first;
# SEEDS "z,y,x,b z,y,x,b" pre-placed boxes (chunk units, size b). Env: BZ (box z size in chunks, default B), ZLO / ZHI (chunk z range
# the box must lie in). Prints: cz cy cx b nlab meanpack (b = the xy size; z size BZ)
M=$1; B=$2; N=$3; MIN=$4; SEEDS=${5:-}
read gz gy gx < <(grep -oE '"grid":\[[0-9,]+\]' $M.json | grep -oE '[0-9]+' | tr '\n' ' ')
od -An -v -tu1 -w1 $M | awk -v gz=$gz -v gy=$gy -v gx=$gx -v B=$B -v N=$N -v MIN=$MIN -v seeds="$SEEDS" -v BZ=${BZ:-$B} -v ZLO=${ZLO:-0} -v ZHI=${ZHI:-1000000} '
{ v[NR-1]=$1 }
END {
  ns=split(seeds, S, " "); np=0
  for (i=1;i<=ns;i++) { split(S[i], q, ","); pz[np]=q[1]; py[np]=q[2]; px[np]=q[3]; pb[np]=q[4]; seed[np]=1; np++ }
  for (z=ZLO; z+BZ<=gz && z+BZ<=ZHI; z++) for (y=0; y+B<=gy; y++) for (x=0; x+B<=gx; x++) {
    c=0; p=0; for (a=0;a<BZ;a++) for (b=0;b<B;b++) for (d=0;d<B;d++) { t=v[((z+a)*gy+y+b)*gx+x+d]; if (t>0) { c++; p+=(t-1)/254 } }
    if (c>=MIN) { k=z" "y" "x; sc[k]=(ENVIRON["KEY"]=="pack" ? p/c : c); pk[k]=p/c }
  }
  for (i=0;i<np;i++) { k=pz[i]" "py[i]" "px[i]; print pz[i], py[i], px[i], pb[i], (k in sc ? sc[k] : "seed"), (k in pk ? sprintf("%.3f", pk[k]) : "-") }
  while (np < N) {
    best=""; bv=-1
    for (k in sc) { if (sc[k] <= bv) continue; split(k, o, " "); ok=1
      for (i=0;i<np && ok;i++) if (o[1] < pz[i]+(seed[i] ? pb[i] : BZ)+1 && pz[i] < o[1]+BZ+1 && o[2] < py[i]+pb[i]+1 && py[i] < o[2]+B+1 && o[3] < px[i]+pb[i]+1 && px[i] < o[3]+B+1) ok=0
      if (ok) { bv=sc[k]; best=k } }
    if (best=="") break
    split(best, o, " "); pz[np]=o[1]; py[np]=o[2]; px[np]=o[3]; pb[np]=B; np++; print o[1], o[2], o[3], B, sc[best], sprintf("%.3f", pk[best]); delete sc[best]
  }
}'
