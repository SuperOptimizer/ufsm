# tests/srcdeps.awk: which repo files each binary of a ninja build dir is made of (run_suite.sh -c). Prints "BIN PATH" lines, PATH
# relative to the repo root R (-v R=/abs/repo/): every source and header (ninja's depfiles: also headers with no code, which DWARF line
# tables miss) of the objects the binary's own objects reach through symbol references (nm U -> D), so a binary that links every CUDA
# object but calls three of them depends on those three only. Constructor objects (.init_array) count as reached.
# Input records (one stream, any order):
#   L BIN OBJ    BIN links OBJ (build.ninja)          D OBJ SYM / U OBJ SYM   OBJ defines / references SYM (nm)
#   R OBJ        OBJ has a constructor                 H OBJ FILE              OBJ was compiled from FILE (ninja -t deps)
$1 == "D" { def[$3] = def[$3] " " $2; next }
$1 == "U" { und[$2] = und[$2] " " $3; next }
$1 == "R" { root[$2] = 1; next }
$1 == "L" { lk[$2] = lk[$2] " " $3; inb[$2 SUBSEP $3] = 1; next }
$1 == "H" { if (index($3, R) == 1) hd[$2] = hd[$2] " " substr($3, length(R) + 1); next }
END {
  for (b in lk) {
    n = split(lk[b], os, " "); qn = 0; split("", seen); split("", out)
    for (i = 1; i <= n; i++) if (index(os[i], "/" b ".dir/") || (os[i] in root)) { q[++qn] = os[i]; seen[os[i]] = 1 }
    for (k = 1; k <= qn; k++) {
      m = split(und[q[k]], us, " ")
      for (j = 1; j <= m; j++) { d = split(def[us[j]], ds, " ")
        for (t = 1; t <= d; t++) if (((b SUBSEP ds[t]) in inb) && !(ds[t] in seen)) { seen[ds[t]] = 1; q[++qn] = ds[t] } }
    }
    for (k = 1; k <= qn; k++) { h = split(hd[q[k]], hs, " "); for (j = 1; j <= h; j++) if (!(hs[j] in out)) { out[hs[j]] = 1; print b, hs[j] } }
  }
}
