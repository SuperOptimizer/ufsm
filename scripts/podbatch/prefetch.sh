#!/usr/bin/env bash
# prefetch.sh SCROLL SCAN LEVEL [PARALLEL]: download one level of a mirror volume (whole shard files, curl --parallel) to
# /workspace/vol/<SCROLL>/volumes/<SCAN>.zarr so predict reads local files (root /workspace/vol) instead of HTTP range requests.
# --http1.1: one TCP connection per transfer (over HTTP/2 curl multiplexes all of them on one connection, ~30 MB/s).
# Writes <dst>/<LEVEL>/.complete when every shard of the grid is either downloaded with the server's size or absent (404).
set -u; s=$1 scan=$2 lv=$3 P=${4:-48}; R=https://dl.ash2txt.org/community-uploads/forrest/volcomp
base=$R/$s/volumes/$scan.zarr; dst=/workspace/vol/$s/volumes/$scan.zarr; mkdir -p $dst/$lv
[ -f $dst/$lv/.complete ] && exit 0
curl -sf --retry 5 $base/zarr.json -o $dst/zarr.json && curl -sf --retry 5 $base/$lv/zarr.json -o $dst/$lv/zarr.json || { echo "prefetch: no zarr.json for $s $lv"; exit 1; }
read -r Z Y X <<< "$(jq -r '.shape | @tsv' $dst/$lv/zarr.json)"; read -r cz cy cx <<< "$(jq -r '.chunk_grid.configuration.chunk_shape | @tsv' $dst/$lv/zarr.json)"
nz=$(( (Z + cz - 1) / cz )) ny=$(( (Y + cy - 1) / cy )) nx=$(( (X + cx - 1) / cx ))
for pass in 1 2 3; do
  cfg=$dst/$lv/.curl.cfg; : > $cfg; n=0
  for ((z = 0; z < nz; z++)); do for ((y = 0; y < ny; y++)); do for ((x = 0; x < nx; x++)); do
    k=c/$z/$y/$x; [ -f $dst/$lv/$k ] || [ -f $dst/$lv/$k.absent ] && continue
    printf 'url = "%s"\noutput = "%s.part"\n' "$base/$lv/$k" "$dst/$lv/$k" >> $cfg; n=$((n + 1))
  done; done; done
  [ $n = 0 ] && break
  curl -s --http1.1 --parallel --parallel-max $P --create-dirs --retry 3 -K $cfg -w '%{http_code} %{size_download} %{filename_effective}\n' > $dst/$lv/.curl.log 2>&1
  while read -r code sz f; do t=${f%.part}
    case $code in 200) [ -s "$f" ] && [ $(stat -c %s "$f") = "$sz" ] && mv "$f" "$t" || rm -f "$f";; 404) rm -f "$f"; : > "$t.absent";; *) rm -f "$f";; esac
  done < $dst/$lv/.curl.log
done
[ $n = 0 ] && { echo "$nz x $ny x $nx shards" > $dst/$lv/.complete; rm -f $dst/$lv/.curl.cfg $dst/$lv/.curl.log; exit 0; }
echo "prefetch: $s $lv: $n shards still missing after 3 passes"; exit 1
