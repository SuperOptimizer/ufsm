#!/bin/bash
# hfget.sh: the human-labelled Paris 4 ink segments of the public HF bucket (ink/phercparis4/*): meta, labels, mask, x/y/z (no zarrs)
D=/vesuvius/ufsm/pbr/hfink; B="https://huggingface.co/buckets/scrollprize/datasets/resolve/ink/phercparis4"
for w in w00_20231016151002 w01_20230702185753_r15 w02_20231031143852 w03_20231106155351 w05_4424 w06_1321 w07_1619 w09_20230929220926; do
  mkdir -p $D/$w
  for f in $(curl -s --max-time 60 "https://huggingface.co/api/buckets/scrollprize/datasets/tree/ink/phercparis4/$w?recursive=false" | python3 -I -c "import sys,json; [print(r['path'].split('/')[-1]) for r in json.load(sys.stdin) if r['type']=='file']"); do
    [ -s $D/$w/$f ] || curl -sL --max-time 1800 -o $D/$w/$f "$B/$w/$f" &
  done; wait
done
