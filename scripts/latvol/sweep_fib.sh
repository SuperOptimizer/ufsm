#!/bin/bash
# sweep_fib.sh ARM...: the compression sweep's fibre score on the pod: fibhv-c on HOLD from the arm's CT tree (sweep.sh), fiber-trace +
# fiber-judge through scripts/univ/eval.sh (TASKS=fib), one arm = one shared GPU unit. Results: /workspace/latstore/ev/ARM/fibhv-c-ARM/fib/*.judge
L=/workspace/latstore; H=$L/ufsm; SW=$L/sw; P4=PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
for a in "$@"; do
  while [ ! -e $SW/$a/$P4/.done ]; do sleep 20; done
  WB=$L/fwb/$a; mkdir -p $WB/res $L/ev/$a; [ -e $WB/inkeval ] || ln -s /workspace/ue/wb/inkeval $WB/inkeval; for d in bin ceil; do [ -e $WB/$d ] || ln -s /workspace/ue/wb/$d $WB/$d; done
  SHN=latstore-fib bash $H/scripts/latvol/pod_shared.sh env SITE=pod K=/workspace/ue WB=$WB E=$L/ev/$a CACHE=$L/cache/$a GPU=0 U=$H/build/ufsm CTROOT=$SW/$a TASKS=fib \
    FIBCH="H=0-0 V=1-1 dyad=2-7" FIBPH=/workspace/ue/wbgt/paris4-wind-20261019 bash $H/scripts/univ/eval.sh /workspace/ue/ckpt/fibhv-c-12000.ckpt fibhv-c-$a > $SW/log/fib_$a.log 2>&1
  for f in H-tube V-tube H-dyad V-dyad; do echo "$a $f $(tr '\n' ' ' < $L/ev/$a/fibhv-c-$a/fib/$f.judge 2>/dev/null | cut -c1-300)"; done
  rm -rf $L/ev/$a/fibhv-c-$a/fib/{H,V,dyad} $L/cache/$a
done
