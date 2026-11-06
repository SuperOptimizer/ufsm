#!/bin/bash
# boxeval.sh NAME Z Y X NZ NY NX: extract GT / our spiral / m7 th0.2 / m7 prob for a Paris 4 box (9.6 um), radial ridge, detector scores
set -e; N=$1; Z=$2; Y=$3; X=$4; NZ=$5; NY=$6; NX=$7; U=/vesuvius/ufsm/v2/strip-src/build/ufsm
S=/vesuvius/ufsm/volcomp/PHercParis4/representations/predictions/surfaces
mkdir -p /mnt/scratch/v2/trace/$N; cd /mnt/scratch/v2/trace/$N
$U read /vesuvius/ufsm/gt/paris4-q9p6/q-9p6.zarr 9.6 $Z $Y $X $NZ $NY $NX gt.raw --threads 8 > /dev/null
$U read /mnt/scratch/v2/cal/PHercParis4/codes.zarr 9.6 $Z $Y $X $NZ $NY $NX ours.raw --threads 8 > /dev/null
$U read $S/20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr 9.6 $Z $Y $X $NZ $NY $NX m7.raw --threads 8 > /dev/null
$U read https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHercParis4/representations/predictions/surfaces/20260411134726-surface-20260925170000-surface-m7-L2-prob.zarr 9.6 $Z $Y $X $NZ $NY $NX prob.raw --threads 16 --cache /mnt/scratch/vesuvius-offload/cache > /dev/null
~/miniconda3/bin/python3 /mnt/scratch/v2/trace/deteval.py $Z $Y $X $NZ $NY $NX
