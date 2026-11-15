#!/bin/sh
# native_id.sh: one line naming what a native build (-march=native, SASS for the GPU present) is specific to: the CPU model and
# its flags hash, the GPU compute capabilities. CMake writes it to $BUILD/native_id at configure; scripts/pod/setup.sh compares it
# with the machine it runs on before using an existing binary (NOBUILD=1), so a binary built elsewhere is never run by accident.
cpu=$(grep -m1 '^model name' /proc/cpuinfo 2> /dev/null | cut -d: -f2- | sed 's/^ *//')
fl=$(grep -m1 '^flags' /proc/cpuinfo 2> /dev/null | cksum | cut -d' ' -f1)
gpu=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' ' | sort -u | tr '\n' ',')
echo "cpu=$cpu flags=$fl gpu=$gpu"
