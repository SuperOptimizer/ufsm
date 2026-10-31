// ufsm sysinfo: the CUDA devices this process sees (MIG slices included: a MIG instance is one device with its slice's memory)
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

extern "C" int nn_sysinfo_gpus(FILE *o) {
    int drv = 0, rt = 0, n = 0;
    cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
    fprintf(o, "cuda: driver %d.%d, runtime %d.%d (built with %d.%d)\n", drv / 1000, drv % 1000 / 10, rt / 1000, rt % 1000 / 10,
            CUDART_VERSION / 1000, CUDART_VERSION % 1000 / 10);
    const cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) { fprintf(o, "gpu: none (%s)\n", cudaGetErrorString(e)); return 0; }
    for (int d = 0; d < n; d++) {
        cudaDeviceProp p; if (cudaGetDeviceProperties(&p, d) != cudaSuccess) continue;
        size_t fr = 0, tot = 0;
        if (cudaSetDevice(d) == cudaSuccess) cudaMemGetInfo(&fr, &tot);
        char u[40]; const unsigned char *b = (const unsigned char *)p.uuid.bytes;
        snprintf(u, sizeof u, "%02x%02x%02x%02x-%02x%02x-%02x%02x", b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
        fprintf(o, "gpu %d: %s, sm_%d%d, %d SMs, %.1f GB (%.1f GB free), uuid %s...%s\n", d, p.name, p.major, p.minor, p.multiProcessorCount,
                p.totalGlobalMem / 1e9, fr / 1e9, u, p.major * 10 + p.minor >= 120 ? "" : "  [sm_120a fp4/fp8 kernels need ARCH=86 software MX here]");
    }
    return n;
}
