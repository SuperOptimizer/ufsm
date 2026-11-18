/* blksync.cuh: blk_sync(st), a cudaStreamSynchronize whose waiting thread sleeps instead of spinning (DESIGN 2026-11-17 "sampler
   waits"). The sampler threads' GPU label paths (pair rule, fibre planes, mesh codes, inkfly) share card 0's context with the trainer,
   whose own syncs should stay spin-fast, so the context's schedule flag is left alone: each wait records a cudaEventBlockingSync
   event (one per thread and device, created on first use) and sleeps on it. Same errors as cudaStreamSynchronize (a fault in the
   stream is reported by the event's sync). UFSM_SPIN_SYNC=1: the plain stream sync (the A/B). */
#pragma once
#include <cuda_runtime.h>
#include <stdlib.h>

static inline cudaError_t blk_sync(cudaStream_t st)
{
    static int spin = -1;
    static thread_local struct { int dev; cudaEvent_t ev; } c[4];
    static thread_local int nc;
    int d = 0, k;
    if (spin < 0) { const char *e = getenv("UFSM_SPIN_SYNC"); spin = e && atoi(e) > 0; }
    if (spin || cudaGetDevice(&d) != cudaSuccess) return cudaStreamSynchronize(st);
    for (k = 0; k < nc && c[k].dev != d; k++) {}
    if (k == nc) {
        if (nc == 4 || cudaEventCreateWithFlags(&c[k].ev, cudaEventBlockingSync | cudaEventDisableTiming) != cudaSuccess) {
            (void)cudaGetLastError(); return cudaStreamSynchronize(st); }
        c[k].dev = d; nc++;
    }
    const cudaError_t r = cudaEventRecord(c[k].ev, st);
    return r != cudaSuccess ? r : cudaEventSynchronize(c[k].ev);
}
