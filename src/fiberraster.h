/* src/fiberraster.c: ufsm fiber-raster, and the fibre planes on the fly (derived arrays over a CT level, the raster's bytes per read box) */
#pragma once
#include "zarr3.h"
int cmd_fiber_raster(int argc, char **argv);
int cmd_fiber_onfly(int argc, char **argv);   /* ufsm fiber-onfly: the on-the-fly planes over random windows, ms and byte diffs vs a stored raster */
/* afv: an AFV or UFIB file; extra: "a.afv,b.afv@map" or ""; map / tr: nullptr or a path; iso < 0: no air test; level: the CT level of
   ct0..ct2 (three handles, all taken over). out[0] / out[1]: the H / V planes (prob254, fill 255). 0 or -1. */
int fiber_onfly_open(const char *afv, const char *extra, const char *map, const char *tr, double radius, double guard, double ring, int iso, int level,
                     z3 *ct0, z3 *ct1, z3 *ct2, z3 *out[2]);
int fiber_onfly_gpu(int on);   /* the planes' stamp and combine on the GPU (default: when nn_fibre.o is linked, UFSM_FIBRE_CPU unset); returns the mode set */
int fiber_onfly_config(int gpu, int edge);   /* the trainer: the planes on GPU gpu (-1: the CPU), buffers for edge^3 boxes reserved; 1 on the GPU, 0 the CPU (gpu < 0, UFSM_FIBRE_CPU=1), -1 no GPU path / the reserve failed */
