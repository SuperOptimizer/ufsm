/* Real super-resolution pairs (ufsm sr-pairs output) for train --task restore: a sources.json entry with
   "sr_pair": {"role": "train" | "val"} and "root": the pair directory (input.zarr, target.zarr, boxes.json, intensity_map.json).
   input.zarr: n boxes of B^3 native voxels stacked along z; target.zarr: the fine scan through the registration map on the input's x4 grid,
   n boxes of (4B)^3 (0 = invalid); boxes.json: per box "split" "train" / "test"; intensity_map.json: "lut", 256 floats mapping the target
   (fine scan) intensity onto the input (native scan) intensity (monotone).
   Role "train": training windows from the "train" boxes, validation windows from the "test" boxes (held-out slabs). Role "val": every box
   validates, none trains (a held-out scroll or an energy-matched calibration pair). */
#pragma once
#include "zarr3.h"
#include <stdint.h>

typedef struct srpair {
    z3 *in, *tg;            /* input (native) and target zarr arrays */
    int B, T;               /* input box edge (native voxels) and target box edge (= 4 B) */
    int n;                  /* boxes */
    int role;               /* 0 train, 1 val */
    int *tr, ntr, *te, nte; /* box indices of the training and validation draws */
    uint8_t lut[256];       /* target byte -> input-scale byte (0 kept, else clamped to 1..255) */
    double keV;             /* the target scan's energy (0 unknown), from boxes.json's target key ("...-77keV-...") */
} srpair;

/* dir: the sr-pairs output directory; role "train" or "val"; lut 0: identity map (the targets as stored). nullptr on error (message printed) */
srpair *srpair_open(const char *dir, const char *role, int lut);
void srpair_free(srpair *p);
/* the host side of a window: target crop (P^3 at fine offset 4 oc) through the LUT and the input crop ((P / 4)^3 at native offset oc) of box k.
   P a multiple of 4, P / 4 <= B. 0 on success */
int srpair_read(const srpair *p, int k, const int oc[3], int P, uint8_t *tgt, uint8_t *coarse);
