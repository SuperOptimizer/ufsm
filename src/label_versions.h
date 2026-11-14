/* Label-code versions (2026-11-14, the dcache audit: /vesuvius/ufsm/audit-20261114/targeted/dcache.md). Every disk cache of derived
   labels is keyed by its inputs; these constants key it by the code that computes the values too. Bump the matching constant with any
   change to that code that can change a value (not for refactors that keep every byte): every dependent key then changes and the old
   cells / packs are never read again (the derived cache's LRU evicts them). test_label_versions holds golden digests per version: a
   value change without a bump fails the fast suite.
     LBL_PHASE_PAIR       the pair rule (band_onfly.c, nn_band.cu, bandgpu.h): the phase of a codes array at rmax
     LBL_CODES_POOL       codes_pool2_open / codes_pool.h (host and device pooling of codes to the next coarser level)
     LBL_MESHCODES_LOAD   meshcodes.c mc_load / the map applied to the vertices (what the .meshcodes pack holds)
     LBL_MESHCODES_STAMP  meshcodes.c mc_stamp / nn_mcodes.cu (codes stamped from the loaded meshes)
     LBL_INKFLY           inksplat.c inkfly (the on-the-fly ink targets)
     LBL_UDC_CODEC        dcache.c UDC2 (the near-lossless phase codec: predictor, quantiser, side bits) */
#pragma once
/* (macros, not an enum: the ink key stringifies its version) */
#define LBL_PHASE_PAIR 2
#define LBL_CODES_POOL 2
#define LBL_MESHCODES_LOAD 2
#define LBL_MESHCODES_STAMP 2
#define LBL_INKFLY 3
#define LBL_UDC_CODEC 1
