/* src/labcomp.h: label codecs for the stored training labels (2026-11-15, branch labcomp-20261115; DESIGN.md "Label compression").
   One 128^3 uint8 chunk per stream, the zarr v3 inner codec "labcomp" (inside sharding_indexed, before crc32c), so a store is an
   ordinary sharded array whose readers decode through zarr3.c. Stream: "LBC" + kind byte + payload.
     kind 1, LCS (winding codes rasters, exact): the support (code != 0) as volcomp's lossless mask (mode 5, the 12-neighbour context
       coder), then every nonzero voxel in raster order from its 13 causal 26-neighbours: "equals their majority code" (adaptive binary,
       context: agreement x neighbour count), else "equals the second most frequent", else the byte through an adaptive bit tree. A sheet
       carries one code over its whole patch, so 96-100 % of the sheet voxels cost ~0.01 bit: the stream is the sheet geometry.
   All coders are exact and deterministic (integer range coder, LZMA style, 15-bit probabilities). */
#pragma once
#include <stddef.h>
#include <stdint.h>

enum { LBC_KIND_CODES = 1 };
enum { LBC_CHUNK = 128, LBC_VOXELS = 128 * 128 * 128 };

/* encode one chunk (z-major 128^3); *out malloc'd. 0 ok, -1 error (allocation) */
int lbc_codes_encode(const uint8_t *src, uint8_t **out, size_t *out_n);
/* decode any labcomp stream into dst (LBC_VOXELS bytes). 0 ok, -1 corrupt / unknown kind */
int lbc_decode(const void *enc, size_t n, uint8_t *dst);
/* the kind byte of a stream, or -1 if it is not a labcomp stream */
int lbc_kind(const void *enc, size_t n);
/* the inner codec object for zarr.json */
const char *lbc_codec_json(int kind);
