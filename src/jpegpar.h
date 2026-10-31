/* Multithreaded baseline JPEG encoder whose output is byte-identical to libjpeg-turbo 2.1.5's jpeg_set_defaults +
   jpeg_set_quality(q, TRUE) encode (JFIF, 4:2:0 YCbCr for RGB / one component for gray, standard Huffman tables, JDCT_ISLOW, no
   restart markers, no optimize_coding). The headers come from libjpeg itself (a 16 x 16 encode with the same parameters, the SOF
   size patched); the scan is coded in bands of MCU rows on threads (each band's DC predictors seeded from the previous band's last
   MCU, computed directly) and the bands' bit streams joined in order with the 0xFF00 stuffing. Unusual parameters (and, unless
   jpar_force_own, small images) go through libjpeg. */
#pragma once
#include <stdint.h>
#include <stdio.h>

/* pixels [x0, x0 + n) of row y (nc bytes each, interleaved) into dst; called from the encoder's threads at once (must be thread-safe) */
typedef void (*jpar_rows_fn)(void *u, int y, int x0, int n, uint8_t *dst);

/* w x h, nc = 1 (gray) or 3 (RGB): rows from px (w * nc bytes per row) or, when px is nullptr, from fn(u, ...). threads <= 0: the
   process-wide setting (jpar_threads). Writes the whole file to f; 0 on success. */
int jpar_encode(FILE *f, int w, int h, int nc, int quality, const uint8_t *px, jpar_rows_fn fn, void *u, int threads);
int jpar_write(const char *path, int w, int h, int nc, int quality, const uint8_t *px, jpar_rows_fn fn, void *u, int threads);

/* process-wide default thread count: UFSM_JPEG_THREADS, else the online CPUs (at most 32); jpar_set_threads(n > 0) overrides
   (1 = single-threaded, for callers that already run many encodes at once) */
void jpar_set_threads(int n);
int jpar_threads(void);
/* tests: 1 = our encoder even for small images (parameters libjpeg-only still fall back) */
extern int jpar_force_own;
extern _Thread_local int jpar_last_own;   /* tests: 1 when the last jpar_encode on this thread used our encoder, 0 = libjpeg */
