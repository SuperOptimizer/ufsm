/* Minimal TIFF reader for the upstream data: baseline classic TIFF (both byte orders), strips or tiles,
   compression none / LZW / deflate, predictor none / horizontal / floating point, 8/16-bit unsigned
   and 32-bit float samples, one or more samples per pixel (chunky). Multi-page stacks (Kaggle cubes)
   are exposed as pages. Everything is read from a memory buffer. */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct tiff tiff;
typedef struct { int w, h, bits, spp, fmt; /* fmt 1 = unsigned int, 3 = IEEE float */ } tiff_page;

tiff *tiff_open_mem(const uint8_t *buf, size_t n);        /* buffer must outlive the handle */
tiff *tiff_open_file(const char *path);                    /* reads the whole file */
tiff *tiff_open_lazy(const char *path);                    /* maps the file for its IFDs; strips are read on demand (pread) */
void tiff_close(tiff *t);
int tiff_npages(const tiff *t);
int tiff_page_info(const tiff *t, int page, tiff_page *info);
/* Decode a page into out (w*h*spp samples, native byte order, row-major). out must hold
   w*h*spp*(bits/8) bytes. Returns 0 on success. */
int tiff_read_page(tiff *t, int page, void *out);
/* Return the ImageDescription tag text of a page (nullptr if absent); owned by the handle. */
const char *tiff_description(tiff *t, int page);
const char *tiff_error(void);
/* Strip access of a strip (not tiled) page: rows per strip and the strip count; then strip si decoded into out (rps rows, the last
   one shorter; w*spp*(bits/8) bytes per row). scratch / cap: the caller's buffer for the compressed bytes of a lazy file (grown as
   needed, freed by the caller), so several threads can read one handle. */
int tiff_strip_rows(const tiff *t, int page, int *rps, int *nstrips);
int tiff_read_strip(const tiff *t, int page, int si, void *out, uint8_t **scratch, size_t *cap);
