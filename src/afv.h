/* AFV fibre collections (community AutomatedFiberVolumes, Qualzz AVF_SPLINE_GENERATOR): a SQLite file with
     fibers(id, name, family 'H' | 'V', point_count, length, min/max x/y/z, annotation JSON),
     blocks(id, fiber_id, first_segment, points BLOB: float64 little-endian XYZ native level-0 voxels of the AFV's own scan,
            256 segments per block, so consecutive blocks share one point),
     block_bounds (rtree over the blocks).
   libsqlite3 is loaded at run time (dlopen "libsqlite3.so.0"; no build dependency). Reading only. */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct afv afv;

typedef struct {
    int64_t fiber;        /* fibers.id */
    char family;          /* 'H', 'V' or '?' */
    int n;                /* points */
    double *xyz;          /* n x 3, AFV coordinates (x, y, z) */
} afv_run;               /* one contiguous stretch of a fibre (blocks with consecutive first_segment) */

afv *afv_open(const char *path);           /* nullptr on error (message on stderr) */
void afv_close(afv *a);
/* every run of the blocks whose bounds meet the AFV-coordinate box [lo, hi] (x, y, z); whole blocks, so points may lie outside.
   Returns the run count (*out malloc'd, free with afv_runs_free) or -1. */
int afv_query(afv *a, const double lo[3], const double hi[3], afv_run **out);
void afv_runs_free(afv_run *r, int n);
/* the metadata value of key (JSON text, malloc'd) or nullptr */
char *afv_meta(afv *a, const char *key);

/* Fibre polylines prepared for one fit box (ufsm fiber-export -> fit --fibers): n runs, run r = points off[r] .. off[r + 1] - 1,
   pts (z, y, x) in the fields' box-local native voxels, fam 'H' / 'V' / '?'. File: "UFIB" int32 version 1, int32 n, int32 npts,
   int32 off[n + 1], uint8 fam[n], float32 pts[3 npts] (little-endian). */
typedef struct { int n, npts; int *off; char *fam; float *pts; } fib_set;
int fib_save(const char *path, const fib_set *f);
int fib_load(const char *path, fib_set *f);   /* 0 or -1 */
void fib_free(fib_set *f);
