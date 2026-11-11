/* Manual winding annotation (ufsm annotate-serve / annot-export / annot-regions): a local HTTP viewer of CT slices with our wraps,
   label phase and fibres as overlays, and a per-scroll annotation store (JSON snapshot + append-only JSONL log). src/annot.c. */
#pragma once
#include <stddef.h>
#include <stdint.h>

/* ---- annotation store ---------------------------------------------------------------------------------------------------------
   Items are JSON objects {"id":N,"t":unix_s,"author":"..","type":"point|pair|line|region|wraperr", ...} with level-0 voxel
   coordinates [z, y, x]:
     point   {"p":[z,y,x], "w":winding (optional), "same_as":id (optional)}
     pair    {"a":[z,y,x], "b":[z,y,x], "dw":k}            (k = winding(b) - winding(a); 0 = same sheet)
     line    {"pts":[[z,y,x],...], "w":winding (optional)}  (one sheet followed through several clicks)
     region  {"box":[z0,y0,x0,z1,y1,x1], "verdict":"good|bad|unknown"}   (half-open box: our spiral is right / wrong here)
     wraperr {"p":[z,y,x], "wrap":i, "kind":"cuts|skips|doubles|other"} (our wrap i is wrong at p)
   The store file <path> holds {"version":1,"scroll":..,"scan":..,"next_id":N,"items":[live items]}; <path>.log gets one line per
   operation ({"op":"add","item":{..}} / {"op":"del","id":N} / {"op":"undo","of":"add|del","id":N}), never rewritten. */
typedef struct annot_store annot_store;
annot_store *annot_open(const char *path, const char *scroll, const char *scan);   /* loads the snapshot if present; nullptr on a bad file */
void annot_close(annot_store *s);
int annot_add(annot_store *s, const char *item_json, const char *author, char *err, size_t errn);   /* new id (> 0) or -1 */
int annot_del(annot_store *s, int id);    /* 0 ok, -1 no such live item */
int annot_undo(annot_store *s);           /* reverts the newest add / del of this session: 0 ok, -1 nothing to undo */
char *annot_snapshot(annot_store *s);     /* the snapshot JSON text (malloc'd) */
int annot_count(annot_store *s);          /* live items */

/* ---- wrap / plane intersection --------------------------------------------------------------------------------------------------
   A wrap grid (w x h vertices, level-0 z y x per vertex, z < 0 = invalid) cut by the plane coordinate[axis] = pos: segments
   (a0, b0, a1, b1) in the plane's two other axes (axis 0: y, x; 1: z, x; 2: z, y), appended to *seg (capacity grown). Returns the
   segment count added. Segments outside [lo, hi) in both in-plane axes (lo / hi: 2 values) are dropped (hi <= lo: no limit). */
size_t annot_cut_wrap(const float *zyx, int w, int h, int axis, double pos, const double lo[2], const double hi[2], float **seg, size_t *cap,
                      size_t *n);

/* ---- the server -------------------------------------------------------------------------------------------------------------- */
typedef struct {
    const char *ct_root, *ct_key;   /* CT pyramid (the key's level subdirectories 0..7 that exist are the zoom levels) */
    double um0;                     /* level-0 voxel size (um) */
    int base;                       /* the finest level shown */
    const char *wraps;              /* wraps directory (tifxyz dirs or .sfc files), may be null */
    int wrap_stride;                /* vertex subsampling of the wraps on load (default 2) */
    const char *lab_root, *lab_key; /* label array (phase / codes) on the base level grid, may be null */
    const char *fibers;             /* UFIB file (level-0 z y x points), may be null */
    const char *store;              /* annotation store path */
    const char *regions;            /* REGIONS.tsv work queue (annot-regions output), may be null */
    const char *scroll, *author, *page, *cache;
    int port, threads;
    double cache_gb;
} annot_cfg;
/* starts the server; returns the bound port (> 0) and keeps serving in background threads, or -1. annot_serve_stop ends it. */
int annot_serve_start(const annot_cfg *c);
void annot_serve_stop(void);
