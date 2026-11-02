/* ufsm recto-ruler / recto-cls (src/rectoruler.c, src/rectocls.c): recto predictions as a winding ruler, an interval classifier */
#pragma once
#include <stdint.h>

enum { RR_NF = 25,            /* features of a hit interval (rr_feat in rectoruler.c):
                                 0 normal width um, 1 width / network spacing, 2 width / median of the +-4 neighbouring hit widths,
                                 3 density turns (integral |n.r| / sp), 4 network phase advance (turns), 5 min / 6 mean prob between (0..1),
                                 7 / 8 the end peaks' heights, 9 / 10 their prominences, 11 / 12 their half-prominence widths (um, normal),
                                 13 CT mean / 14 min / 15 max between (0..1), 16 CT peaks strictly inside, 17 / 18 the previous / next hit
                                 interval over the network spacing, 19 radius mm, 20 network spacing um, 21 |n.r| (ray obliquity),
                                 22-24 the other sources' hits strictly inside */
       RR_REC = RR_NF + 4,    /* a dump record: features, GT turns, global native z, ray, start (native voxels along the ray) */
       RR_K = 4,              /* classes: 0, 1, 2, 3+ turns */
       RR_NCOL = 29 };        /* fit, density, CT, (pkw, pk, run) per source (8), cls, clsE: pkw hits in the half-shifted window, pk / run offset-free (cumulative hit index linear between hits) */

typedef struct { float pos, h, prom, wid; } rr_hit;   /* samples along the ray; height / prominence 0..255; half-prominence width (samples) */
typedef struct { float z, r, gtq, gapn, sp; int ray, ia, ib; float t[RR_NCOL]; } rr_iv;   /* one GT interval and each source's turns */

int rr_peaks(const float *v, const uint8_t *mat, int n, double sig, double prom, int W, double sep, rr_hit *out, float *tmp);

typedef struct rr_cls rr_cls;
rr_cls *rr_cls_load(const char *path);
void rr_cls_free(rr_cls *c);
void rr_cls_predict(const rr_cls *c, const float *x, float *p);   /* p[RR_K] class probabilities */
