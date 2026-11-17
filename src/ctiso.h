#pragma once
/* the per-scan air / papyrus iso from a CT histogram (src/ctiso.c, ufsm ct-iso) */
typedef struct { int air, pap, valley, mid; double depth; } ct_iso_t;
int ct_iso_from_hist(const double h[256], ct_iso_t *r);   /* 0: found; 1: no air maximum in [32, 90] (fields -1) */
