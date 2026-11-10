/* latcomp v1: the learned 4^3-cell transform of the latent-grid networks (train --lat 4 --lat-basis). A cell's 64 normalised CT
   values v become coefficients c_k = <b_k, v> on an orthonormal basis: b_0 the cell mean direction (1/8 each), b_1.. the principal
   axes of the mean-removed cells of real training windows (after the augmentations), by falling variance. The coefficients are
   quantised with one step (orthonormal basis: a uniform step is the rate-optimal allocation) and are what the network reads and what
   a latent store keeps. latfit_*: the fit (train --lat-fit FILE) with its rate / distortion table. */
#ifndef UFSM_LATFIT_H
#define UFSM_LATFIT_H
#include <stddef.h>
#include <stdint.h>
typedef struct { float sigma[64], b[64][64]; } lat_basis;   /* sigma[k]: the standard deviation of coefficient k; b[k]: its 64 weights (sub-voxel order z y x) */
void latfit_acc(const uint16_t *ct16, int D, int P, int stride, int phase);   /* one window's fp16 CT channel: cells on a stride lattice */
int latfit_finish(const char *path);                                          /* solve, print the table, write the basis; 0 on success */
int lat_basis_load(const char *path, lat_basis *lb);
#endif
