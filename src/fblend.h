/* ufsm field-blend: several models' fit-field sets (predict --fit-fields / --fit-zarr, band --fit-fields) on one grid averaged
   into one set, so phase noise averages down and disagreement lowers the phase confidence (the fitter weights the phase by it).
   Per plane (weights w_i, default 1):
     pc ps        complex mean of the confidence vectors c_i e^{i 2 pi phi_i} = ((pc - 128) / 127, (ps - 128) / 127): the new phase
                  is the angle of sum w_i v_i / sum w_i and its length the new confidence (agreeing models keep it, opposing ones cancel)
     t0..t5 r     weighted mean (n n^T; recto probability)
     sp           weighted median of the known (non-zero) spacing codes (0 when no input knows it)
     w            max (material)
     dz dy dx dq  winding mod 4 relative planes: per axis the complex mean of dq_i e^{i 2 pi d_i / 256}, dq the mean resultant length
     anything else (e.g. ct, cs) is passed through from the first input that has it (the first input's plane when it has one)
   An input without a plane does not take part in that plane. */
#pragma once
#include <stddef.h>
#include <stdint.h>

/* per-cell combiners over n inputs (src[i] = input i's cells, nullptr: input i lacks the plane, skipped), ncell cells */
void fb_phase(int n, const uint8_t *const *pc, const uint8_t *const *ps, const float *w, size_t ncell, uint8_t *opc, uint8_t *ops);
void fb_mean(int n, const uint8_t *const *a, const float *w, size_t ncell, uint8_t *o);
void fb_max(int n, const uint8_t *const *a, size_t ncell, uint8_t *o);
void fb_median_known(int n, const uint8_t *const *a, const float *w, size_t ncell, uint8_t *o);
/* d[i][0..3] = input i's dz dy dx dq (d[i][0] nullptr: input i lacks them); o[0..3] the blended planes */
void fb_wind4(int n, const uint8_t *const (*d)[4], const float *w, size_t ncell, uint8_t *const o[4]);

/* blend the field directories in[0..n) (same cells: shape, h x um and origin x um equal; raw or zarr, with zarr_offset / raw_offset
   crops) into out (zarr arrays when in[0] is zarr, else raw), with fields.json from in[0] (format, recto, w4_d and a "blend" record
   adjusted); w nullptr = equal weights. Processes 128 field rows at a time. 0 on success. */
int fblend_run(const char *out, int n, const char *const *in, const float *w, int nthreads);
