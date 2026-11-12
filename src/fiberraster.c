/* ufsm fiber-raster <fibres.afv | fibres.fib> OUT --um U (--shape Z,Y,X | --ct ROOT KEY [--iso 64]) [--level 0] [--tr T.json | --map reg.json]
       [--zband Z0,Z1] [--radius 2] [--guard 3] [--ring 0] [--extra A.afv[@map],B.afv[@map]..] [--resume 0|1] [--shard 512] [--threads 4]
   Training labels for fibre-family mask heads from fibre polylines (an AFV, or UFIB level-0 points): two prob254 volumes on the level-L
   CT grid (U = that grid's voxel size), OUT/H/<U> and OUT/V/<U> (zarr v3, lossless, fill 255 = unknown), usable as the "recto" and
   "sheet" targets of a sources entry ({"root": OUT/H, "encoding": "prob254"}).
   Per voxel, with dH / dV its distance to the nearest H / V fibre axis (level voxels):
     plane H: 254 where dH <= radius and dV > radius; 0 where dV <= radius and dH > radius (the other family: the contrast the head has
              to learn); 0 where the CT is below --iso (air; only with --ct); 255 (no loss) elsewhere - within radius of both, in the
              guard ring radius .. radius + guard about any fibre, near a fibre without a family, and in all papyrus no tracing covers
              (the published tracings hold fibres > 1 mm only, 11-13 % of the papyrus within 4 voxels: untraced is not "no fibre");
     plane V: the same with H and V exchanged.
   --ring K (2026-11-11): voxels farther than radius + guard from every fibre but within radius + guard + K of one are 0 in both planes
   ("no fibre axis here": the net learns tubes, not the fibre layer; with ring 0 it marked 84 % of the papyrus H or V, DESIGN.md 2026-11-11),
   except within radius + guard of any fibre of --extra (an all-lengths AFV of the same scan: untraced short fibres stay unknown).
   --extra takes up to 4 comma-separated AFVs (2026-11-11): an entry is in the CT's own level-0 voxels, or, suffixed "@map", in the
   --map's coarse frame (e.g. Paris 4: the 2023 all-lengths tracing through the reg map plus the native 2.4 um Paris4_v1 chains).
   Shards (--shard^3) are independent; --zband limits the z range written (level voxels, widened to whole shards). CPU. */
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "afv.h"
#include "fibregpu.h"
#include "reg.h"
#include "store.h"
#include "z3w.h"
#include "zarr3.h"

static const char *fr_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 4; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static int fr_opti(int argc, char **argv, const char *k) { for (int i = 4; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return i; return 0; }
typedef struct {
    afv *A, *X[4]; int nX, xmap[4]; const fib_set *F, *XF; double (*fbb)[6], (*xbb)[6]; pthread_mutex_t mu; int has_tr; double M[3][4], Mi[3][3]; double sc; const rmap *map;   /* --map: a ufsm reg map (coarse = the AFV's scan) instead of --tr */
    int64_t shape[3]; int S, iso, resume; double rad, guard, ring; z3 *ct; z3w *w[2]; char dir[2][2200]; atomic_long skipped;
    int64_t s0[3], ns[3]; atomic_long next; atomic_int fail; atomic_llong cnt[4];   /* H 254, H 0, V 254, V 0 (voxels) */
} fr_job;
/* distance code: 16 x voxels, 255 = farther than radius + guard */
static void fr_stamp(uint8_t *d, const int64_t so[3], const int64_t n[3], const double p[3], double R) {
    const int64_t lo[3] = {(int64_t)ceil(p[0] - R), (int64_t)ceil(p[1] - R), (int64_t)ceil(p[2] - R)}, hi[3] = {(int64_t)floor(p[0] + R), (int64_t)floor(p[1] + R), (int64_t)floor(p[2] + R)};
    for (int64_t z = lo[0] < so[0] ? so[0] : lo[0]; z <= hi[0] && z < so[0] + n[0]; z++) for (int64_t y = lo[1] < so[1] ? so[1] : lo[1]; y <= hi[1] && y < so[1] + n[1]; y++) {
        uint8_t *row = d + ((size_t)(z - so[0]) * n[1] + (size_t)(y - so[1])) * n[2]; const double dz = z - p[0], dy = y - p[1], a = dz * dz + dy * dy;
        for (int64_t x = lo[2] < so[2] ? so[2] : lo[2]; x <= hi[2] && x < so[2] + n[2]; x++) { const double dx = x - p[2], dd = a + dx * dx; if (dd > R * R) continue;
            const int c = (int)(16 * sqrt(dd)); if (c < row[x - so[2]]) row[x - so[2]] = (uint8_t)c; }
    }
}
/* the H / V planes of one box (so, n; level voxels): every fibre step within radius + guard + ring of the box stamped, the planes from the
   distance codes and the CT air test. Box-local: a voxel depends only on the fibre steps within that reach of it and its own CT, so any
   box gives the bytes the shard raster gives (the derived fibre arrays read windows through it). d: 4 scratch planes, ct: one, n
   voxels each; o0 / o1 the planes (n voxels). Returns 1 when anything was stamped or the CT read, 0 when all unknown (o0 / o1 unset),
   -1 on a failure. c (optional): the four voxel counts added. */
/* the steps of a box (thread-local, grown): fr_box collects them, then stamps on the CPU or hands them to the GPU */
static thread_local struct { fr_step *s; long n, cap; } t_st;
static int fr_emit(int fam, const double u[3]) {
    if (t_st.n == t_st.cap) { const long c = t_st.cap ? 2 * t_st.cap : 65536; fr_step *q = realloc(t_st.s, (size_t)c * sizeof *q); if (!q) return -1; t_st.s = q; t_st.cap = c; }
    fr_step *q = &t_st.s[t_st.n++]; q->p[0] = u[0]; q->p[1] = u[1]; q->p[2] = u[2]; q->fam = fam; q->pad = 0; return 0;
}
static int g_fr_gpu = -1;
static double fr_t(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int g_fr_timing = -1;
static thread_local int t_fr_nth = 1;   /* the CT read's threads (the onfly read's caller's) */   /* the fibre planes on the GPU: -1 not yet decided (UFSM_FIBRE_CPU=1: never), 0 no, 1 yes */
int fiber_onfly_gpu(int on);
int fiber_onfly_config(int gpu, int edge) {
    const char *e = getenv("UFSM_FIBRE_CPU");
    if (gpu < 0 || (e && atoi(e))) { fiber_onfly_gpu(0); return 0; }
    if (!fibre_gpu_box || !fibre_gpu_device || !fibre_gpu_reserve) return -1;
    fibre_gpu_device(gpu); g_fr_gpu = 1; return fibre_gpu_reserve(edge) ? -1 : 1;
}
int fiber_onfly_gpu(int on) { g_fr_gpu = on && fibre_gpu_box ? 1 : 0; return g_fr_gpu; }
static int fr_box(fr_job *J, const int64_t so[3], const int64_t n[3], uint8_t *d[4], uint8_t *ct, uint8_t *o0, uint8_t *o1, long long *c) {
    if (g_fr_timing < 0) g_fr_timing = getenv("UFSM_FIBRE_TIMING") != nullptr;
    const double ft0 = g_fr_timing ? fr_t() : 0; double ft1 = 0, ft2 = 0;
    const size_t nv = (size_t)n[0] * n[1] * n[2]; const double R = J->rad + J->guard + J->ring;
    const double lo[3] = {(double)so[0] - R, (double)so[1] - R, (double)so[2] - R}, hi[3] = {(double)(so[0] + n[0]) + R, (double)(so[1] + n[1]) + R, (double)(so[2] + n[2]) + R};
    afv_run *Rn = nullptr; int nr = 0, own = 0;
    if (J->A) {
        double qlo[3] = {1e30, 1e30, 1e30}, qhi[3] = {-1e30, -1e30, -1e30};
        for (int c8 = 0; c8 < 8; c8++) { const double g[3] = {((c8 & 1 ? hi[2] : lo[2]) + 0.5) * J->sc - 0.5, ((c8 & 2 ? hi[1] : lo[1]) + 0.5) * J->sc - 0.5, ((c8 & 4 ? hi[0] : lo[0]) + 0.5) * J->sc - 0.5}; double a[3];
            if (J->map) rmap_f2c(J->map, g, a); else for (int k = 0; k < 3; k++) a[k] = J->has_tr ? J->M[k][0] * g[0] + J->M[k][1] * g[1] + J->M[k][2] * g[2] + J->M[k][3] : g[k];
            for (int k = 0; k < 3; k++) { if (a[k] < qlo[k]) qlo[k] = a[k]; if (a[k] > qhi[k]) qhi[k] = a[k]; } }
        pthread_mutex_lock(&J->mu); nr = afv_query(J->A, qlo, qhi, &Rn); pthread_mutex_unlock(&J->mu);
        if (nr < 0) return -1;
        own = 1;
        if (g_fr_timing) { long np = 0; for (int r = 0; r < nr; r++) np += Rn[r].n; fprintf(stderr, "  fibre query: %d runs, %ld points in %.0f ms\n", nr, np, (fr_t() - ft0) * 1e3); }
    } else {
        Rn = calloc((size_t)J->F->n + 1, sizeof *Rn);
        for (int r = 0; r < J->F->n; r++) if (J->fbb[r][3] >= lo[0] && J->fbb[r][0] < hi[0] && J->fbb[r][4] >= lo[1] && J->fbb[r][1] < hi[1] && J->fbb[r][5] >= lo[2] && J->fbb[r][2] < hi[2]) {
            Rn[nr].fiber = r; Rn[nr].family = J->F->fam[r]; Rn[nr].n = J->F->off[r + 1] - J->F->off[r]; Rn[nr].xyz = nullptr; nr++; }
    }
    long stamped = 0; t_st.n = 0;
    for (int r = 0; r < nr; r++) {
        const afv_run *q = &Rn[r]; const int fam = q->family == 'H' ? 0 : q->family == 'V' ? 1 : 2; double prev[3] = {0, 0, 0};
        for (int i = 0; i < q->n; i++) {
            double p[3];
            if (own) { double g[3] = {q->xyz[3 * i], q->xyz[3 * i + 1], q->xyz[3 * i + 2]};
                if (J->map) { const double cc[3] = {g[0], g[1], g[2]}; rmap_c2f(J->map, cc, g); }
                else if (J->has_tr) { const double u[3] = {g[0] - J->M[0][3], g[1] - J->M[1][3], g[2] - J->M[2][3]}; for (int k = 0; k < 3; k++) g[k] = J->Mi[k][0] * u[0] + J->Mi[k][1] * u[1] + J->Mi[k][2] * u[2]; }
                for (int k = 0; k < 3; k++) p[k] = (g[2 - k] + 0.5) / J->sc - 0.5;
            } else { const float *g = J->F->pts + 3 * (size_t)(J->F->off[q->fiber] + i); for (int k = 0; k < 3; k++) p[k] = (g[k] + 0.5) / J->sc - 0.5; }
            /* the segment prev -> p in steps of <= 1 voxel */
            const double len = i ? sqrt(pow(p[0] - prev[0], 2) + pow(p[1] - prev[1], 2) + pow(p[2] - prev[2], 2)) : 0; const int m = i ? (int)ceil(len) : 1;
            for (int k = 1; k <= m; k++) { const double f = (double)k / m, sv[3] = {prev[0] + f * (p[0] - prev[0]), prev[1] + f * (p[1] - prev[1]), prev[2] + f * (p[2] - prev[2])};
                const double *u = i ? sv : p; if (u[0] < lo[0] || u[0] > hi[0] || u[1] < lo[1] || u[1] > hi[1] || u[2] < lo[2] || u[2] > hi[2]) continue;
                stamped++;
                if (fr_emit(fam, u)) { if (own) afv_runs_free(Rn, nr); else free(Rn); return -1; } }
            for (int k = 0; k < 3; k++) prev[k] = p[k];
        }
    }
    if (own) afv_runs_free(Rn, nr); else free(Rn);
    if (J->XF && J->ring > 0) {   /* --extra as UFIB (level-0 z, y, x) */
        for (int r = 0; r < J->XF->n; r++) { if (!(J->xbb[r][3] >= lo[0] && J->xbb[r][0] < hi[0] && J->xbb[r][4] >= lo[1] && J->xbb[r][1] < hi[1] && J->xbb[r][5] >= lo[2] && J->xbb[r][2] < hi[2])) continue;
            double prev[3] = {0, 0, 0};
            for (int i = J->XF->off[r]; i < J->XF->off[r + 1]; i++) { double p[3]; for (int k = 0; k < 3; k++) p[k] = (J->XF->pts[3 * (size_t)i + k] + 0.5) / J->sc - 0.5;
                const int first = i == J->XF->off[r]; const double len = first ? 0 : sqrt(pow(p[0] - prev[0], 2) + pow(p[1] - prev[1], 2) + pow(p[2] - prev[2], 2)); const int m = first ? 1 : (int)ceil(len);
                for (int k = 1; k <= m; k++) { const double f = (double)k / m, q[3] = {prev[0] + f * (p[0] - prev[0]), prev[1] + f * (p[1] - prev[1]), prev[2] + f * (p[2] - prev[2])};
                    const double *u = first ? p : q; if (u[0] < lo[0] || u[0] > hi[0] || u[1] < lo[1] || u[1] > hi[1] || u[2] < lo[2] || u[2] > hi[2]) continue;
                    if (fr_emit(3, u)) return -1; }
                for (int k = 0; k < 3; k++) prev[k] = p[k]; } }
    }
    for (int e = 0; e < J->nX && J->ring > 0; e++) {   /* --extra: every other traced fibre (any length / family) vetoes the negative ring within radius + guard of it */
        double qlo[3] = {1e30, 1e30, 1e30}, qhi[3] = {-1e30, -1e30, -1e30};
        if (J->xmap[e]) {   /* @map: the query box is the box's 8 corners through the map (as the main AFV) */
            for (int c8 = 0; c8 < 8; c8++) { const double g[3] = {((c8 & 1 ? hi[2] : lo[2]) + 0.5) * J->sc - 0.5, ((c8 & 2 ? hi[1] : lo[1]) + 0.5) * J->sc - 0.5, ((c8 & 4 ? hi[0] : lo[0]) + 0.5) * J->sc - 0.5}; double a[3];
                rmap_f2c(J->map, g, a); for (int k = 0; k < 3; k++) { if (a[k] < qlo[k]) qlo[k] = a[k]; if (a[k] > qhi[k]) qhi[k] = a[k]; } }
        } else for (int k = 0; k < 3; k++) { qlo[k] = (lo[2 - k] + 0.5) * J->sc - 0.5; qhi[k] = (hi[2 - k] + 0.5) * J->sc - 0.5; }
        afv_run *Xr = nullptr; pthread_mutex_lock(&J->mu); const int nx = afv_query(J->X[e], qlo, qhi, &Xr); pthread_mutex_unlock(&J->mu);
        if (nx < 0) return -1;
        for (int r = 0; r < nx; r++) { double prev[3] = {0, 0, 0};
            for (int i = 0; i < Xr[r].n; i++) { double g[3] = {Xr[r].xyz[3 * i], Xr[r].xyz[3 * i + 1], Xr[r].xyz[3 * i + 2]}, p[3];
                if (J->xmap[e]) { const double cc[3] = {g[0], g[1], g[2]}; rmap_c2f(J->map, cc, g); }
                for (int k = 0; k < 3; k++) p[k] = (g[2 - k] + 0.5) / J->sc - 0.5;
                const double len = i ? sqrt(pow(p[0] - prev[0], 2) + pow(p[1] - prev[1], 2) + pow(p[2] - prev[2], 2)) : 0; const int m = i ? (int)ceil(len) : 1;
                for (int k = 1; k <= m; k++) { const double f = (double)k / m, q[3] = {prev[0] + f * (p[0] - prev[0]), prev[1] + f * (p[1] - prev[1]), prev[2] + f * (p[2] - prev[2])};
                    const double *u = i ? q : p; if (u[0] < lo[0] || u[0] > hi[0] || u[1] < lo[1] || u[1] > hi[1] || u[2] < lo[2] || u[2] > hi[2]) continue;
                    if (fr_emit(3, u)) return -1; }
                for (int k = 0; k < 3; k++) prev[k] = p[k]; } }
        afv_runs_free(Xr, nx);
    }
    if (g_fr_timing) ft1 = fr_t();
    int have_ct = 0;
    if (J->ct) { int64_t cn[3], co[3]; int inside = 1;
        for (int k = 0; k < 3; k++) { co[k] = so[k] < 0 ? 0 : so[k]; cn[k] = (so[k] + n[k] > J->shape[k] ? J->shape[k] : so[k] + n[k]) - co[k]; if (cn[k] <= 0) inside = 0; }
        const int whole = inside && co[0] == so[0] && co[1] == so[1] && co[2] == so[2] && cn[0] == n[0] && cn[1] == n[1] && cn[2] == n[2];
        if (!whole) memset(ct, 255, nv);
        if (whole) { if (z3_read(J->ct, co, cn, ct, t_fr_nth)) { fprintf(stderr, "fiber-raster: CT read failed\n"); return -1; } }   /* (the box inside the CT: straight in) */
        else if (inside) {
            uint8_t *tmp = malloc((size_t)cn[0] * cn[1] * cn[2]);
            if (!tmp || z3_read(J->ct, co, cn, tmp, t_fr_nth)) { fprintf(stderr, "fiber-raster: CT read failed\n"); free(tmp); return -1; }
            for (int64_t z = 0; z < cn[0]; z++) for (int64_t y = 0; y < cn[1]; y++) memcpy(ct + ((size_t)(z + co[0] - so[0]) * n[1] + (y + co[1] - so[1])) * n[2] + (co[2] - so[2]), tmp + ((size_t)z * cn[1] + y) * cn[2], (size_t)cn[2]);
            free(tmp);
        }
        have_ct = 1; }
    if (g_fr_timing) ft2 = fr_t();
    if (!stamped && !have_ct) return 0;   /* all unknown */
    const int cr = (int)(16 * J->rad), cg = (int)(16 * (J->rad + J->guard)); long long cc4[4] = {0, 0, 0, 0};
    if (g_fr_gpu < 0) { const char *e = getenv("UFSM_FIBRE_CPU"); g_fr_gpu = fibre_gpu_box && !(e && atoi(e)); }
    if (g_fr_gpu && !c && !fibre_gpu_box(t_st.s, t_st.n, so, n, R, J->rad + J->guard, cr, cg, J->ring > 0, have_ct ? ct : nullptr, J->iso, o0, o1)) {
        if (g_fr_timing) fprintf(stderr, "fibre box %lldx%lldx%lld: %ld steps, %.0f ms polylines, %.0f ms CT, %.0f ms GPU\n", (long long)n[0], (long long)n[1], (long long)n[2], t_st.n, (ft1 - ft0) * 1e3, (ft2 - ft1) * 1e3, (fr_t() - ft2) * 1e3);
        return 1; }   /* (the counts: the CPU raster) */
    uint8_t *dl[4] = {nullptr, nullptr, nullptr, nullptr};   /* d nullptr (the onfly reads): scratch planes only on the CPU path */
    if (!d[0]) { for (int k = 0; k < 4; k++) if (!(dl[k] = malloc(nv))) { for (int e = 0; e < k; e++) free(dl[e]); return -1; } d = dl; }
    for (int k = 0; k < 4; k++) memset(d[k], 255, nv);
    for (long k = 0; k < t_st.n; k++) fr_stamp(d[t_st.s[k].fam], so, n, t_st.s[k].p, t_st.s[k].fam == 3 ? J->rad + J->guard : R);
    for (size_t i = 0; i < nv; i++) {
        const int h = d[0][i] <= cr, v = d[1][i] <= cr, air = have_ct && ct[i] < J->iso;
        const int dm = d[0][i] < d[1][i] ? (d[0][i] < d[2][i] ? d[0][i] : d[2][i]) : (d[1][i] < d[2][i] ? d[1][i] : d[2][i]);   /* nearest fibre, any family */
        const int any = dm <= cg;   /* within radius + guard of a fibre (the old stamp reach) */
        /* --ring: beyond the guard, out to radius + guard + ring, no fibre axis (0) unless another traced fibre (--extra) is near */
        const int neg = J->ring > 0 && dm != 255 && dm > cg && d[3][i] > cg;
        o0[i] = h && !v && d[2][i] > cr ? 254 : v && !h && d[2][i] > cr ? 0 : neg ? 0 : !any && air ? 0 : 255;
        o1[i] = v && !h && d[2][i] > cr ? 254 : h && !v && d[2][i] > cr ? 0 : neg ? 0 : !any && air ? 0 : 255;
        cc4[0] += o0[i] == 254; cc4[1] += o0[i] == 0; cc4[2] += o1[i] == 254; cc4[3] += o1[i] == 0;
    }
    for (int k = 0; k < 4; k++) free(dl[k]);
    if (c) for (int k = 0; k < 4; k++) c[k] += cc4[k];
    return 1;
}
static void *fr_worker(void *arg) {
    fr_job *J = arg; const int S = J->S; const size_t nv = (size_t)S * S * S;
    uint8_t *d[4], *o[2], *ct = J->ct ? malloc(nv) : nullptr; for (int k = 0; k < 4; k++) d[k] = malloc(nv); for (int k = 0; k < 2; k++) o[k] = malloc(nv);
    const long total = (long)(J->ns[0] * J->ns[1] * J->ns[2]);
    for (long t; (t = atomic_fetch_add(&J->next, 1)) < total && !atomic_load(&J->fail);) {
        const int64_t sh[3] = {J->s0[0] + t / (J->ns[1] * J->ns[2]), J->s0[1] + t / J->ns[2] % J->ns[1], J->s0[2] + t % J->ns[2]}, so[3] = {sh[0] * S, sh[1] * S, sh[2] * S}, n[3] = {S, S, S};
        if (J->resume) {   /* --resume 1: a shard both planes already hold (written whole, by rename) is not redone */
            int have = 1; for (int k = 0; k < 2 && have; k++) { char fp[2400]; snprintf(fp, sizeof fp, "%s/c/%lld/%lld/%lld", J->dir[k], (long long)sh[0], (long long)sh[1], (long long)sh[2]); FILE *f = fopen(fp, "rb"); have = f != nullptr; if (f) fclose(f); }
            if (have) { atomic_fetch_add(&J->skipped, 1); continue; }
        }
        long long c[4] = {0, 0, 0, 0};
        const int r = fr_box(J, so, n, d, ct, o[0], o[1], c);
        if (r < 0) { atomic_store(&J->fail, 1); break; }
        if (!r) continue;   /* all unknown: the shard is not written (fill 255) */
        for (int k = 0; k < 4; k++) atomic_fetch_add(&J->cnt[k], c[k]);
        for (int k = 0; k < 2; k++) if (z3w_write_shard(J->w[k], sh[0], sh[1], sh[2], o[k], 1)) { fprintf(stderr, "fiber-raster: %s\n", z3w_error()); atomic_store(&J->fail, 1); }
    }
    for (int k = 0; k < 4; k++) free(d[k]);
    for (int k = 0; k < 2; k++) free(o[k]);
    free(ct);
    return nullptr;
}

int cmd_fiber_raster(int argc, char **argv) {
    fr_job J = {0}; const double um = atof(fr_opt(argc, argv, "--um", "0")); long long sh[3] = {0, 0, 0}; const int ci = fr_opti(argc, argv, "--ct");
    const int has_shape = sscanf(fr_opt(argc, argv, "--shape", ""), "%lld,%lld,%lld", &sh[0], &sh[1], &sh[2]) == 3;
    if (argc < 6 || !(um > 0) || (!has_shape && !ci)) {
        fprintf(stderr, "usage: ufsm fiber-raster <fibres.afv|.fib> OUT --um U (--shape Z,Y,X | --ct ROOT KEY [--iso 64]) [--level 0] [--tr T.json | --map reg.json] [--zband Z0,Z1]\n"
                        "       [--radius 2] [--guard 3] [--shard 512] [--threads 4]   H / V fibre mask labels (prob254: OUT/H/<U>, OUT/V/<U>)\n");
        return 2;
    }
    const int level = atoi(fr_opt(argc, argv, "--level", "0")), nth = atoi(fr_opt(argc, argv, "--threads", "4")); J.S = atoi(fr_opt(argc, argv, "--shard", "512"));
    J.sc = ldexp(1.0, level); J.rad = atof(fr_opt(argc, argv, "--radius", "2")); J.guard = atof(fr_opt(argc, argv, "--guard", "3")); J.ring = atof(fr_opt(argc, argv, "--ring", "0")); J.resume = atoi(fr_opt(argc, argv, "--resume", "0")); J.iso = atoi(fr_opt(argc, argv, "--iso", "64"));
    if (J.S < 128 || J.S % 128 || !(J.rad > 0) || J.guard < 0 || J.ring < 0 || J.rad + J.guard + J.ring > 15 || nth < 1 || level < 0 || level > 8) { fprintf(stderr, "fiber-raster: bad options (--shard a multiple of 128, radius + guard + ring <= 15)\n"); return 2; }
    if (ci) {
        if (ci + 2 >= argc) { fprintf(stderr, "fiber-raster: --ct ROOT KEY\n"); return 2; }
        store *s = store_open(argv[ci + 1]); J.ct = s ? z3_open(s, argv[ci + 2], nullptr) : nullptr; if (!J.ct) { fprintf(stderr, "fiber-raster: cannot open the CT %s %s\n", argv[ci + 1], argv[ci + 2]); return 1; }
        for (int k = 0; k < 3; k++) { const int64_t v = z3_meta_of(J.ct)->shape[k]; if (has_shape && sh[k] != v) { fprintf(stderr, "fiber-raster: --shape differs from the CT's %lld %lld %lld\n", (long long)z3_meta_of(J.ct)->shape[0], (long long)z3_meta_of(J.ct)->shape[1], (long long)z3_meta_of(J.ct)->shape[2]); return 1; } sh[k] = v; }
    }
    for (int k = 0; k < 3; k++) { J.shape[k] = sh[k]; if (sh[k] < 1) { fprintf(stderr, "fiber-raster: bad shape\n"); return 2; } }
    const char *trp = fr_opt(argc, argv, "--tr", nullptr); J.has_tr = trp != nullptr;
    if (trp && fc_transform(trp, J.M, J.Mi)) { fprintf(stderr, "fiber-raster: cannot read the transform %s\n", trp); return 1; }
    const char *mapp = fr_opt(argc, argv, "--map", nullptr);   /* --map reg.json: AFV points through the reg map (affine + displacement field) */
    if (mapp) { if (trp) { fprintf(stderr, "fiber-raster: --tr or --map, not both\n"); return 2; } J.map = reg_map_load(mapp); if (!J.map) return 1; }
    fib_set F = {0};
    { FILE *f = fopen(argv[2], "rb"); char mg[4] = ""; if (!f || fread(mg, 1, 4, f) != 4) { fprintf(stderr, "fiber-raster: cannot read %s\n", argv[2]); return 1; } fclose(f);
      if (!memcmp(mg, "UFIB", 4)) {
          if (fib_load(argv[2], &F)) { fprintf(stderr, "fiber-raster: bad UFIB file %s\n", argv[2]); return 1; }
          J.F = &F; J.fbb = malloc(((size_t)F.n + 1) * sizeof *J.fbb);
          for (int r = 0; r < F.n; r++) { for (int k = 0; k < 3; k++) { J.fbb[r][k] = 1e30; J.fbb[r][3 + k] = -1e30; }
              for (int i = F.off[r]; i < F.off[r + 1]; i++) for (int k = 0; k < 3; k++) { const double v = (F.pts[3 * (size_t)i + k] + 0.5) / J.sc - 0.5; if (v < J.fbb[r][k]) J.fbb[r][k] = v; if (v > J.fbb[r][3 + k]) J.fbb[r][3 + k] = v; } }
      } else if (!(J.A = afv_open(argv[2]))) return 1; }
    fib_set XF = {0};
    if (*fr_opt(argc, argv, "--extra", "")) {   /* one UFIB, or up to 4 AFVs (comma list; "@map" = through --map) */
        char xs[4096]; snprintf(xs, sizeof xs, "%s", fr_opt(argc, argv, "--extra", "")); char *save = nullptr;
        for (char *xp = strtok_r(xs, ",", &save); xp; xp = strtok_r(nullptr, ",", &save)) {
            const size_t L = strlen(xp); const int viamap = L > 4 && !strcmp(xp + L - 4, "@map"); if (viamap) xp[L - 4] = 0;
            if (viamap && !J.map) { fprintf(stderr, "fiber-raster: --extra %s@map needs --map\n", xp); return 2; }
            FILE *f = fopen(xp, "rb"); char mg[4] = "";
            if (!f || fread(mg, 1, 4, f) != 4) { fprintf(stderr, "fiber-raster: cannot read %s\n", xp); if (f) fclose(f); return 1; } fclose(f);
            if (!memcmp(mg, "UFIB", 4)) {
                if (J.XF || J.nX || viamap) { fprintf(stderr, "fiber-raster: a UFIB --extra must be the only one (native)\n"); return 2; }
                if (fib_load(xp, &XF)) { fprintf(stderr, "fiber-raster: bad UFIB file %s\n", xp); return 1; }
                J.XF = &XF; J.xbb = malloc(((size_t)XF.n + 1) * sizeof *J.xbb);
                for (int r = 0; r < XF.n; r++) { for (int k = 0; k < 3; k++) { J.xbb[r][k] = 1e30; J.xbb[r][3 + k] = -1e30; }
                    for (int i = XF.off[r]; i < XF.off[r + 1]; i++) for (int k = 0; k < 3; k++) { const double v = (XF.pts[3 * (size_t)i + k] + 0.5) / J.sc - 0.5; if (v < J.xbb[r][k]) J.xbb[r][k] = v; if (v > J.xbb[r][3 + k]) J.xbb[r][3 + k] = v; } }
            } else {
                if (J.XF || J.nX == 4) { fprintf(stderr, "fiber-raster: at most 4 AFV --extra entries, not with a UFIB one\n"); return 2; }
                if (!(J.X[J.nX] = afv_open(xp))) return 1;
                J.xmap[J.nX++] = viamap;
            }
        }
    }
    for (int e = 0; e < J.nX; e++) if (!J.xmap[e] && J.has_tr) { fprintf(stderr, "fiber-raster: a native --extra with --tr is not supported\n"); return 2; }
    long long zb[2] = {0, sh[0]}; if (*fr_opt(argc, argv, "--zband", "") && sscanf(fr_opt(argc, argv, "--zband", ""), "%lld,%lld", &zb[0], &zb[1]) != 2) { fprintf(stderr, "fiber-raster: --zband Z0,Z1\n"); return 2; }
    if (zb[0] < 0) zb[0] = 0;
    if (zb[1] > sh[0]) zb[1] = sh[0];
    if (zb[1] <= zb[0]) { fprintf(stderr, "fiber-raster: empty --zband\n"); return 2; }
    J.s0[0] = zb[0] / J.S; J.ns[0] = (zb[1] + J.S - 1) / J.S - J.s0[0]; for (int k = 1; k < 3; k++) { J.s0[k] = 0; J.ns[k] = (J.shape[k] + J.S - 1) / J.S; }
    char lv[64], p[2200], at[5000]; z3w_level_name(um, lv, sizeof lv);
    for (int k = 0; k < 2; k++) {
        snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"fiber-raster %c\",\"encoding\":\"prob254: 254 = a %c fibre, 0 = a %c fibre%s, 255 = unknown\",\"radius\":%g,\"guard\":%g,\"ring\":%g,\"extra\":\"%s\",\"zband\":[%lld,%lld]}}",
                 "HV"[k], "HV"[k], "VH"[k], J.ct ? " or air" : "", J.rad, J.guard, J.ring, fr_opt(argc, argv, "--extra", ""), zb[0], zb[1]);
        snprintf(p, sizeof p, "mkdir -p '%s/%c'", argv[3], "HV"[k]); if (system(p)) return 1;
        snprintf(p, sizeof p, "%s/%c", argv[3], "HV"[k]); if (z3w_write_group(p, &um, 1, "fiber-raster", at)) { fprintf(stderr, "fiber-raster: %s\n", z3w_error()); return 1; }
        snprintf(p, sizeof p, "%s/%c/%s", argv[3], "HV"[k], lv); snprintf(J.dir[k], sizeof J.dir[k], "%s", p); J.w[k] = z3w_create(p, J.shape, J.S, 0, 255, at); if (!J.w[k]) { fprintf(stderr, "fiber-raster: %s\n", z3w_error()); return 1; }
    }
    pthread_mutex_init(&J.mu, nullptr); pthread_t th[64]; const int nt = nth > 64 ? 64 : nth;
    for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, fr_worker, &J);
    for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
    int bad = atomic_load(&J.fail); for (int k = 0; k < 2; k++) bad |= z3w_close(J.w[k]) != 0;
    if (bad) { fprintf(stderr, "fiber-raster: FAILED\n"); return 1; }
    if (J.resume) printf("fiber-raster: --resume: %ld shards already written, skipped (their voxels are not in the counts below)\n", (long)atomic_load(&J.skipped));
    printf("fiber-raster: %lld x %lld x %lld level-%d voxels of %g um, z %lld..%lld, radius %g guard %g: H plane %lld fibre / %lld other voxels, V plane %lld / %lld -> %s/{H,V}/%s\n",
           sh[0], sh[1], sh[2], level, um, zb[0], zb[1], J.rad, J.guard, (long long)atomic_load(&J.cnt[0]), (long long)atomic_load(&J.cnt[1]), (long long)atomic_load(&J.cnt[2]), (long long)atomic_load(&J.cnt[3]), argv[3], lv);
    if (J.A) afv_close(J.A);
    for (int e = 0; e < J.nX; e++) afv_close(J.X[e]);
    fib_free(&XF); free(J.xbb);
    fib_free(&F); free(J.fbb); if (J.ct) z3_close(J.ct);
    return 0;
}

/* ---- the fibre planes on the fly (2026-11-11): two derived arrays (zarr3.h z3_open_derived) over a CT level, computed per read box by
   fr_box from the fibre polylines, so a source needs no stored fiber-raster output. Both planes come from one computation: a reading
   thread keeps its last box, so reading V right after H (the sampler reads the aux planes of a window in turn) costs nothing. ---- */
typedef struct { fr_job J; int refs; pthread_mutex_t rmu; fib_set F, XF; } fr_ctx;
typedef struct { fr_ctx *X; int plane; } fr_plane;
static thread_local struct { const fr_ctx *X; int64_t o[3], n[3]; size_t cap; uint8_t *ct, *o2[2]; int ok, pin; } t_fr;
static int fr_onfly_read(void *vp, z3 *base, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    (void)base; fr_plane *P = vp; fr_ctx *X = P->X; const size_t nv = (size_t)n[0] * n[1] * n[2];
    if (!(t_fr.X == X && !memcmp(t_fr.o, o, sizeof t_fr.o) && !memcmp(t_fr.n, n, sizeof t_fr.n) && t_fr.ok)) {
        if (nv > t_fr.cap) {   /* (the planes pinned when the GPU path is linked: its D2H lands in them directly) */
            for (int k = 0; k < 2; k++) { if (t_fr.pin) fibre_gpu_host_free(t_fr.o2[k]); else free(t_fr.o2[k]); }
            if (t_fr.pin) fibre_gpu_host_free(t_fr.ct); else free(t_fr.ct);
            t_fr.pin = fibre_gpu_host_alloc && fibre_gpu_host_free; t_fr.cap = 0;
            t_fr.ct = t_fr.pin ? fibre_gpu_host_alloc(nv) : malloc(nv); for (int k = 0; k < 2; k++) t_fr.o2[k] = t_fr.pin ? fibre_gpu_host_alloc(nv) : malloc(nv);
            if (!t_fr.ct || !t_fr.o2[0] || !t_fr.o2[1]) return -1;
            t_fr.cap = nv; }
        t_fr.ok = 0;
        uint8_t *dn[4] = {nullptr, nullptr, nullptr, nullptr}; t_fr_nth = nthreads > 0 ? nthreads : 1;
        const int r = fr_box(&X->J, o, n, dn, X->J.ct ? t_fr.ct : nullptr, t_fr.o2[0], t_fr.o2[1], nullptr);
        if (r < 0) return -1;
        if (!r) { memset(t_fr.o2[0], 255, nv); memset(t_fr.o2[1], 255, nv); }
        int outside = 0; for (int k = 0; k < 3; k++) outside |= o[k] < 0 || o[k] + n[k] > X->J.shape[k];
        for (int64_t z = 0; outside && z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) {   /* outside the shape: the stored fill */
            if (o[0] + z >= 0 && o[0] + z < X->J.shape[0] && o[1] + y >= 0 && o[1] + y < X->J.shape[1] && o[2] + x >= 0 && o[2] + x < X->J.shape[2]) continue;
            const size_t i = ((size_t)z * n[1] + y) * n[2] + x; t_fr.o2[0][i] = t_fr.o2[1][i] = 255; }
        t_fr.X = X; memcpy(t_fr.o, o, sizeof t_fr.o); memcpy(t_fr.n, n, sizeof t_fr.n); t_fr.ok = 1;
    }
    memcpy(out, t_fr.o2[P->plane], nv);
    return 0;
}
static void fr_ctx_unref(fr_ctx *X) {
    pthread_mutex_lock(&X->rmu); const int last = --X->refs == 0; pthread_mutex_unlock(&X->rmu);
    if (!last) return;
    if (X->J.A) afv_close(X->J.A);
    for (int e = 0; e < X->J.nX; e++) afv_close(X->J.X[e]);
    fib_free(&X->F); fib_free(&X->XF); free(X->J.fbb); free(X->J.xbb); if (X->J.ct) z3_close(X->J.ct); free(X);   /* (ct: the context's own handle; each plane's base is closed with it) */
}
static void fr_plane_free(void *vp) { fr_plane *P = vp; fr_ctx *X = P->X; free(P); fr_ctx_unref(X); }
/* the fibre options of a sources "onfly" object: afv (an AFV or a UFIB file), extra ("a.afv,b.afv@map"), map / tr, radius (2), guard (3),
   ring (0), iso (64; < 0: no air test), level (the CT level; its voxel size is the planes' grid). ct: three handles on that CT level, all
   taken over (closed on failure too): ct0 / ct1 the planes' bases (shape, chunk map), ct2 the air test's reads.
   out[0] / out[1]: the H / V arrays; 0 or -1 (message on stderr). */
int fiber_onfly_open(const char *afvp, const char *extra, const char *mapp, const char *trp, double rad, double guard, double ring, int iso, int level,
                     z3 *ct0, z3 *ct1, z3 *ct2, z3 *out[2]) {
    out[0] = out[1] = nullptr;
    fr_ctx *X = calloc(1, sizeof *X); if (!X || !ct0 || !ct1 || !ct2) { free(X); z3_close(ct0); z3_close(ct1); z3_close(ct2); return -1; }
    fr_job *J = &X->J; pthread_mutex_init(&J->mu, nullptr); pthread_mutex_init(&X->rmu, nullptr);
    J->sc = ldexp(1.0, level); J->rad = rad; J->guard = guard; J->ring = ring; J->iso = iso; if (iso >= 0) J->ct = ct2; else z3_close(ct2);
    for (int k = 0; k < 3; k++) J->shape[k] = z3_meta_of(ct0)->shape[k];
    if (!(rad > 0) || guard < 0 || ring < 0 || rad + guard + ring > 15 || level < 0 || level > 8) { fprintf(stderr, "fibre onfly: radius > 0, guard / ring >= 0, radius + guard + ring <= 15, level 0..8\n"); X->refs = 1; fr_ctx_unref(X); z3_close(ct0); z3_close(ct1); return -1; }
    J->has_tr = trp != nullptr;
    if (trp && fc_transform(trp, J->M, J->Mi)) { fprintf(stderr, "fibre onfly: cannot read the transform %s\n", trp); X->refs = 1; fr_ctx_unref(X); z3_close(ct0); z3_close(ct1); return -1; }
    if (mapp) { if (trp) { fprintf(stderr, "fibre onfly: map or tr, not both\n"); X->refs = 1; fr_ctx_unref(X); z3_close(ct0); z3_close(ct1); return -1; } J->map = reg_map_load(mapp); if (!J->map) { X->refs = 1; fr_ctx_unref(X); z3_close(ct0); z3_close(ct1); return -1; } }
    { FILE *f = fopen(afvp, "rb"); char mg[4] = ""; if (!f || fread(mg, 1, 4, f) != 4) { fprintf(stderr, "fibre onfly: cannot read %s\n", afvp); if (f) fclose(f); X->refs = 1; fr_ctx_unref(X); z3_close(ct0); z3_close(ct1); return -1; } fclose(f);
      if (!memcmp(mg, "UFIB", 4)) {
          if (fib_load(afvp, &X->F)) { fprintf(stderr, "fibre onfly: bad UFIB file %s\n", afvp); X->refs = 1; fr_ctx_unref(X); z3_close(ct0); z3_close(ct1); return -1; }
          J->F = &X->F; J->fbb = malloc(((size_t)X->F.n + 1) * sizeof *J->fbb);
          for (int r = 0; r < X->F.n; r++) { for (int k = 0; k < 3; k++) { J->fbb[r][k] = 1e30; J->fbb[r][3 + k] = -1e30; }
              for (int i = X->F.off[r]; i < X->F.off[r + 1]; i++) for (int k = 0; k < 3; k++) { const double v = (X->F.pts[3 * (size_t)i + k] + 0.5) / J->sc - 0.5; if (v < J->fbb[r][k]) J->fbb[r][k] = v; if (v > J->fbb[r][3 + k]) J->fbb[r][3 + k] = v; } }
      } else if (!(J->A = afv_open(afvp))) { X->refs = 1; fr_ctx_unref(X); z3_close(ct0); z3_close(ct1); return -1; } }
    if (extra && *extra) {
        char xs[4096]; snprintf(xs, sizeof xs, "%s", extra); char *save = nullptr;
        for (char *xp = strtok_r(xs, ",", &save); xp; xp = strtok_r(nullptr, ",", &save)) {
            const size_t L = strlen(xp); const int viamap = L > 4 && !strcmp(xp + L - 4, "@map"); if (viamap) xp[L - 4] = 0;
            FILE *xf = fopen(xp, "rb"); char xm[4] = ""; const int ufib = xf && fread(xm, 1, 4, xf) == 4 && !memcmp(xm, "UFIB", 4); if (xf) fclose(xf);
            if (ufib) {   /* a UFIB extra (native level-0 z, y, x): the only one */
                if (J->XF || J->nX || viamap || fib_load(xp, &X->XF)) { fprintf(stderr, "fibre onfly: extra %s: a UFIB extra stands alone, native\n", xp); X->refs = 1; fr_ctx_unref(X); z3_close(ct0); z3_close(ct1); return -1; }
                J->XF = &X->XF; J->xbb = malloc(((size_t)X->XF.n + 1) * sizeof *J->xbb);
                for (int r = 0; r < X->XF.n; r++) { for (int k = 0; k < 3; k++) { J->xbb[r][k] = 1e30; J->xbb[r][3 + k] = -1e30; }
                    for (int q = X->XF.off[r]; q < X->XF.off[r + 1]; q++) for (int k = 0; k < 3; k++) { const double v = (X->XF.pts[3 * (size_t)q + k] + 0.5) / J->sc - 0.5; if (v < J->xbb[r][k]) J->xbb[r][k] = v; if (v > J->xbb[r][3 + k]) J->xbb[r][3 + k] = v; } }
                continue;
            }
            if ((viamap && !J->map) || J->nX == 4 || J->XF) { fprintf(stderr, "fibre onfly: extra %s: @map needs a map; at most 4 AFVs\n", xp); X->refs = 1; fr_ctx_unref(X); z3_close(ct0); z3_close(ct1); return -1; }
            if (!(J->X[J->nX] = afv_open(xp))) { X->refs = 1; fr_ctx_unref(X); z3_close(ct0); z3_close(ct1); return -1; }
            J->xmap[J->nX++] = viamap;
        }
    }
    for (int k = 0; k < 2; k++) {
        fr_plane *P = malloc(sizeof *P); P->X = X; P->plane = k; X->refs++;
        out[k] = z3_open_derived(k ? ct1 : ct0, 255, fr_onfly_read, P, fr_plane_free);
        if (!out[k]) { fr_plane_free(P); if (k) { z3_close(out[0]); out[0] = nullptr; } else z3_close(ct1); return -1; }
    }
    return 0;
}

/* ufsm fiber-onfly <fibres.afv|.fib> --ct ROOT KEY [--level 0] [--radius 2 --guard 3 --ring 0 --iso 64 --extra L --map M | --tr T]
       [--ref H_ROOT V_ROOT KEY] [--windows 8] [--P 512] [--zband Z0,Z1] [--seed 1]
   The fibre planes on the fly over random windows (level voxels, within --zband): ms per window, and with --ref the byte differences
   against stored fiber-raster planes (exit 1 on any). */
static double fr_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int cmd_fiber_onfly(int argc, char **argv) {
    int ci = 0, ri = 0; for (int i = 3; i < argc; i++) { if (!strcmp(argv[i], "--ct") && i + 2 < argc) ci = i; if (!strcmp(argv[i], "--ref") && i + 3 < argc) ri = i; }
    if (argc < 3 || !ci) { fprintf(stderr, "usage: ufsm fiber-onfly <fibres.afv|.fib> --ct ROOT KEY [--level 0] [--radius 2 --guard 3 --ring 0 --iso 64 --extra L --map M] [--ref H_ROOT V_ROOT KEY] [--windows 8] [--P 512] [--zband Z0,Z1]\n"); return 2; }
#define FO(k, d) fr_opt(argc, argv, k, d)
    const int level = atoi(FO("--level", "0")), nw = atoi(FO("--windows", "8")), P = atoi(FO("--P", "512")); uint64_t seed = (uint64_t)atoll(FO("--seed", "1"));
    store *cs = store_open(argv[ci + 1]); z3 *pl[2];
    z3 *c0 = cs ? z3_open(cs, argv[ci + 2], nullptr) : nullptr; if (!c0) { fprintf(stderr, "fiber-onfly: cannot open the CT %s %s\n", argv[ci + 1], argv[ci + 2]); return 1; }
    const int64_t *sh = z3_meta_of(c0)->shape; int64_t shp[3] = {sh[0], sh[1], sh[2]};
    const double t0 = fr_now();
    if (fiber_onfly_open(argv[2], FO("--extra", ""), FO("--map", nullptr), FO("--tr", nullptr), atof(FO("--radius", "2")), atof(FO("--guard", "3")), atof(FO("--ring", "0")), atoi(FO("--iso", "64")), level,
                         c0, z3_open(cs, argv[ci + 2], nullptr), z3_open(cs, argv[ci + 2], nullptr), pl)) return 1;
    printf("fiber-onfly: opened in %.0f ms (shape %lld %lld %lld)\n", (fr_now() - t0) * 1e3, (long long)shp[0], (long long)shp[1], (long long)shp[2]);
    z3 *ref[2] = {nullptr, nullptr};
    if (ri) for (int k = 0; k < 2; k++) { store *rs = store_open(argv[ri + 1 + k]); ref[k] = rs ? z3_open(rs, argv[ri + 3], nullptr) : nullptr; if (!ref[k]) { fprintf(stderr, "fiber-onfly: cannot open %s %s\n", argv[ri + 1 + k], argv[ri + 3]); return 1; } }
    long long zb[2] = {0, shp[0]}; if (*FO("--zband", "")) sscanf(FO("--zband", ""), "%lld,%lld", &zb[0], &zb[1]);
    const size_t nv = (size_t)P * P * P; uint8_t *a = malloc(nv), *b = malloc(nv); long long bad = 0; double tsum = 0, tmax = 0;
    for (int w = 0; w < nw; w++) {
        int64_t o[3], n[3] = {P, P, P};
        for (int k = 0; k < 3; k++) { seed = seed * 6364136223846793005ull + 1442695040888963407ull; const int64_t lo = k ? 0 : zb[0], hi = (k ? shp[k] : zb[1]) - P; o[k] = hi > lo ? lo + (int64_t)((seed >> 33) % (uint64_t)(hi - lo)) : lo; }
        long long lab[2] = {0, 0}, d = 0; double wdt = 0;
        for (int k = 0; k < 2; k++) {
            const double t1 = fr_now(); if (z3_read(pl[k], o, n, a, 1)) { fprintf(stderr, "fiber-onfly: read failed\n"); return 1; } const double dt = fr_now() - t1;
            if (!k) { tsum += dt; wdt = dt; if (dt > tmax) tmax = dt; }
            for (size_t i = 0; i < nv; i++) lab[k] += a[i] != 255;
            if (ref[k]) { if (z3_read(ref[k], o, n, b, 8)) { fprintf(stderr, "fiber-onfly: ref read failed: %s\n", z3_error()); return 1; } for (size_t i = 0; i < nv; i++) d += a[i] != b[i]; }
        }
        bad += d;
        printf("window %d (%lld,%lld,%lld): %.0f ms, %lld / %lld labelled H / V voxels", w, (long long)o[0], (long long)o[1], (long long)o[2], wdt * 1e3, lab[0], lab[1]);
        if (ref[0]) printf(", %lld bytes differ", d);
        printf("\n"); fflush(stdout);
    }
    printf("fiber-onfly: %d windows of %d^3: %.0f ms mean, %.0f ms max per window (both planes, one computation)%s\n", nw, P, tsum / nw * 1e3, tmax * 1e3,
           ref[0] ? (bad ? ": DIFFERS from the stored raster" : ": byte-identical to the stored raster") : "");
    free(a); free(b); z3_close(pl[0]); z3_close(pl[1]); z3_close(ref[0]); z3_close(ref[1]);
#undef FO
    return bad ? 1 : 0;
}
