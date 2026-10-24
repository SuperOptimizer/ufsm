/* sampler throughput without a GPU step: bench_sampler <sources.json> [P] [workers] [batches] (env UFSM_SAMPLER_PROF=1 for stages)
   env UFSM_BENCH_TRAIN=1: the trainer's surface recipe (B 1, --soft 7 --soft-unit um --erode 1 --zfix 1 --symmetry-p 0.5 --det 1,
   seed 1, fp16 input); UFSM_BENCH_COVER=plan.json: draw the cover's tiles (as train --cover); UFSM_BENCH_HASH=1: print a hash of
   every batch's input, targets, mask and weights (deterministic mode: the same for any worker count; byte-identity checks);
   UFSM_BENCH_AUG=N: augment mode; UFSM_BENCH_DET=1: deterministic default recipe; UFSM_BENCH_SOFT=S: soft sigma;
   UFSM_BENCH_SIDE=2: the winding phase of band sources (task unwrap), hashed too; UFSM_BENCH_LEVELS=p0,p1,...: level draw weights;
   UFSM_BENCH_PAPYRUS=sigma,tau_lo,tau_hi,clean: task papyrus labels (cutoffs U(50, 128), conditioned); UFSM_BENCH_SCAN=P: the
   papyrus recipe's scanner ops (noise cnoise blur lowres sharpen ring slice quant) at probability P;
   UFSM_BENCH_DUMP=prefix: mid-z planes of the first batch (prefix-input.pgm, -target.pgm, -mask.pgm, -phase.pgm with SIDE=2) and its
   cutoff code; UFSM_BENCH_PZ=N: N x P x P slab windows; UFSM_BENCH_NEAR=X: the phase only within X um of a labelled wrap (train
   --phase-near-um; the known-phase fraction is printed); UFSM_BENCH_PRESET=all|none: train --aug's preset (sample_aug_preset);
   UFSM_BENCH_PACK=1: packed targets (train --pack-targets); UFSM_BENCH_WIND4=1: --task unwrap2's winding mod 4 (with SIDE=2; hashed);
   UFSM_BENCH_GEOM=rotate_p,zoom_p,elastic_p[,zoom_out_p]: geometry augmentation probabilities; UFSM_BENCH_DUMPN=N: dump the first N batches (prefix-NN-*.pgm; the
   target as recto 255 / verso 128 / void 0, the mask, and the input's mid-z plane); UFSM_BENCH_SLOTS=N: sampler slots. A synthetic band + phase
   source in the desk's unwrap2 layout: tests/make_wind_fixture.c. UFSM_BENCH_SCROLLAUG=aug=p[,..]: xscan / tear / tear_um / crush (per-item loader cost) */
#include "nn.h"
#include "sample.h"
#include "sources.h"
#include "cover.h"
#include "papyrus.h"
#include "scan_augment.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t hash(uint64_t h, const void *p, size_t n) {
    const unsigned char *c = p; size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, c + i, 8); h = (h ^ w) * 0x100000001b3ull; h ^= h >> 29; }
    for (; i < n; i++) h = (h ^ c[i]) * 0x100000001b3ull;
    return h;
}
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: bench_sampler <sources.json> [P] [workers] [batches]\n"); return 2; }
    int P = argc > 2 ? atoi(argv[2]) : 128, W = argc > 3 ? atoi(argv[3]) : 16, NB = argc > 4 ? atoi(argv[4]) : 64;
    sources *S = sources_load(argv[1]); if (!S) return 1;
    sample_cfg c = sample_cfg_default(); c.P = P; c.B = 2; c.nworkers = W; c.nbuf = 2 * W; c.soft = 3; c.augment = 4; c.xfmt = 1;
    if (getenv("UFSM_BENCH_TRAIN")) { c.B = 1; c.soft = 7; c.soft_um = 1; c.erode = 1; c.augment = 3; c.symmetry_p = 0.5f; c.deterministic = 1; c.seed = 1; }
    if (getenv("UFSM_BENCH_SLOTS")) c.nbuf = atoi(getenv("UFSM_BENCH_SLOTS"));   /* slots (default 2 x workers: at P512 ~1 GB pinned each) */
    if (getenv("UFSM_BENCH_AUG")) c.augment = atoi(getenv("UFSM_BENCH_AUG"));
    if (getenv("UFSM_BENCH_CTAUG")) c.ct_augment = atoi(getenv("UFSM_BENCH_CTAUG"));   /* train --ct-aug */
    if (getenv("UFSM_BENCH_DET")) { c.deterministic = 1; c.seed = 1; }
    if (getenv("UFSM_BENCH_SOFT")) c.soft = (float)atof(getenv("UFSM_BENCH_SOFT"));
    if (getenv("UFSM_BENCH_LEVELS")) { char *t = strdup(getenv("UFSM_BENCH_LEVELS")); int l = 0; memset(c.level_p, 0, sizeof c.level_p); for (char *q = strtok(t, ","); q && l < MAXLEV; q = strtok(nullptr, ",")) c.level_p[l++] = atof(q); free(t); }   /* as train --levels */
    if (getenv("UFSM_BENCH_SIDE")) c.side = atoi(getenv("UFSM_BENCH_SIDE"));
    if (getenv("UFSM_BENCH_NEAR")) c.phase_near_um = (float)atof(getenv("UFSM_BENCH_NEAR"));   /* --phase-near-um */
    if (getenv("UFSM_BENCH_PZ")) c.Pz = atoi(getenv("UFSM_BENCH_PZ"));
    if (getenv("UFSM_BENCH_MINFG")) c.min_fg = atof(getenv("UFSM_BENCH_MINFG"));   /* train --min-fg */
    if (getenv("UFSM_BENCH_GPUIN")) c.gpu_input = 1;   /* train --gpu-input: the CT channel and radial records only (hashed as such) */
    if (getenv("UFSM_BENCH_PRESET") && sample_aug_preset(&c, getenv("UFSM_BENCH_PRESET"))) { fprintf(stderr, "UFSM_BENCH_PRESET: all | none\n"); return 2; }
    if (getenv("UFSM_BENCH_PACK")) c.pack_tm = 1;
    if (getenv("UFSM_BENCH_WIND4")) c.wind4 = 1;   /* train --task unwrap2 (with UFSM_BENCH_SIDE=2): the winding mod 4 and its component, hashed too */
    if (getenv("UFSM_BENCH_SCROLLAUG")) {   /* aug=p[,aug=p..]: the scroll augmentations (train --aug-xscan / --aug-tear / ...; crush: --aug-crush) */
        for (const char *q = getenv("UFSM_BENCH_SCROLLAUG"); q && *q;) {
            char key[32]; double v; int nc = 0;
            if (sscanf(q, "%31[^=]=%lf%n", key, &v, &nc) != 2) { fprintf(stderr, "UFSM_BENCH_SCROLLAUG=aug=p[,aug=p..]\n"); return 2; }
            if (!strcmp(key, "xscan")) c.xscan_p = (float)v; else if (!strcmp(key, "tear")) c.tear_p = (float)v; else if (!strcmp(key, "tear_um")) c.tear_um = (float)v;
            else if (!strcmp(key, "crush")) c.crush.p = (float)v;
            else { fprintf(stderr, "UFSM_BENCH_SCROLLAUG: unknown aug %s\n", key); return 2; }
            q += nc; if (*q == ',') q++;
        }
    }
    if (getenv("UFSM_BENCH_GEOM")) {   /* rotate_p,zoom_p,elastic_p[,zoom_out_p]: the geometry augmentation's probabilities (train --rotate-p ...) */
        float q[4] = {0, 0, 0, -1};
        if (sscanf(getenv("UFSM_BENCH_GEOM"), "%f,%f,%f,%f", &q[0], &q[1], &q[2], &q[3]) < 3) { fprintf(stderr, "UFSM_BENCH_GEOM=rotate_p,zoom_p,elastic_p[,zoom_out_p]\n"); return 2; }
        c.geometry_augment = 1; if (c.rotate_degrees <= 0) c.rotate_degrees = 10; if (c.elastic <= 0) c.elastic = 1; if (c.zoom_min <= 0 || c.zoom_min >= 1) c.zoom_min = .8f;
        c.rotate_p = q[0]; c.zoom_p = q[1]; c.elastic_p = q[2]; if (q[3] >= 0) c.zoom_out_p = q[3];
    }
    if (getenv("UFSM_BENCH_OP")) {   /* name[,name..]: only these scanner ops, at p 1 (per-op cost); "geometry": warps every window */
        memset(c.scan.p, 0, sizeof c.scan.p); c.scan.strength = 1; char *t = strdup(getenv("UFSM_BENCH_OP"));
        for (char *q = strtok(t, ","); q; q = strtok(nullptr, ",")) {
            int k = 0; while (k < SA_N && strcmp(q, scan_aug_names[k])) k++;
            if (k < SA_N) c.scan.p[k] = 1;
            else if (!strcmp(q, "geometry")) { c.geometry_augment = 1; c.rotate_degrees = 10; c.rotate_p = 1; c.elastic = 1; c.elastic_p = 1; }
            else { fprintf(stderr, "UFSM_BENCH_OP: unknown op %s\n", q); return 2; }
        }
        free(t);
    }
    const int Pz = c.Pz > 0 ? c.Pz : P;   /* task unwrap's phase (no band affinities): the side array is hashed too */
    if (getenv("UFSM_BENCH_COVER")) {
        cover_plan *plan = cover_load(getenv("UFSM_BENCH_COVER"), S, P); if (!plan) return 1;
        c.B = 1; c.cover = plan; c.deterministic = 1; c.snap = 0; memset(c.level_p, 0, sizeof c.level_p); c.level_p[0] = 1;
    }
    papyrus_cfg pc = {.t_lo = 50, .t_hi = 128, .cond = 1};
    if (getenv("UFSM_BENCH_PAPYRUS")) {
        if (sscanf(getenv("UFSM_BENCH_PAPYRUS"), "%f,%f,%f,%d", &pc.sigma, &pc.tau_lo, &pc.tau_hi, &pc.clean) != 4) { fprintf(stderr, "UFSM_BENCH_PAPYRUS=sigma,tau_lo,tau_hi,clean\n"); return 2; }
        c.papyrus = &pc; c.soft = 0;
    }
    if (getenv("UFSM_BENCH_SCAN")) {
        const float p = (float)atof(getenv("UFSM_BENCH_SCAN")); c.scan.strength = 1;
        const int ops[] = {SA_NOISE, SA_CNOISE, SA_BLUR, SA_LOWRES, SA_SHARPEN, SA_RING, SA_SLICE, SA_QUANT};
        for (size_t k = 0; k < sizeof ops / sizeof ops[0]; k++) c.scan.p[ops[k]] = p;
    }
    const int hashing = getenv("UFSM_BENCH_HASH") != nullptr;
    const size_t p3 = (size_t)Pz * P * P;
    uint64_t h = 0xcbf29ce484222325ull;
    sampler *sp = sampler_start(S, &c); if (!sp) return 1;
    if (getenv("UFSM_BENCH_DUMP")) {   /* mid-z planes of the first batch(es): input (z-scored CT x 40 + 128), target, mask */
        const int nd = getenv("UFSM_BENCH_DUMPN") ? atoi(getenv("UFSM_BENCH_DUMPN")) : 1;
        for (int bi = 0; bi < nd; bi++) {
            batch *b = sampler_next(sp); if (!b) return 1;
            const char *pre0 = getenv("UFSM_BENCH_DUMP"); char pre[900];
            if (nd > 1) snprintf(pre, sizeof pre, "%s-%02d", pre0, bi); else snprintf(pre, sizeof pre, "%s", pre0);
            const size_t pl = (size_t)P * P, z0 = (size_t)(Pz / 2) * pl;
            uint8_t *img = malloc(pl); char fn[1024];
            for (int k = 0; k < (b->side ? 4 : 3); k++) {
                for (size_t j = 0; j < pl; j++) {
                    const uint8_t code = b->tm ? b->tm[z0 + j] : 0;
                    const int t0 = b->tm ? (code & SAMPLE_TM_T0) != 0 : b->t[z0 + j] > 127, t1 = b->tm ? (code & SAMPLE_TM_T1) != 0 : b->t[p3 + z0 + j] > 127;
                    const int mk = b->tm ? (code & SAMPLE_TM_M) != 0 : b->m[z0 + j] != 0;
                    if (k == 0) { _Float16 h; memcpy(&h, b->x16 + z0 + j, 2); const float v = 128.f + 40.f * (float)h; img[j] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }
                    else if (k == 3) img[j] = b->side[z0 + j] < 252 ? (uint8_t)(1 + b->side[z0 + j]) : 0;   /* phase codes 0..251 (0: unknown) */
                    else if (k == 1) img[j] = !mk ? 0 : t0 ? 255 : t1 ? 128 : 40;   /* recto 255, verso 128, void 40, unsupervised 0 */
                    else img[j] = (uint8_t)(mk ? 255 : 0);
                }
                snprintf(fn, sizeof fn, "%s-%s.pgm", pre, k == 0 ? "input" : k == 1 ? "target" : k == 2 ? "mask" : "phase");
                FILE *f = fopen(fn, "wb"); if (f) { fprintf(f, "P5\n%d %d\n255\n", P, P); fwrite(img, 1, pl, f); fclose(f); }
            }
            int warped = 0; if (b->rad) { int32_t hd[8]; memcpy(hd, b->rad, sizeof hd); warped = hd[6] & 1; }
            printf("dumped %s-*.pgm: level %d um / cutoff code %g, corner %lld %lld %lld%s\n", pre, b->level[0], b->um[0], (long long)b->corner[0][0],
                   (long long)b->corner[0][1], (long long)b->corner[0][2], warped ? ", warped" : "");
            free(img); sampler_release(sp, b);
        }
    }
    for (int i = 0; i < 4; i++) { batch *b = sampler_next(sp); if (!b) return 1; if (hashing) { h = hash(h, b->x16, (size_t)c.B * b->xch * p3 * 2); if (b->rad) h = hash(h, b->rad, (size_t)c.B * NN_RAD_REC(Pz)); if (b->tm) h = hash(h, b->tm, (size_t)c.B * p3); else { h = hash(h, b->t, (size_t)c.B * NCH * p3); h = hash(h, b->m, (size_t)c.B * p3); } h = hash(h, b->w, (size_t)c.B * NCH); if (b->side) h = hash(h, b->side, (size_t)c.B * p3); if (b->w4) { h = hash(h, b->w4, (size_t)c.B * p3); h = hash(h, b->wcomp, (size_t)c.B * p3); } } sampler_release(sp, b); }   /* warm-up */
    size_t side_known = 0, side_all = 0;   /* phase voxels known (UFSM_BENCH_SIDE=2) */
    double t0 = now();
    for (int i = 0; i < NB; i++) {
        batch *b = sampler_next(sp); if (!b) return 1;
        if (b->side) for (size_t k = 0; k < (size_t)c.B * p3; k++) { side_known += b->side[k] < 252; side_all++; }
        if (hashing) { h = hash(h, b->x16, (size_t)c.B * b->xch * p3 * 2); if (b->rad) h = hash(h, b->rad, (size_t)c.B * NN_RAD_REC(Pz)); if (b->tm) h = hash(h, b->tm, (size_t)c.B * p3); else { h = hash(h, b->t, (size_t)c.B * NCH * p3); h = hash(h, b->m, (size_t)c.B * p3); } h = hash(h, b->w, (size_t)c.B * NCH); if (b->side) h = hash(h, b->side, (size_t)c.B * p3); if (b->w4) { h = hash(h, b->w4, (size_t)c.B * p3); h = hash(h, b->wcomp, (size_t)c.B * p3); } }
        sampler_release(sp, b);
    }
    double dt = now() - t0;
    if (side_all) printf("phase known in %.4f of the voxels\n", (double)side_known / side_all);
    uint64_t prod, rej; sampler_stats(sp, &prod, &rej);
    printf("%dx%dx%d workers=%d: %.1f samples/s (%d batches of %d in %.1fs; produced %llu rejected %llu)\n", Pz, P, P, W, NB * c.B / dt, NB, c.B, dt, (unsigned long long)prod, (unsigned long long)rej);
    if (hashing) printf("hash %016llx over %d batches\n", (unsigned long long)h, NB + 4);
    { uint64_t why[4]; sampler_reject_reasons(sp, why); printf("rejected: coarse probe %llu, after the CT read %llu, empty targets %llu; holdout overlaps %llu\n",
                                                        (unsigned long long)why[0], (unsigned long long)why[1], (unsigned long long)why[2], (unsigned long long)why[3]); }
    sampler_prof_print(sp);
    sampler_stop(sp);
    return 0;
}
