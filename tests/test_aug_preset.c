/* train --aug presets on a synthetic two-level source with side labels (CPU only, plain host buffers): "none" draws exactly as the
   fields it sets (the recipe before the presets); "all" with packed targets and --gpu-input batches is reproducible across worker
   counts, its packed codes stay hard under warps (recto and verso exclusive, every warped voxel a source voxel's code), zoom-out
   windows read the coarser level (um in [1, zoom_max]), warped windows carry their warp in the radial record, and the CT channel is
   finite. */
#include "../src/sample.c"
#include "z3w.h"
#include <assert.h>
void *nn_host_alloc(size_t n) { return malloc(n); }
void nn_host_free(void *p) { free(p); }

static int fails;
static void check(const char *name, int ok) { printf("aug preset %-62s %s\n", name, ok ? "ok" : "FAIL"); fails += !ok; }
static uint64_t mix(uint64_t h, const void *p, size_t n) {
    const unsigned char *c = p;
    for (size_t i = 0; i < n; i++) h = (h ^ c[i]) * 0x100000001b3ull;
    return h;
}
enum { N0 = 128 };
/* CT: level 0 (1 um, 128^3) and level 1 (2 um, the 2^3 means); sheets along x (period 16, papyrus x % 16 in 3..12, CT 110..160,
   void 30); side labels on the 2 um grid: recto where (x % 16) < 8, else verso (void is decided by the CT cutoff) */
static void fixture(const char *dir) {
    uint8_t *v0 = malloc((size_t)N0 * N0 * N0), *v1 = calloc((size_t)N0 * N0 * N0, 1);
    for (int z = 0; z < N0; z++) for (int y = 0; y < N0; y++) for (int x = 0; x < N0; x++) {
        const int px = x % 16, pap = px >= 3 && px <= 12;
        v0[((size_t)z * N0 + y) * N0 + x] = pap ? (uint8_t)(110 + (z * 7 + y * 3 + x * 5) % 50) : 30;
    }
    const int h = N0 / 2;
    for (int z = 0; z < h; z++) for (int y = 0; y < h; y++) for (int x = 0; x < h; x++) {
        int s = 0;
        for (int d = 0; d < 8; d++) s += v0[((size_t)(2 * z + (d >> 2)) * N0 + 2 * y + (d >> 1 & 1)) * N0 + 2 * x + (d & 1)];
        v1[((size_t)z * N0 + y) * N0 + x] = (uint8_t)((s + 4) / 8);   /* (shard buffer: 128^3, the 64^3 array in its corner) */
    }
    char path[1400]; const int64_t s0[3] = {N0, N0, N0}, s1[3] = {h, h, h}; const double um[2] = {1, 2};
    snprintf(path, sizeof path, "%s/ct/1", dir); z3w *w = z3w_create(path, s0, 128, 0, 0, nullptr); assert(w);
    assert(!z3w_write_shard(w, 0, 0, 0, v0, 2) && !z3w_close(w));
    uint8_t *b1 = calloc((size_t)N0 * N0 * N0, 1);
    for (int z = 0; z < h; z++) for (int y = 0; y < h; y++) memcpy(b1 + ((size_t)z * N0 + y) * N0, v1 + ((size_t)z * N0 + y) * N0, h);
    snprintf(path, sizeof path, "%s/ct/2", dir); w = z3w_create(path, s1, 128, 0, 0, nullptr); assert(w);
    assert(!z3w_write_shard(w, 0, 0, 0, b1, 2) && !z3w_close(w));
    snprintf(path, sizeof path, "%s/ct", dir); assert(!z3w_write_group(path, um, 2, "ct", nullptr));
    memset(b1, 0, (size_t)N0 * N0 * N0);
    for (int z = 0; z < h; z++) for (int y = 0; y < h; y++) for (int x = 0; x < h; x++) b1[((size_t)z * N0 + y) * N0 + x] = (2 * x) % 16 < 8 ? RV_RECTO : RV_VERSO;
    snprintf(path, sizeof path, "%s/side/2", dir); w = z3w_create(path, s1, 128, 0, 0, nullptr); assert(w);
    assert(!z3w_write_shard(w, 0, 0, 0, b1, 2) && !z3w_close(w));
    snprintf(path, sizeof path, "%s/side", dir); assert(!z3w_write_group(path, um + 1, 1, "side", nullptr));
    snprintf(path, sizeof path, "%s/sources.json", dir);
    FILE *f = fopen(path, "w"); assert(f);
    fprintf(f, "{\"sources\":[{\"name\":\"s\",\"root\":\"%s\",\"ct\":\"ct\",\"um\":1,\"iso\":60,\"targets\":{\"recto\":{\"root\":\"%s\",\"group\":\"side\","
               "\"encoding\":\"side\",\"min_level\":1}}}]}\n", dir, dir);
    assert(!fclose(f));
    free(v0); free(v1); free(b1);
}
static sample_cfg config(void) {
    sample_cfg c = sample_cfg_default(); c.P = 32; c.B = 1; c.nworkers = 1; c.nbuf = 3; c.deterministic = 1; c.seed = 5;
    memset(c.level_p, 0, sizeof c.level_p); c.level_p[0] = 1; c.min_fg = 0; c.empty_keep = 1; c.augment = 1; c.xfmt = 1;
    return c;
}
typedef struct { uint64_t h; int warped, level1, bad_um, bad_code, nonfinite; } run_stats;
static run_stats run(sources *S, const sample_cfg *c, int nb) {
    run_stats st = {.h = 0xcbf29ce484222325ull};
    sampler *sp = sampler_start(S, c); assert(sp);
    const size_t p3 = (size_t)c->P * c->P * c->P;
    for (int i = 0; i < nb; i++) {
        batch *b = sampler_next(sp); assert(b);
        st.h = mix(st.h, b->x16, (size_t)b->xch * p3 * 2);
        if (b->rad) st.h = mix(st.h, b->rad, NN_RAD_REC(c->P));
        if (b->tm) st.h = mix(st.h, b->tm, p3); else { st.h = mix(st.h, b->t, NCH * p3); st.h = mix(st.h, b->m, p3); }
        st.h = mix(st.h, b->um, sizeof *b->um); st.h = mix(st.h, b->level, 1);
        if (b->level[0] == 1) { st.level1++; st.bad_um += !(b->um[0] >= 1.f - 1e-6f && b->um[0] <= c->zoom_max + 1e-6f); }
        if (b->rad) { int32_t hd[8]; memcpy(hd, b->rad, sizeof hd); st.warped += hd[6] & 1; }
        if (b->tm) for (size_t k = 0; k < p3; k++) st.bad_code += b->tm[k] > 7 || (b->tm[k] & 3) == 3;   /* recto and verso exclusive */
        for (size_t k = 0; k < (size_t)b->xch * p3; k++) { _Float16 v; memcpy(&v, b->x16 + k, 2); st.nonfinite += !isfinite((float)v); }
        sampler_release(sp, b);
    }
    sampler_stop(sp);
    return st;
}
int main(int argc, char **argv) {
    char dir[] = "/tmp/ufsm-aug-preset-XXXXXX";
    if (!mkdtemp(dir)) return 2;
    (void)argc; (void)argv;
    fixture(dir);
    char path[1400]; snprintf(path, sizeof path, "%s/sources.json", dir);
    sources *S = sources_load(path); assert(S);
    {   /* "none": the fields the trainer used before the presets (sample_cfg_default + the old flag defaults) */
        sample_cfg a = config(), b = config();
        a.rotate_degrees = 5; a.rotate_p = .2f; a.elastic = 1; a.elastic_p = .15f; a.zoom_min = .7071f; a.affine_aniso = .1f; a.affine_shear = .06f;
        a.scan.strength = 1; a.scan.drop_ignore = 1; a.zoom_max = 1;
        check("preset none: unknown name rejected", sample_aug_preset(&b, "nonsense") == -1);
        assert(!sample_aug_preset(&b, "none"));
        int same = a.ct_augment == b.ct_augment && a.geometry_augment == b.geometry_augment && a.zoom_p == b.zoom_p && a.zoom_out_p == b.zoom_out_p &&
                   a.affine_p == b.affine_p && a.axis_jitter == b.axis_jitter && a.axis_jitter_p == b.axis_jitter_p && a.axis_jitter_deg == b.axis_jitter_deg &&
                   a.rotate_degrees == b.rotate_degrees && a.rotate_p == b.rotate_p && a.elastic == b.elastic && a.elastic_p == b.elastic_p &&
                   a.zoom_min == b.zoom_min && a.zoom_max == b.zoom_max && a.affine_aniso == b.affine_aniso && a.affine_shear == b.affine_shear &&
                   a.scan.strength == b.scan.strength && a.scan.drop_ignore == b.scan.drop_ignore && !memcmp(a.scan.p, b.scan.p, sizeof a.scan.p);
        check("preset none sets the old defaults", same);
        run_stats ra = run(S, &a, 6), rb = run(S, &b, 6);
        check("preset none draws the old batches", ra.h == rb.h && ra.level1 == 0);
    }
    sample_cfg c = config(); assert(!sample_aug_preset(&c, "all"));
    c.zoom_out_p = .5f; c.pack_tm = 1; c.gpu_input = 1;
    const int NB = 32;
    run_stats r1 = run(S, &c, NB);
    c.nworkers = 3; c.nbuf = 5;
    run_stats r3 = run(S, &c, NB);
    check("preset all (packed, gpu input) reproducible across worker counts", r1.h == r3.h);
    check("zoom-out windows read the coarser level", r1.level1 > 0);
    check("zoom-out um within [1, zoom max]", r1.bad_um == 0);
    check("warped windows carry their warp in the radial record", r1.warped > 0 && r1.warped < NB);
    check("packed codes valid, recto and verso exclusive", r1.bad_code == 0);
    check("CT channel finite", r1.nonfinite == 0);
    printf("aug preset: %d windows, %d warped, %d zoomed out\n", NB, r1.warped, r1.level1);
    sources_free(S);
    char cmd[1500]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir); if (system(cmd)) {}
    if (fails) { printf("test_aug_preset: %d FAIL\n", fails); return 1; }
    printf("test_aug_preset: ok\n");
    return 0;
}
