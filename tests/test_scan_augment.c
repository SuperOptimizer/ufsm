/* CPU-only scanner-domain augmentation invariants: identity at strength 0, determinism, label / ignore consistency of
   every op, exact operators on analytic volumes. */
#include "../src/scan_augment.c"
#include <stdio.h>

static int failures;
#define CHECK(ok, name) do { if (!(ok)) { fprintf(stderr, "FAIL: %s (line %d)\n", name, __LINE__); failures++; } } while (0)

#define P 48
#define N ((size_t)P * P * P)
#define NCHT 2
static const int64_t O[3] = {1000, 2000, 3000};
static const int CHUNK[3] = {16, 16, 16};
typedef struct { uint8_t ct[N], tgt[NCHT * N], ign[N]; float cy[P], cx[P]; } vol;
static vol v0, v1, v2;
static uint8_t scratch[5 * N];

/* textured CT with a masked air slab (x < 6), labelled sheets, some ignore */
static void fill(vol *v, int air, int flat) {
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        size_t k = ((size_t)z * P + y) * P + x;
        v->ct[k] = flat ? 100 : air && x < 6 ? 0 : (uint8_t)(30 + (z * 7 + y * 13 + x * 5 + (x * y) % 17) % 200);
        v->tgt[k] = (x + 2 * y + z) % 12 == 0 ? 254 : (x + 2 * y + z) % 12 == 1 ? 120 : 0;
        v->tgt[N + k] = (uint8_t)(k * 2654435761u >> 24);
        v->ign[k] = (z + y) % 23 == 0;
    }
    for (int z = 0; z < P; z++) { v->cy[z] = (float)O[1] + .5f * (P - 1); v->cx[z] = (float)O[2] + .5f * (P - 1); }
}
static float g_tcdf[256];   /* the hist op's target: material uniform over 150..220 */
static scan_aug_cfg one(int op, float strength) {
    scan_aug_cfg c = {.strength = strength, .drop_ignore = 1}; if (op >= 0) c.p[op] = 1;
    for (int v = 0; v < 256; v++) g_tcdf[v] = v < 150 ? 0.f : v >= 220 ? 1.f : (v - 149) / 71.f;
    c.hist = g_tcdf; c.nhist = 1;
    return c;
}
static unsigned run(const scan_aug_cfg *c, uint64_t seed, vol *v, int hasax, int seam) {
    return scan_aug_apply(c, seed, v->ct, v->tgt, NCHT, v->ign, P, O, v->cy, v->cx, hasax, CHUNK, seam, scratch);
}
static int same(const vol *a, const vol *b) { return !memcmp(a, b, sizeof *a); }

static void test_identity(void) {
    fill(&v0, 1, 0); v1 = v0;
    scan_aug_cfg c = one(-1, 0); for (int i = 0; i < SA_N; i++) c.p[i] = 1;
    CHECK(!scan_aug_enabled(&c) && run(&c, 7, &v1, 1, 1) == 0 && same(&v0, &v1), "strength 0 is the identity");
    c = one(-1, 2);
    CHECK(!scan_aug_enabled(&c) && run(&c, 7, &v1, 1, 1) == 0 && same(&v0, &v1), "all probabilities 0 is the identity");
    c = one(SA_SEAM, 1);
    CHECK(run(&c, 7, &v1, 1, 0) == 0 && same(&v0, &v1), "seam is skipped where labels are not dense rasters");
}

static void test_ops(void) {
    const float strengths[3] = {.5f, 1, 2};
    for (int op = 0; op < SA_N; op++) for (int si = 0; si < 3; si++) for (uint64_t seed = 0; seed < 6; seed++) {
        int hasax = seed & 1;
        fill(&v0, 1, 0); v1 = v0; v2 = v0;
        scan_aug_cfg c = one(op, strengths[si]);
        unsigned on = run(&c, seed, &v1, hasax, 1); run(&c, seed, &v2, hasax, 1);
        CHECK(on == 1u << op, "op fires alone at p = 1");
        CHECK(same(&v1, &v2), "seed determinism");
        int ign_kept = 1, zeros_kept = 1, labels_kept = !memcmp(v0.tgt, v1.tgt, sizeof v0.tgt), changed = 0, new_ign = 0;
        int axis_kept = !memcmp(v0.cy, v1.cy, sizeof v0.cy) && !memcmp(v0.cx, v1.cx, sizeof v0.cx);
        for (size_t k = 0; k < N; k++) {
            ign_kept &= !v0.ign[k] || v1.ign[k];
            new_ign += !v0.ign[k] && v1.ign[k];
            zeros_kept &= !v0.ct[k] == !v1.ct[k];
            changed += v0.ct[k] != v1.ct[k];
        }
        if (op != SA_SEAM) CHECK(ign_kept, "ignore is never cleared (seam moves it with the labels)");
        CHECK(changed > 0, "op changes the CT");
        if (op != SA_SEAM) CHECK(labels_kept && axis_kept, "labels and axis untouched");
        if (op != SA_SEAM && op != SA_AIR && op != SA_CHUNK) CHECK(zeros_kept, "CT > 0 supervision mask preserved");
        if (op != SA_SEAM && op != SA_AIR && op != SA_DROPSLICE && op != SA_CUTOUT) CHECK(new_ign == 0, "intensity ops keep supervision");
        if (op == SA_DROPSLICE || op == SA_CUTOUT) CHECK(new_ign > 0, "dropped regions are ignored");
        if (op == SA_AIR) {
            int ok = 1; for (size_t k = 0; k < N; k++) ok &= v0.ct[k] ? v1.ct[k] != 0 : v1.ct[k] != 0 && v1.ign[k];
            CHECK(ok, "air fill: masked air becomes unsupervised material");
        }
        if (op == SA_CHUNK) {   /* new zeros fill whole cells of the global chunk grid */
            int ok = 1, any = 0;
            for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 6; x < P; x++) {
                if (v1.ct[((size_t)z * P + y) * P + x]) continue;
                any = 1;
                int z0 = (int)(((O[0] + z) / 16) * 16 - O[0]), y0 = (int)(((O[1] + y) / 16) * 16 - O[1]), x0 = (int)(((O[2] + x) / 16) * 16 - O[2]);
                for (int a = z0 < 0 ? 0 : z0; a < z0 + 16 && a < P; a++) for (int b = y0 < 0 ? 0 : y0; b < y0 + 16 && b < P; b++)
                    for (int d = x0 < 0 ? 0 : x0; d < x0 + 16 && d < P; d++) ok &= !v1.ct[((size_t)a * P + b) * P + d];
            }
            CHECK(ok && any, "chunk dropouts are aligned chunk cells");
        }
        if (op == SA_DROPSLICE || op == SA_CUTOUT) {
            fill(&v2, 1, 0); c.drop_ignore = 0; run(&c, seed, &v2, hasax, 1);
            CHECK(!memcmp(v2.ign, v0.ign, sizeof v0.ign), "drop_ignore 0 keeps the labels supervised");
        }
    }
}

/* seam: CT, labels, ignore and axis move together */
static void test_seam(void) {
    for (uint64_t seed = 0; seed < 20; seed++) {
        fill(&v0, 0, 0); memcpy(v0.tgt, v0.ct, N); memset(v0.ign, 0, N); v1 = v0;
        scan_aug_cfg c = one(SA_SEAM, 2); run(&c, seed, &v1, 1, 1);
        int zc = -1; for (int z = 0; z < P && zc < 0; z++) if (v1.cy[z] != v0.cy[z] || v1.cx[z] != v0.cx[z]) zc = z;
        CHECK(zc > 0, "seam moves the axis of a slab");
        if (zc < 0) continue;
        int dy = (int)lroundf(v1.cy[zc] - v0.cy[zc]), dx = (int)lroundf(v1.cx[zc] - v0.cx[zc]), ok = 1;
        for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
            size_t k = ((size_t)z * P + y) * P + x;
            int sy = z >= zc ? y - dy : y, sx = z >= zc ? x - dx : x;
            if (sy < 0 || sy >= P || sx < 0 || sx >= P) ok &= !v1.ct[k] && v1.ign[k] && !v1.tgt[k];
            else { size_t q = ((size_t)z * P + sy) * P + sx; ok &= v1.ct[k] == v0.ct[q] && v1.tgt[k] == v0.tgt[q] && !v1.ign[k]; }
            ok &= v1.ct[k] == v1.tgt[k];
        }
        CHECK(ok, "seam registration of CT, labels and ignore");
    }
}

/* resolution operators keep a constant; lowres reproduces a ramp away from the edges; rings depend on radius only */
static void test_operators(void) {
    for (int op = SA_LOWRES; op <= SA_SHARPEN; op++) for (uint64_t seed = 0; seed < 8; seed++) {
        fill(&v0, 0, 1); v1 = v0; scan_aug_cfg c = one(op, 2); run(&c, seed, &v1, 1, 1);
        CHECK(!memcmp(v0.ct, v1.ct, N), "resolution op preserves a constant");
    }
    static bop b; float dense[P] = {0}, ramp[P], out[P];
    for (double f = 1.3; f <= 4; f += .45) {
        bop_lowres(&b, P, f, .37 * f, dense);
        for (int i = 0; i < P; i++) ramp[i] = (float)i;
        int ok = 1, m = 0; double bias = 0;   /* area sampling of voxel steps: unbiased, within .15 of the ramp */
        for (int i = 0; i < P; i++) {
            out[i] = 0; float wsum = 0;
            for (int t = 0; t < b.n[i]; t++) { out[i] += b.w[i][t] * ramp[b.s[i] + t]; wsum += b.w[i][t]; }
            ok &= fabsf(wsum - 1) < 1e-5f;
            if (i > f + 2 && i < P - f - 3) { ok &= fabsf(out[i] - i) < .15f; bias += out[i] - i; m++; }
        }
        CHECK(ok && fabs(bias / m) < .03, "lowres rows sum to 1 and reproduce a ramp");
    }
    for (uint64_t seed = 0; seed < 6; seed++) for (int op = SA_RING; op <= SA_CUPPING; op++) {
        fill(&v0, 0, 1); v1 = v0; scan_aug_cfg c = one(op, 2); run(&c, seed, &v1, 1, 1);
        int ok = 1;
        for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
            uint8_t v = v1.ct[((size_t)z * P + y) * P + x];
            ok &= v == v1.ct[((size_t)0 * P + y) * P + x] && v == v1.ct[((size_t)z * P + P - 1 - y) * P + x] && v == v1.ct[((size_t)z * P + x) * P + y];
        }
        CHECK(ok, "radial profile is a function of the radius from the scan axis");
    }
}

/* the ops added for scroll-to-scroll generalisation: paganin keeps a constant and lowers the Laplacian energy; gin keeps the
   material mean / sd (it is rescaled to them); hist moves the material median toward the target's; a per-op strength overrides
   the global one exactly */
static double lap_energy(const vol *v) {
    double e = 0; for (int z = 1; z < P - 1; z++) for (int y = 1; y < P - 1; y++) for (int x = 1; x < P - 1; x++) {
        const size_t i = ((size_t)z * P + y) * P + x; const double L = v->ct[i + P * P] + v->ct[i - P * P] + v->ct[i + P] + v->ct[i - P] + v->ct[i + 1] + v->ct[i - 1] - 6.0 * v->ct[i]; e += L * L; }
    return e;
}
static void stats(const vol *v, double *mean, double *sd, int *median) {
    size_t h[256] = {0}, n = 0; double s1 = 0, s2 = 0; for (size_t k = 0; k < N; k++) if (v->ct[k]) { h[v->ct[k]]++; n++; s1 += v->ct[k]; s2 += (double)v->ct[k] * v->ct[k]; }
    *mean = s1 / n; *sd = sqrt(s2 / n - *mean * *mean); size_t c = 0; *median = 0; for (int b = 1; b < 256; b++) { c += h[b]; if (2 * c >= n) { *median = b; break; } }
}
static void test_generalisation_ops(void) {
    for (uint64_t seed = 0; seed < 6; seed++) {
        fill(&v0, 0, 1); v1 = v0; scan_aug_cfg c = one(SA_PAGANIN, 2); run(&c, seed, &v1, 1, 1);
        CHECK(!memcmp(v0.ct, v1.ct, N), "paganin preserves a constant");
        fill(&v0, 1, 0); v1 = v0; c = one(SA_PAGANIN, 1); run(&c, seed, &v1, 1, 1);
        CHECK(lap_energy(&v1) < .7 * lap_energy(&v0), "paganin is a low-pass");
        fill(&v0, 1, 0); v1 = v0; c = one(SA_GIN, 1); run(&c, seed, &v1, 1, 1);
        double m0, s0, m1, s1; int md0, md1; stats(&v0, &m0, &s0, &md0); stats(&v1, &m1, &s1, &md1);
        CHECK(fabs(m1 - m0) < .1 * s0 && fabs(s1 - s0) < .35 * s0, "gin keeps the material mean and sd");
        fill(&v0, 1, 0); v1 = v0; c = one(SA_HIST, 1); run(&c, seed, &v1, 1, 1); stats(&v1, &m1, &s1, &md1);
        CHECK(abs(md1 - 185) < abs(md0 - 185), "hist moves the material median toward the target's");
        fill(&v0, 1, 0); v1 = v0; v2 = v0;
        scan_aug_cfg a = one(SA_BLUR, .5f), b = one(SA_BLUR, 2); a.s[SA_BLUR] = 2;
        run(&a, seed, &v1, 1, 1); run(&b, seed, &v2, 1, 1);
        CHECK(same(&v1, &v2), "a per-op strength overrides the global strength");
    }
}

int main(void) {
    test_identity(); test_ops(); test_seam(); test_operators(); test_generalisation_ops();
    if (failures) { fprintf(stderr, "scan augment: %d failure(s)\n", failures); return 1; }
    {   /* the per-axis operator chain at its widest (blur, sharpen, motion, lowres at per-op strength 3) stays within MAXTAP */
        for (uint64_t seed = 0; seed < 40; seed++) {
            fill(&v0, 1, 0); v1 = v0;
            scan_aug_cfg c = one(-1, 2); c.p[SA_BLUR] = c.p[SA_SHARPEN] = c.p[SA_MOTION] = c.p[SA_LOWRES] = 1;
            c.s[SA_BLUR] = c.s[SA_SHARPEN] = c.s[SA_MOTION] = c.s[SA_LOWRES] = 3;
            CHECK(run(&c, seed, &v1, 1, 1) == (1u << SA_BLUR | 1u << SA_SHARPEN | 1u << SA_MOTION | 1u << SA_LOWRES), "widest chain fires");
        }
        /* motion on a constant volume is the identity; on a ramp along every axis a box blur keeps the interior and a ghost
           shifts it by a constant (sum of echo weight x offset): one value per interior voxel minus the original */
        for (uint64_t seed = 0; seed < 12; seed++) {
            fill(&v0, 0, 1); v1 = v0; scan_aug_cfg c = one(SA_MOTION, 1); run(&c, seed, &v1, 1, 1);
            CHECK(!memcmp(v0.ct, v1.ct, sizeof v0.ct), "motion keeps a constant volume");
            for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) v0.ct[((size_t)z * P + y) * P + x] = (uint8_t)(20 + x + y + z);
            v1 = v0; run(&c, seed, &v1, 1, 1);
            int dmin = 999, dmax = -999;
            for (int z = 10; z < P - 10; z++) for (int y = 10; y < P - 10; y++) for (int x = 10; x < P - 10; x++) {
                const size_t k = ((size_t)z * P + y) * P + x; const int d = (int)v1.ct[k] - (int)v0.ct[k];
                if (d < dmin) dmin = d;
                if (d > dmax) dmax = d;
            }
            CHECK(dmax - dmin <= 1, "motion: a ramp's interior moves by one constant");
        }
    }
    printf("scan augment: identity, determinism, label/ignore consistency of %d ops, seam registration, operators, paganin / gin / hist / per-op strength: ok\n", SA_N);
    return 0;
}
