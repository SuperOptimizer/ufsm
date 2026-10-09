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
static scan_aug_cfg one(int op, float strength) { scan_aug_cfg c = {.strength = strength, .drop_ignore = 1}; if (op >= 0) c.p[op] = 1; return c; }
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

int main(void) {
    test_identity(); test_ops(); test_seam(); test_operators();
    if (failures) { fprintf(stderr, "scan augment: %d failure(s)\n", failures); return 1; }
    printf("scan augment: identity, determinism, label/ignore consistency of %d ops, seam registration, operators: ok\n", SA_N);
    return 0;
}
