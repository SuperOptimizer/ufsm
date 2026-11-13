/* the z-range label builders (src/nn_aug.cu nn_label_warp_z / nn_label_sym_z / nn_w4runs_label_z / nn_ink_bits_z, nn_w4runs_apply_z,
   nn_aug_src_range: one window split in z across GPUs) against the whole-window calls, byte for byte. Per case (seeds x warped /
   cube symmetries, P 256 by default) and z range (0 / P/2+32, P/2-32 / P/2+32, 40 / 72, the whole window): the output planes
   [z0, z0 + nz) of every kind (packed targets + pmask, phase, winding mod 4 + component, ink plane, ink bits, the runs-rebuilt winding,
   also from the phase plane left staged) equal the whole-window call's planes. The host planes outside [s0, s1) are poisoned and the
   staging buffer is filled with poison before each call, so a read outside the bound shows. The bound covers every corner the warp
   reads (a host replica of lw_coords) and is at most 4 planes wider than that. Usage: test_aug_z [P 256] */
#define WIND_FIXTURE_NO_MAIN
#include "make_wind_fixture.c"
#include "../src/sample.c"
#include "nn.h"


/* the loopback collectives (two ranks as two host threads on one GPU): staged through host memory between two barriers */
static int g_lb_break;   /* (the negative control: the exchanges deliver nothing) */
typedef struct { pthread_barrier_t bar; void *hb[2]; size_t hn[2]; } loop_ctx;
static void lb_xchg(void *c_, int peer, const void *sd, size_t ns, void *rd, size_t nr, void *st) {
    loop_ctx *c = c_; const int me = 1 - peer; (void)st;
    nn_sync();
    c->hb[me] = ns ? malloc(ns) : nullptr; c->hn[me] = ns; if (ns) nn_d2h(c->hb[me], sd, ns);
    pthread_barrier_wait(&c->bar);
    if (c->hn[peer] != nr) { fprintf(stderr, "FAIL: loopback xchg: rank %d expects %zu bytes, the peer sends %zu\n", me, nr, c->hn[peer]); abort(); }
    if (nr && !g_lb_break) nn_h2d(rd, c->hb[peer], nr);
    pthread_barrier_wait(&c->bar);
    free(c->hb[me]);
}
static __thread int t_rank;
static void lb_sum(void *c_, void *d, int n, int type, void *st) {
    loop_ctx *c = c_; const int me = t_rank; const size_t es = type == 2 ? 4 : 8, nb = (size_t)n * es; (void)st;
    nn_sync();
    c->hb[me] = malloc(nb); nn_d2h(c->hb[me], d, nb);
    pthread_barrier_wait(&c->bar);
    void *o = malloc(nb);
    for (int i = 0; i < n; i++) {
        if (type == 0) ((uint64_t *)o)[i] = ((uint64_t *)c->hb[0])[i] + ((uint64_t *)c->hb[1])[i];
        else if (type == 1) ((double *)o)[i] = ((double *)c->hb[0])[i] + ((double *)c->hb[1])[i];
        else ((uint32_t *)o)[i] = ((uint32_t *)c->hb[0])[i] + ((uint32_t *)c->hb[1])[i];
    }
    nn_h2d(d, o, nb); free(o);
    pthread_barrier_wait(&c->bar);
    free(c->hb[me]);
}
typedef struct { const uint8_t *ct8; const void *rec, *rad_h; void *rad_d, *out; int P, z0, nz; nn_zcomm zc; } ct_job;
static void *ct_rank(void *j_) {
    ct_job *j = j_; t_rank = j->zc.rank;
    nn_aug_ct_z(j->ct8, j->rec, j->rad_h, j->rad_d, j->P, j->z0, j->nz, j->out, &j->zc, 0);   /* (stream 0: the two ranks' work serialised on the device) */
    nn_sync();
    return nullptr;
}
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

/* the warp's source z corners over output planes [z0, z0 + nz) (lw_coords on the host, in double: the bound check has a plane of slack) */
static void src_corners(const void *rad_h, int P, int z0, int nz, int *bmin, int *bmax) {
    const int32_t *h = (const int32_t *)rad_h; const float *rad = (const float *)rad_h;
    const int sb = h[5], affine = h[6] >> 1 & 1;
    const int perm[3] = {sb & 3, sb >> 2 & 3, sb >> 4 & 3}, flip[3] = {sb >> 8 & 1, sb >> 9 & 1, sb >> 10 & 1};
    const float *tb = rad + 8 + 2 * (size_t)P, *R = tb + 6 * (size_t)P, *A = R + 9; const double sc = R[27], c = .5 * (P - 1);
    *bmin = P; *bmax = -1;
    for (int z = z0; z < z0 + nz; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        const int io[3] = {z, y, x}; double vt[3], U[3], V[3];
        for (int d = 0; d < 3; d++) vt[d] = io[d] - c - tb[d * P + io[d]];
        if (affine) {
            double w[3]; for (int d = 0; d < 3; d++) w[d] = R[d] * vt[0] + R[3 + d] * vt[1] + R[6 + d] * vt[2];
            for (int d = 0; d < 3; d++) U[d] = c + sc * (A[d * 3] * w[0] + A[d * 3 + 1] * w[1] + A[d * 3 + 2] * w[2]);
        } else for (int d = 0; d < 3; d++) U[d] = c + sc * (R[d] * vt[0] + R[3 + d] * vt[1] + R[6 + d] * vt[2]);
        for (int d = 0; d < 3; d++) V[perm[d]] = flip[d] ? (P - 1) - U[d] : U[d];
        int in = 1; for (int d = 0; d < 3; d++) if (!(V[d] >= 0 && V[d] <= P - 1)) in = 0;
        if (!in) continue;
        int b = (int)V[0]; if (b > P - 2) b = P - 2;
        if (b < *bmin) *bmin = b;
        if (b + 1 > *bmax) *bmax = b + 1;
    }
}

int main(int argc, char **argv) {
    const int P = argc > 1 ? atoi(argv[1]) : 256; const size_t p2 = (size_t)P * P, p3 = p2 * P;
    nn_init(0);
    const float *cs, *sn; sample_phase_cssn(&cs, &sn);
    /* the source-grid planes: 0 packed targets / mask, 1 phase, 2 w4, 3 comp, 4 ink, 5 ct8; bits: the ink known bits */
    uint8_t *src = nn_host_alloc(6 * p3), *bits = nn_host_alloc(p3 / 8), *pois = nn_host_alloc(6 * p3), *pbits = nn_host_alloc(p3 / 8);
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        const size_t k = ((size_t)z * P + y) * P + x; const uint32_t hsh = (uint32_t)(k * 2654435761u);
        const double r = hypot(y - P * .5 - .13 * z + 40, x - P * .45), q = r / 17.0 + .003 * z, f = q - floor(q);
        src[k] = (uint8_t)(hsh >> 29);
        src[p3 + k] = (x + 2 * y) % 61 < 3 || (r < 9) ? 255 : (uint8_t)((int)(f * 252) % 252);
        src[4 * p3 + k] = (hsh >> 24) % 19 == 0 ? 255 : (uint8_t)((hsh >> 16) % 255);
        src[5 * p3 + k] = (uint8_t)(x < 3 ? 0 : 30 + (hsh >> 25) + (f < .4 ? 100 : 0));
    }
    for (size_t k = 0; k < p3 / 8; k++) bits[k] = (uint8_t)(k * 40503u >> 7 ^ k);
    w4runs ru = {0};
    const int nc = band_wind4_runs(src + p3, P, P, P, 48, 80, 512, 15, &ru, nullptr);   /* (tile 48: strips that the ranges cut) */
    CHECK(nc > 0 && ru.noff, "the runs: %d components", nc);
    w4runs_apply_host(src + p3, P, P, P, ru.tile, ru.ms, &ru, src + 2 * p3, src + 3 * p3);
    { size_t kn = 0; for (size_t k = 0; k < p3; k++) kn += src[3 * p3 + k] != 255; CHECK(kn > p3 / 10, "winding known on %zu voxels only", kn); }
    for (size_t k = 0; k < 6 * p3; k++) pois[k] = (uint8_t)(k * 2246822519u >> 23);
    for (size_t k = 0; k < p3 / 8; k++) pbits[k] = (uint8_t)~bits[k];

    void *rad_h = malloc(NN_RAD_REC(P)), *rad_d = nn_malloc(NN_RAD_REC(P));
    uint8_t *dw = nn_malloc(8 * p3), *dz = nn_malloc(8 * p3), *ref = malloc(8 * p3), *got = malloc(8 * p3);
    float cy[1024], cx[1024]; for (int z = 0; z < P; z++) { cy[z] = P * .5f + .1f * z - 300; cx[z] = P * .4f - 200; }
    const int64_t o[3] = {1000, 2000, 3000};
    const int R[4][2] = {{0, P / 2 + 32}, {P / 2 - 32, P / 2 + 32}, {40, 72}, {0, P}};
    int ncase = 0, nrange = 0, worst_extra = 0; double wide = 0; size_t known = 0;
    /* a staged copy of the host planes, poisoned outside [s0, s1) (the builders may read only those planes) */
    uint8_t *hp = nn_host_alloc(6 * p3), *hb = nn_host_alloc(p3 / 8);
    for (int cs_ = 0; cs_ < 12; cs_++) {
        const int warped = cs_ < 8; const uint64_t seed = 501 + 97 * (uint64_t)cs_;
        const int codes[4] = {1 | 0 << 2 | 2 << 4 | 1 << 9, 0 | 1 << 2 | 2 << 4 | 1 << 8, 2 | 1 << 2 | 0 << 4, 0 | 2 << 2 | 1 << 4 | 1 << 8 | 1 << 10};
        sym sy = sym_of((int)((seed * 7) % 48)); spatial_aug sa = {0};
        if (warped) { spatial_aug_make(&sa, seed, P, cs_ < 4 ? 10 : 30, 1, 1, 1); if (cs_ & 1) spatial_aug_zoom(&sa, .8); }
        rad_record(rad_h, o, P, P, 1, sy, cy, cx, warped ? &sa : nullptr);
        int32_t *h = (int32_t *)rad_h;
        if (!warped) { h[5] = codes[cs_ - 8]; h[6] &= ~1; }   /* (the unwarped cases: given symmetries, two with z from y / x) */
        nn_h2d(rad_d, rad_h, NN_RAD_REC(P));
        const int wv = h[6] & 1, code = h[5];
        /* the whole-window references: 0 tm (+ pm), 1 phase, 2 w4 + comp, 3 ink, 4 ink bits, 5 runs-rebuilt w4 + comp */
        nn_aug_stage_reset();
        if (wv) {
            nn_label_warp(0, src, nullptr, rad_d, P, dw, dw + p3, cs, sn, 0);
            nn_label_warp(1, src + p3, nullptr, rad_d, P, dw + 2 * p3, nullptr, cs, sn, 0);
            nn_label_warp(2, src + 2 * p3, src + 3 * p3, rad_d, P, dw + 3 * p3, dw + 4 * p3, cs, sn, 0);
            nn_label_warp(3, src + 4 * p3, nullptr, rad_d, P, dw + 5 * p3, nullptr, cs, sn, 0);
        } else {
            nn_label_sym(src + p3, code, P, dw + 2 * p3, 0); nn_label_sym(src + 4 * p3, code, P, dw + 5 * p3, 0);
            nn_label_sym(src + 2 * p3, code, P, dw + 3 * p3, 0); nn_label_sym(src + 3 * p3, code, P, dw + 4 * p3, 0);
        }
        nn_ink_bits(src + 5 * p3, bits, 64, rad_d, wv, code, P, dw + 6 * p3, 0);
        nn_aug_stage_reset(); nn_w4runs_label(src + p3, &ru, rad_d, wv, code, P, dz, dz + p3, cs, sn, 0);
        uint8_t *ref2 = malloc(2 * p3); nn_d2h(ref, dw, 7 * p3); nn_d2h(ref2, dz, 2 * p3);   /* (ref planes 0-6 as above; ref2: the runs' w4 / comp) */
        for (size_t k = 0; k < p3; k++) known += ref[2 * p3 + k] != 255;
        for (int ri = 0; ri < 4; ri++) {
            const int z0 = R[ri][0], nz = R[ri][1]; int s0, s1; nn_aug_src_range(rad_h, P, z0, nz, &s0, &s1);
            if (wv) {
                int bmin, bmax; src_corners(rad_h, P, z0, nz, &bmin, &bmax);
                CHECK(bmax < 0 || (s0 <= bmin && s1 > bmax), "case %d range %d: the bound [%d, %d) misses the corners [%d, %d]", cs_, ri, s0, s1, bmin, bmax);
                const int extra = bmax < 0 ? 0 : (s1 - s0) - (bmax + 1 - bmin); if (extra > worst_extra) worst_extra = extra;
                CHECK(extra <= 4, "case %d range %d: the bound [%d, %d) is %d planes wider than the corners [%d, %d]", cs_, ri, s0, s1, extra, bmin, bmax);
            }
            if ((double)(s1 - s0) / nz > wide && nz < P) wide = (double)(s1 - s0) / nz;
            memcpy(hp, pois, 6 * p3); memcpy(hb, pbits, p3 / 8);
            for (int pl = 0; pl < 6; pl++) memcpy(hp + pl * p3 + (size_t)s0 * p2, src + pl * p3 + (size_t)s0 * p2, (size_t)(s1 - s0) * p2);
            memcpy(hb + (size_t)s0 * p2 / 8, bits + (size_t)s0 * p2 / 8, (size_t)(s1 - s0) * p2 / 8);
            const size_t n = (size_t)nz * p2, off = (size_t)z0 * p2;
#define POISON() do { nn_aug_stage_reset(); nn_label_sym(pois, 0 | 1 << 2 | 2 << 4, P, (uint8_t *)dw, 0); nn_label_warp_z(2, pois, pois + p3, rad_d, P, 0, P, 0, P, (uint8_t *)dw, (uint8_t *)dw + p3, cs, sn, 0); } while (0)
            /* (POISON: whole-window calls on the poison planes leave poison in both halves of the staging buffer) */
            if (wv) {
                POISON(); nn_label_warp_z(0, hp, nullptr, rad_d, P, z0, nz, s0, s1, dz, dz + n, cs, sn, 0); nn_d2h(got, dz, 2 * n);
                CHECK(!memcmp(got, ref + off, n) && !memcmp(got + n, ref + p3 + off, n), "case %d range %d: packed targets / pmask differ", cs_, ri);
                POISON(); nn_label_warp_z(1, hp + p3, nullptr, rad_d, P, z0, nz, s0, s1, dz, nullptr, cs, sn, 0); nn_d2h(got, dz, n);
                CHECK(!memcmp(got, ref + 2 * p3 + off, n), "case %d range %d: phase differs", cs_, ri);
                nn_label_warp_ref(1); POISON(); nn_label_warp_z(1, hp + p3, nullptr, rad_d, P, z0, nz, s0, s1, dz, nullptr, cs, sn, 0); nn_label_warp_ref(0); nn_d2h(got, dz, n);
                CHECK(!memcmp(got, ref + 2 * p3 + off, n), "case %d range %d: phase (reference kernel) differs", cs_, ri);
                POISON(); nn_label_warp_z(2, hp + 2 * p3, hp + 3 * p3, rad_d, P, z0, nz, s0, s1, dz, dz + n, cs, sn, 0); nn_d2h(got, dz, 2 * n);
                CHECK(!memcmp(got, ref + 3 * p3 + off, n) && !memcmp(got + n, ref + 4 * p3 + off, n), "case %d range %d: w4 / comp differ", cs_, ri);
                POISON(); nn_label_warp_z(3, hp + 4 * p3, nullptr, rad_d, P, z0, nz, s0, s1, dz, nullptr, cs, sn, 0); nn_d2h(got, dz, n);
                CHECK(!memcmp(got, ref + 5 * p3 + off, n), "case %d range %d: ink plane differs", cs_, ri);
            } else {
                POISON(); nn_label_sym_z(hp + p3, code, P, z0, nz, s0, s1, dz, 0); nn_d2h(got, dz, n);
                CHECK(!memcmp(got, ref + 2 * p3 + off, n), "case %d range %d: symmetry (phase) differs", cs_, ri);
                POISON(); nn_label_sym_z(hp + 4 * p3, code, P, z0, nz, s0, s1, dz, 0); nn_d2h(got, dz, n);
                CHECK(!memcmp(got, ref + 5 * p3 + off, n), "case %d range %d: symmetry (ink) differs", cs_, ri);
            }
            POISON(); nn_ink_bits_z(hp + 5 * p3, hb, 64, rad_d, wv, code, P, z0, nz, s0, s1, dz, 0); nn_d2h(got, dz, n);
            CHECK(!memcmp(got, ref + 6 * p3 + off, n), "case %d range %d: ink bits differ", cs_, ri);
            nn_label_warp_ref(1); POISON(); nn_ink_bits_z(hp + 5 * p3, hb, 64, rad_d, wv, code, P, z0, nz, s0, s1, dz, 0); nn_label_warp_ref(0); nn_d2h(got, dz, n);
            CHECK(!memcmp(got, ref + 6 * p3 + off, n), "case %d range %d: ink bits (plane written first) differ", cs_, ri);
            POISON(); nn_w4runs_label_z(hp + p3, &ru, rad_d, wv, code, P, z0, nz, s0, s1, dz, dz + n, cs, sn, 0); nn_d2h(got, dz, 2 * n);
            CHECK(!memcmp(got, ref2 + off, n) && !memcmp(got + n, ref2 + p3 + off, n), "case %d range %d: the runs-rebuilt w4 / comp differ", cs_, ri);
            /* the phase plane left staged by its own call (warped: nn_label_warp_z kind 1; unwarped: nn_label_sym_z) */
            POISON();
            if (wv) nn_label_warp_z(1, hp + p3, nullptr, rad_d, P, z0, nz, s0, s1, dz + 2 * n, nullptr, cs, sn, 0); else nn_label_sym_z(hp + p3, code, P, z0, nz, s0, s1, dz + 2 * n, 0);
            nn_w4runs_label_z(hp + p3, &ru, rad_d, wv, code, P, z0, nz, s0, s1, dz, dz + n, cs, sn, 0); nn_d2h(got, dz, 2 * n);
            CHECK(!memcmp(got, ref2 + off, n) && !memcmp(got + n, ref2 + p3 + off, n), "case %d range %d: the runs-rebuilt w4 / comp from the staged phase differ", cs_, ri);
            POISON(); w4dev_fail_inject(1); nn_w4runs_label_z(hp + p3, &ru, rad_d, wv, code, P, z0, nz, s0, s1, dz, dz + n, cs, sn, 0); nn_d2h(got, dz, 2 * n);
            CHECK(!memcmp(got, ref2 + off, n) && !memcmp(got + n, ref2 + p3 + off, n), "case %d range %d: the runs-rebuilt w4 / comp (host fallback) differ", cs_, ri);
#undef POISON
            nrange++;
        }
        free(ref2); ncase++;
    }
    {   /* nn_w4runs_apply_z on its own: planes [s0, s1) of the whole-window rebuild, for ranges cutting the 48-plane strips */
        uint8_t *ph = nn_malloc(p3), *w4 = nn_malloc(2 * p3), *h0 = malloc(2 * p3), *h1 = malloc(2 * p3);
        nn_h2d(ph, src + p3, p3); CHECK(!nn_w4runs_apply(ph, P, P, P, ru.tile, ru.ms, &ru, w4, w4 + p3, nullptr), "apply"); nn_sync(); nn_d2h(h0, w4, 2 * p3);
        CHECK(!memcmp(h0, src + 2 * p3, 2 * p3), "nn_w4runs_apply differs from w4runs_apply_host");
        const int rr[3][2] = {{5, 77}, {96, 200}, {131, 132}};
        for (int i = 0; i < 3; i++) {
            const int s0 = rr[i][0] * P / 256, s1 = rr[i][1] * P / 256; const size_t n = (size_t)(s1 - s0) * p2;
            nn_h2d(ph, src + p3 + (size_t)s0 * p2, n); CHECK(!nn_w4runs_apply_z(ph, P, P, P, s0, s1, ru.tile, ru.ms, &ru, ph, w4, nullptr), "apply_z"); nn_sync();
            nn_d2h(h1, ph, n); nn_d2h(h1 + n, w4, n);
            CHECK(!memcmp(h1, h0 + (size_t)s0 * p2, n) && !memcmp(h1 + n, h0 + p3 + (size_t)s0 * p2, n), "nn_w4runs_apply_z [%d, %d) differs (in place)", s0, s1);
        }
        nn_free(ph); nn_free(w4); free(h0); free(h1);
    }

    {   /* the CT channel (nn_aug_ct_z): one GPU's whole-window call against (a) a z range on one GPU and (b) the split: two ranks (threads)
           through the loopback collectives, each its output planes. Every scanner op alone, a mixture, none; warped and not */
        uint8_t *ct = malloc(p3), *tgt = malloc(2 * p3 + 64), *ign = malloc(p3), *cin = malloc(p3);
        float cy0[1024], cx0[1024], cyy[1024], cxx[1024];
        for (int z = 0; z < P; z++) { cy0[z] = P * .5f + .1f * z - 300; cx0[z] = P * .4f - 200; }
        for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
            const size_t k = ((size_t)z * P + y) * P + x;
            const double r = hypot(y - cy0[z], x - cx0[z]), q = r / 14.0 + .002 * z, f = q - floor(q);
            ct[k] = (uint8_t)(x < 5 || (x + y) % 97 == 0 ? 0 : f < .35 ? 140 + (k * 2654435761u >> 28) : 40 + (k * 40503u >> 29));
            tgt[k] = f < .2 ? 255 : 0; tgt[p3 + k] = f >= .2 && f < .35 ? 255 : 0;
        }
        float g_t[256]; for (int v = 0; v < 256; v++) g_t[v] = v < 120 ? 0.f : v >= 200 ? 1.f : (v - 119) / 81.f;
        scan_plan *pl = malloc(sizeof *pl); scan_ct_rec *rec = nn_host_alloc(sizeof *rec); uint8_t *cin_p = nn_host_alloc(p3);
        const int chunk[3] = {32, 32, 32};
        uint16_t *Hw = malloc(2 * p3), *Hz = malloc(2 * p3); void *Hd = nn_malloc(2 * p3), *Hr[2] = {nn_malloc(2 * p3), nn_malloc(2 * p3)};
        int nct = 0, nbad = 0; size_t worst = 0;
        const int ranges[2][2][2] = {{{0, P / 2 + 32}, {P / 2 - 32, P / 2 + 32}}, {{40, 72}, {P / 2 - 32, P / 2 + 32}}};
        for (int op = -2; op < SA_N; op++) for (int sv = 0; sv < 2; sv++) {
            scan_aug_cfg c = {.strength = sv ? 2.f : 1.f, .drop_ignore = 1, .hist = g_t, .nhist = 1};
            if (op >= 0) c.p[op] = 1; else if (op == -1) for (int i = 0; i < SA_N; i++) c.p[i] = .45f;
            const uint64_t seed = 91 + 1009 * (uint64_t)(op + 2) + 31 * sv; const int warped = (op + sv) % 2 == 0;
            memcpy(cin, ct, p3); memset(ign, 0, p3); memcpy(cyy, cy0, sizeof cyy); memcpy(cxx, cx0, sizeof cxx);
            uint8_t *tg = malloc(2 * p3); memcpy(tg, tgt, 2 * p3);
            const unsigned on = scan_aug_plan(&c, seed, cin, tg, 2, ign, P, o, cyy, cxx, sv, chunk, 1, pl); free(tg);
            sym sy = sym_of((int)((seed * 7) % 48)); spatial_aug sa = {0};
            if (warped) { spatial_aug_make(&sa, seed, P, 10, 1, 1, 1); spatial_aug_zoom(&sa, .8); }
            rad_record(rad_h, o, P, P, sv, sy, cyy, cxx, warped ? &sa : nullptr); nn_h2d(rad_d, rad_h, NN_RAD_REC(P));
            memset(rec, 0, sizeof *rec); rec->scan = on != 0; rec->xfmt = 1; rec->jitter = sv; rec->ia = 1.05f; rec->ib = .02f; rec->isg = .05f; rec->nseed = seed; if (on) rec->plan = *pl;
            memcpy(cin_p, cin, p3);
            nn_aug_cnoise_chunk(op == SA_CNOISE && sv ? 7 : -1);   /* (an odd chunk once: the carried plane across chunks and ranks) */
            nn_aug_stage_reset(); nn_aug_ct(cin_p, rec, rad_d, P, Hd, 0); nn_d2h(Hw, Hd, 2 * p3);
            for (int rg = 0; rg < 2; rg++) {
                {   /* (a) one GPU, rank 1's range */
                    const int z0 = ranges[rg][1][0], nz = ranges[rg][1][1];
                    nn_aug_stage_reset(); nn_aug_ct_z(cin_p, rec, rad_h, rad_d, P, z0, nz, Hd, nullptr, 0); nn_d2h(Hz, Hd, 2 * (size_t)nz * p2);
                    if (memcmp(Hz, Hw + (size_t)z0 * p2, 2 * (size_t)nz * p2)) { nbad++; fprintf(stderr, "FAIL: CT %s s%d (on %x): one GPU, range [%d, %d) differs\n", op >= 0 ? scan_aug_names[op] : op == -1 ? "mixture" : "none", sv, on, z0, z0 + nz); }
                }
                loop_ctx lc; pthread_barrier_init(&lc.bar, nullptr, 2);
                ct_job J[2]; pthread_t th[2];
                for (int r = 0; r < 2; r++) {
                    J[r] = (ct_job){.ct8 = cin_p, .rec = rec, .rad_h = rad_h, .rad_d = rad_d, .out = Hr[r], .P = P, .z0 = ranges[rg][r][0], .nz = ranges[rg][r][1]};
                    J[r].zc = (nn_zcomm){.ctx = &lc, .rank = r, .nrank = 2, .slot = r, .peer_z0 = ranges[rg][1 - r][0], .peer_nz = ranges[rg][1 - r][1], .xchg = lb_xchg, .sum = lb_sum};
                }
                nn_aug_stage_reset(); nn_sync();
                for (int r = 0; r < 2; r++) pthread_create(&th[r], nullptr, ct_rank, &J[r]);
                for (int r = 0; r < 2; r++) pthread_join(th[r], nullptr);
                pthread_barrier_destroy(&lc.bar);
                for (int r = 0; r < 2; r++) {
                    const size_t nb = 2 * (size_t)J[r].nz * p2; nn_d2h(Hz, Hr[r], nb);
                    size_t nd = 0; for (size_t k = 0; k < (size_t)J[r].nz * p2; k++) nd += Hz[k] != Hw[(size_t)J[r].z0 * p2 + k];
                    if (nd > worst) worst = nd;
                    if (nd) { nbad++; fprintf(stderr, "FAIL: CT %s s%d (on %x, %s): split rank %d [%d, %d): %zu voxels differ\n", op >= 0 ? scan_aug_names[op] : op == -1 ? "mixture" : "none", sv, on, warped ? "warp" : "symmetry", r, J[r].z0, J[r].z0 + J[r].nz, nd); }
                }
            }
            nct++;
        }
        {   /* the negative control: the mixture case with the exchanges dropped must differ (the split depends on them) */
            scan_aug_cfg c = {.strength = 1.f, .drop_ignore = 1, .hist = g_t, .nhist = 1}; for (int i = 0; i < SA_N; i++) c.p[i] = .6f;
            memcpy(cin, ct, p3); memset(ign, 0, p3); memcpy(cyy, cy0, sizeof cyy); memcpy(cxx, cx0, sizeof cxx);
            uint8_t *tg = malloc(2 * p3); memcpy(tg, tgt, 2 * p3);
            const unsigned on = scan_aug_plan(&c, 4242, cin, tg, 2, ign, P, o, cyy, cxx, 1, chunk, 1, pl); free(tg);
            sym sy = sym_of(5); rad_record(rad_h, o, P, P, 1, sy, cyy, cxx, nullptr); nn_h2d(rad_d, rad_h, NN_RAD_REC(P));
            memset(rec, 0, sizeof *rec); rec->scan = on != 0; rec->xfmt = 1; if (on) rec->plan = *pl; memcpy(cin_p, cin, p3);
            nn_aug_stage_reset(); nn_aug_ct(cin_p, rec, rad_d, P, Hd, 0); nn_d2h(Hw, Hd, 2 * p3);
            loop_ctx lc; pthread_barrier_init(&lc.bar, nullptr, 2); ct_job J[2]; pthread_t th[2];
            for (int r = 0; r < 2; r++) {
                J[r] = (ct_job){.ct8 = cin_p, .rec = rec, .rad_h = rad_h, .rad_d = rad_d, .out = Hr[r], .P = P, .z0 = ranges[0][r][0], .nz = ranges[0][r][1]};
                J[r].zc = (nn_zcomm){.ctx = &lc, .rank = r, .nrank = 2, .slot = r, .peer_z0 = ranges[0][1 - r][0], .peer_nz = ranges[0][1 - r][1], .xchg = lb_xchg, .sum = lb_sum};
            }
            g_lb_break = 1; nn_aug_stage_reset(); nn_sync();
            for (int r = 0; r < 2; r++) pthread_create(&th[r], nullptr, ct_rank, &J[r]);
            for (int r = 0; r < 2; r++) pthread_join(th[r], nullptr);
            g_lb_break = 0; pthread_barrier_destroy(&lc.bar);
            size_t nd = 0; for (int r = 0; r < 2; r++) { nn_d2h(Hz, Hr[r], 2 * (size_t)J[r].nz * p2); for (size_t k = 0; k < (size_t)J[r].nz * p2; k++) nd += Hz[k] != Hw[(size_t)J[r].z0 * p2 + k]; }
            CHECK(on && nd > 0, "negative control: the split without its exchanges still matched (on %x, %zu voxels differ)", on, nd);
            printf("test_aug_z: negative control: the split without its exchanges differs on %zu voxels (on %x)\n", nd, on);
        }
        nn_aug_cnoise_chunk(-1);
        CHECK(!nbad, "CT channel: %d differing calls of %d cases (worst %zu voxels)", nbad, nct, worst);
        printf("test_aug_z: CT channel: %d scanner cases x 2 range pairs: one-GPU ranges and the split ranks = the whole-window planes\n", nct);
        free(ct); free(tgt); free(ign); free(cin); free(pl); nn_host_free(rec); nn_host_free(cin_p); free(Hw); free(Hz); nn_free(Hd); nn_free(Hr[0]); nn_free(Hr[1]);
    }
    printf("test_aug_z: P %d, %d cases x %d ranges (%zu known phase codes): every z-range builder = the whole-window planes; bound <= %d planes over the corners, "
           "widest staging %.2f x nz; %s\n", P, ncase, nrange / (ncase ? ncase : 1), known, worst_extra, wide, fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
