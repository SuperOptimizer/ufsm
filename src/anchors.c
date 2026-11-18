/* ufsm gt-anchors (2026-11-18): sparse ground-truth anchors for count-field's global count ("how much ground truth do we need").
   From the labelled voxels of GT winding codes inside the anchor boxes (a box list minus a held-out set), each kept with probability
   --p (a hash of the seed and the voxel), one of two kinds:
     pin   (absolute) the voxel's U' = code / 18 + G + 14 m, the 14-turn period m nearest the reference model's U' there and one gauge G
           for the whole set (the mode of U'_ref - code / 18 mod 14; the per-box modes are printed: the GT is one numbering if they agree)
     pair  (relative, gauge-free) the voxel and a labelled voxel 1-3 sheets out or in along the radial ray from the axis through it (the
           k-th distinct sheet the ray crosses, k uniform in 1..3, within --pair-max native voxels and the box): dU = the codes'
           difference (mod 14 into (-7, 7], rounded: on one ray U' differs by whole turns), what an annotator's "this sheet is k wraps out
           from that one" gives
   The codes encode U' (count-field's U = W + dir theta / 2 pi) x 18 mod 252 (fit --codes compares them with U, src/main.c). Anchors
   that fall in one solve cell (pins) or one pair of solve cells (pairs) of --cell native voxels are merged (mean position / target,
   count n): above ~1 anchor per cell more points add no constraint. Output float32 records, box-local native coordinates of FIELDS:
   pins (z y x U' n), pairs (za ya xa zb yb xb dU n): count-field --anchor-pins / --anchor-pairs. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fit.h"
#include "optcheck.h"
#include "store.h"
#include "zarr3.h"

static const char *aopt(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return argv[i + 1]; return d; }
static uint64_t ah(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return x; }
static double au(uint64_t seed, uint64_t i, uint64_t salt) { return (ah(ah(seed * 0x9e3779b97f4a7c15ULL + salt) ^ i) >> 11) * (1.0 / 9007199254740992.0); }

typedef struct { char name[64]; long long z, y, x, nz, ny, nx; } a_box;
static int a_boxes(const char *path, a_box **out) {
    FILE *f = fopen(path, "r"); if (!f) return -1; char ln[1024]; int n = 0, cap = 0; a_box *b = nullptr;
    while (fgets(ln, sizeof ln, f)) {
        if (ln[0] == '#' || ln[0] == '\n') continue;
        a_box e; char sc[64]; if (sscanf(ln, "%63s %63s %lld %lld %lld %lld %lld %lld", e.name, sc, &e.z, &e.y, &e.x, &e.nz, &e.ny, &e.nx) != 8) continue;
        if (n == cap) { cap = cap ? 2 * cap : 64; b = realloc(b, cap * sizeof *b); }
        b[n++] = e;
    }
    fclose(f); *out = b; return n;
}
static int a_held(const char *list, const char *name) {   /* name in the comma list */
    const size_t n = strlen(name);
    for (const char *p = list; p && *p;) { const char *e = strchr(p, ','); const size_t l = e ? (size_t)(e - p) : strlen(p); if (l == n && !strncmp(p, name, n)) return 1; p = e ? e + 1 : nullptr; }
    return 0;
}
/* merged anchors: open addressing on a 64-bit key */
typedef struct { uint64_t *key; double (*s)[8]; size_t cap, n; } a_map;
static double *a_slot(a_map *M, uint64_t k) {
    if (2 * (M->n + 1) > M->cap) {
        a_map N = {calloc(M->cap ? 2 * M->cap : 1 << 16, 8), nullptr, M->cap ? 2 * M->cap : 1 << 16, 0}; N.s = calloc(N.cap, sizeof *N.s);
        for (size_t i = 0; i < M->cap; i++) if (M->key[i]) { size_t j = ah(M->key[i]) & (N.cap - 1); while (N.key[j]) j = (j + 1) & (N.cap - 1); N.key[j] = M->key[i]; memcpy(N.s[j], M->s[i], sizeof N.s[j]); N.n++; }
        free(M->key); free(M->s); *M = N;
    }
    size_t j = ah(k) & (M->cap - 1); while (M->key[j] && M->key[j] != k) j = (j + 1) & (M->cap - 1);
    if (!M->key[j]) { M->key[j] = k; M->n++; }
    return M->s[j];
}

int cmd_gt_anchors(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: ufsm gt-anchors CODES_ROOT KEY FIELDS --boxes B.tsv --hold N1,N2.. --kind pin|pair --p P --out F.f32 [--seed 1]\n"
                        "         [--ref MODEL (pin: the reference whose 14-turn period lifts the codes)] [--dir 1|-1 (pair: theta sign)] [--cell 16]\n"
                        "         [--pair-max 400 (native)] [--threads 16]\n"
                        "  sparse GT anchors from the labelled voxels of the boxes not held out (src/anchors.c): pins (z y x U' n) or radial pairs\n"
                        "  1-3 sheets apart (za ya xa zb yb xb dU n), box-local native coordinates of FIELDS, merged per solve cell (--cell)\n");
        return 2;
    }
    const char *root = argv[2], *key = argv[3], *fd = argv[4], *kind = aopt(argc, argv, "--kind", "pair"), *outp = aopt(argc, argv, "--out", nullptr);
    const double p = opt_atof("--p", aopt(argc, argv, "--p", "0.001")), cell = opt_atof("--cell", aopt(argc, argv, "--cell", "16")), pmax = opt_atof("--pair-max", aopt(argc, argv, "--pair-max", "400"));
    const uint64_t seed = (uint64_t)opt_atoi("--seed", aopt(argc, argv, "--seed", "1")); const int nt = opt_atoi("--threads", aopt(argc, argv, "--threads", "16"));
    const int pin = !strcmp(kind, "pin"); if (!pin && strcmp(kind, "pair")) { fprintf(stderr, "gt-anchors: --kind pin or pair\n"); return 2; }
    if (!outp) { fprintf(stderr, "gt-anchors: --out\n"); return 2; }
    a_box *bx = nullptr; const int nb = a_boxes(aopt(argc, argv, "--boxes", ""), &bx); if (nb <= 0) { fprintf(stderr, "gt-anchors: no boxes (--boxes)\n"); return 2; }
    const char *hold = aopt(argc, argv, "--hold", "");
    fit_fields_buf g; if (fit_fields_load(fd, 0, 0, 0, &g)) { fprintf(stderr, "gt-anchors: cannot load %s\n", fd); return 1; }
    const int na = (int)(g.shape[0] * g.h) + 1; const double lcf = atof(key) / g.um;   /* native voxels per label voxel */
    fit_model *ref = nullptr; int dir = opt_atoi("--dir", aopt(argc, argv, "--dir", "0"));
    if (pin) { if (!(ref = fit_load(aopt(argc, argv, "--ref", "")))) { fprintf(stderr, "gt-anchors: --kind pin needs --ref MODEL\n"); return 2; } dir = ref->dir; }
    if (!pin && dir != 1 && dir != -1) { fprintf(stderr, "gt-anchors: --kind pair needs --dir 1 or -1\n"); return 2; }
    const double ro[3] = {ref ? (double)(g.origin[0] - ref->origin[0]) : 0, ref ? (double)(g.origin[1] - ref->origin[1]) : 0, ref ? (double)(g.origin[2] - ref->origin[2]) : 0};
    z3 *zc = z3_open(store_open(root), key, nullptr); if (!zc) { fprintf(stderr, "gt-anchors: %s\n", z3_error()); return 1; }
    const int64_t *csh = z3_meta_of(zc)->shape;
    a_map M = {0}; size_t nlab = 0, nraw = 0, nmiss = 0; double vol = 0; int nab = 0;
    /* pins: the gauge from a first sparse pass (every box's voxels at 1e-3), then the main pass */
    enum { NG = 14 * 36 }; double gauge = 0;
    for (int pass = pin ? 0 : 1; pass < 2; pass++) {
        size_t gh[NG] = {0};
        for (int b = 0; b < nb; b++) {
            if (a_held(hold, bx[b].name)) continue;
            /* the box in label voxels (box: absolute native at the fields' level) */
            const int64_t o3[3] = {(int64_t)floor(bx[b].z / lcf), (int64_t)floor(bx[b].y / lcf), (int64_t)floor(bx[b].x / lcf)};
            int64_t n3[3] = {(int64_t)ceil(bx[b].nz / lcf), (int64_t)ceil(bx[b].ny / lcf), (int64_t)ceil(bx[b].nx / lcf)};
            for (int d = 0; d < 3; d++) if (o3[d] + n3[d] > csh[d]) n3[d] = csh[d] - o3[d];
            const size_t N = (size_t)n3[0] * n3[1] * n3[2]; uint8_t *c = malloc(N);
            if (!c || z3_read(zc, o3, n3, c, nt)) { fprintf(stderr, "gt-anchors: cannot read the codes of box %s\n", bx[b].name); return 1; }
            size_t bgh[NG] = {0}, bl = 0;
            for (size_t i = 0; i < N; i++) {
                const uint8_t cd = c[i]; if (cd < 1 || cd > 252) continue;
                bl++;
                if (au(seed, ((uint64_t)b << 40) ^ i, pass) >= (pass == 0 ? 1e-3 : p)) continue;
                const int64_t lz = (int64_t)(i / ((size_t)n3[1] * n3[2])), ly = (int64_t)(i / n3[2] % n3[1]), lx = (int64_t)(i % n3[2]);
                const double az = lcf * (o3[0] + lz + 0.25) - g.origin[0], ay = lcf * (o3[1] + ly + 0.25) - g.origin[1], ax = lcf * (o3[2] + lx + 0.25) - g.origin[2];
                if (az < 0 || ay < 0 || ax < 0 || az > g.shape[0] * g.h || ay > g.shape[1] * g.h || ax > g.shape[2] * g.h) continue;
                const double cy = g.cy[(int)fmin(na - 1, fmax(0, az))], cx = g.cx[(int)fmin(na - 1, fmax(0, az))], q = (cd - 1) / 18.0;
                if (pin) {
                    const double ur = fit_W(ref, az + ro[0], ay + ro[1], ax + ro[2]) + dir * atan2(ay - cy, ax - cx) / (2 * M_PI);
                    if (pass == 0) { double d = fmod(ur - q, 14.0); if (d < 0) d += 14; gh[(int)(d / 14 * NG) % NG]++; bgh[(int)(d / 14 * NG) % NG]++; continue; }
                    const double t = q + gauge, u = t + 14 * round((ur - t) / 14);
                    nraw++;
                    const uint64_t k = 1 + (uint64_t)floor(az / cell) * 1000003ULL * 1000003ULL + (uint64_t)floor(ay / cell) * 1000003ULL + (uint64_t)floor(ax / cell);
                    double *s = a_slot(&M, k); s[0] += az; s[1] += ay; s[2] += ax; s[3] += u; s[7] += 1;
                    nmiss += fabs(u - ur) >= 0.5;
                    continue;
                }
                /* a pair: walk the radial ray (in or out, random) in half label-voxel steps */
                const double ry = ay - cy, rx = ax - cx, rr = hypot(ry, rx); if (rr < 1) continue;
                const double sgn = au(seed, ((uint64_t)b << 40) ^ i, 7) < 0.5 ? -1 : 1, uy = sgn * ry / rr, ux = sgn * rx / rr;
                const int want = 1 + (int)(au(seed, ((uint64_t)b << 40) ^ i, 9) * 3);
                double lastq = q; int sheets = 0; int64_t hit = -1; double bq = 0;
                for (double s = 0.5 * lcf; s <= pmax; s += 0.5 * lcf) {
                    const double py = ay + uy * s, px = ax + ux * s;
                    const int64_t jy = (int64_t)floor((py + g.origin[1]) / lcf - 0.25 + 0.5) - o3[1], jx = (int64_t)floor((px + g.origin[2]) / lcf - 0.25 + 0.5) - o3[2];
                    if (jy < 0 || jx < 0 || jy >= n3[1] || jx >= n3[2]) break;
                    const uint8_t c2 = c[((size_t)lz * n3[1] + jy) * n3[2] + jx]; if (c2 < 1 || c2 > 252) continue;
                    const double q2 = (c2 - 1) / 18.0; double dq = fmod(q2 - lastq, 14.0); if (dq < -7) dq += 14; if (dq > 7) dq -= 14;
                    if (fabs(dq) < 0.5) continue;   /* (the same sheet) */
                    lastq = q2; if (++sheets == want) { hit = ((int64_t)jy << 32) | jx; bq = q2; break; }
                }
                if (hit < 0) { nmiss++; continue; }
                const int64_t jy = hit >> 32, jx = hit & 0xffffffff;
                const double bz = az, by = lcf * (o3[1] + jy + 0.25) - g.origin[1], bxx = lcf * (o3[2] + jx + 0.25) - g.origin[2];
                double du = fmod(bq - q, 14.0); if (du <= -7) du += 14; if (du > 7) du -= 14; du = round(du);
                nraw++;
                const uint64_t ka = (uint64_t)floor(az / cell) * 1000003ULL * 1000003ULL + (uint64_t)floor(ay / cell) * 1000003ULL + (uint64_t)floor(ax / cell);
                const uint64_t kb = (uint64_t)floor(bz / cell) * 1000003ULL * 1000003ULL + (uint64_t)floor(by / cell) * 1000003ULL + (uint64_t)floor(bxx / cell);
                double *sl = a_slot(&M, 1 + ah(ka * 31 + ah(kb)) % 0xfffffffffffffULL * 16 + (uint64_t)(du + 7));
                sl[0] += az; sl[1] += ay; sl[2] += ax; sl[3] += bz; sl[4] += by; sl[5] += bxx; sl[6] += du; sl[7] += 1;
            }
            if (pass == 1) { nlab += bl; vol += (double)bx[b].nz * bx[b].ny * bx[b].nx; nab++; }
            if (pass == 0) { int bb = 0; for (int k = 1; k < NG; k++) if (bgh[k] > bgh[bb]) bb = k; fprintf(stderr, "gt-anchors: box %s gauge mode %.3f\n", bx[b].name, (bb + 0.5) * 14.0 / NG); }
            free(c);
        }
        if (pass == 0) { int gb = 0; for (int k = 1; k < NG; k++) if (gh[k] > gh[gb]) gb = k; gauge = (gb + 0.5) * 14.0 / NG; fprintf(stderr, "gt-anchors: the pins' gauge %.3f (mode of U'_ref - code / 18)\n", gauge); }
    }
    z3_close(zc);
    FILE *fo = fopen(outp, "wb"); if (!fo) { fprintf(stderr, "gt-anchors: cannot write %s\n", outp); return 1; }
    size_t nout = 0;
    for (size_t i = 0; i < M.cap; i++) if (M.key[i]) {
        const double *s = M.s[i], n = s[7]; float r[8];
        if (pin) { r[0] = (float)(s[0] / n); r[1] = (float)(s[1] / n); r[2] = (float)(s[2] / n); r[3] = (float)(s[3] / n); r[4] = (float)n; if (fwrite(r, 4, 5, fo) != 5) return 1; }
        else { for (int k = 0; k < 7; k++) r[k] = (float)(s[k] / n); r[7] = (float)n; if (fwrite(r, 4, 8, fo) != 8) return 1; }
        nout++;
    }
    if (fclose(fo)) return 1;
    const double vcm3 = vol * pow(g.um * 1e-4, 3);   /* the anchor boxes' volume (native voxels of um) */
    printf("gt-anchors: %s, p %g, seed %llu: %d anchor boxes (%.4f cm^3), %zu labelled voxels; %zu anchors (%.1f per cm^3) -> %zu merged records%s; %s %zu (%s)\n",
           kind, p, (unsigned long long)seed, nab, vcm3, nlab, nraw, vcm3 > 0 ? nraw / vcm3 : 0, nout, pin ? "" : "", pin ? "pins off the reference by >= 0.5 turn" : "pairs without a partner sheet",
           nmiss, outp);
    free(bx); fit_fields_free(&g); if (ref) fit_free(ref); free(M.key); free(M.s);
    return 0;
}
