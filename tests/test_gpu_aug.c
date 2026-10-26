/* train --gpu-aug: the device's CT channel (src/nn_aug.cu nn_aug_ct) against the sampler's CPU path on synthetic windows. Per case:
   scan_aug_plan on the window's bytes, then (CPU) scan_aug_exec + the z-score + write_x_ct (cube symmetry) or write_spatial_tm (warp)
   and (GPU) nn_aug_ct from the same plan and radial record. Checks: the output bytes' zero set exactly; the bytes within float tolerance
   (the device works in 16-bit storage: mean |d| < 0.15, |d| > 2 on < 0.5% of the voxels, max 8 -- blur / paganin / GIN accumulate
   rounding that the final rounding mostly absorbs); the 16-bit CT channel within the bytes' difference times 1 / sd; with no scanner op
   the channel bit for bit (symmetry) or within 2e-3 (warp: FMA); the jitter's noise N(0, 1) and the gain / offset as given. Every op
   alone at strength 1 and 2, mixtures, with and without an axis, warped and not. Usage: test_gpu_aug [P 96] */
#define WIND_FIXTURE_NO_MAIN
#include "make_wind_fixture.c"
#include "../src/sample.c"
#include "nn.h"


static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)
static float h2f_host(uint16_t h) { _Float16 t; memcpy(&t, &h, 2); return (float)t; }

int main(int argc, char **argv) {
    const int P = argc > 1 ? atoi(argv[1]) : 96; const size_t p3 = (size_t)P * P * P;
    nn_init(0);
    uint8_t *ct = malloc(p3), *tgt = malloc(2 * p3 + 64), *msk = malloc(p3), *ign = malloc(p3), *TM = malloc(p3);
    uint8_t *cin = malloc(p3), *cpu = malloc(p3), *gpu = malloc(p3); void *scr = malloc(5 * p3);
    uint16_t *Hc = malloc(2 * p3), *Hg = malloc(2 * p3), *H0 = malloc(2 * p3); float *app = malloc(4 * p3), *nrow = malloc(4 * (size_t)P);
    float cy0[1024], cx0[1024], cy[1024], cx[1024];
    for (int z = 0; z < P; z++) { cy0[z] = P * .5f + .1f * z - 300; cx0[z] = P * .4f - 200; }
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {   /* sheets around an axis outside the window, some air */
        const size_t k = ((size_t)z * P + y) * P + x;
        const double r = hypot(y - cy0[z], x - cx0[z]), q = r / 14.0 + .002 * z, f = q - floor(q);
        ct[k] = (uint8_t)(x < 5 || (x + y) % 97 == 0 ? 0 : f < .35 ? 140 + (k * 2654435761u >> 28) : 40 + (k * 40503u >> 29));
        tgt[k] = f < .2 ? 255 : 0; tgt[p3 + k] = f >= .2 && f < .35 ? 255 : 0;
    }
    float g_t[256]; for (int v = 0; v < 256; v++) g_t[v] = v < 120 ? 0.f : v >= 200 ? 1.f : (v - 119) / 81.f;
    scan_plan *pl = malloc(sizeof *pl); scan_ct_rec *rec = nn_host_alloc(sizeof *rec);
    uint8_t *cin_p = nn_host_alloc(p3);   /* (pinned: the trainer's batches are) */
    void *rad_h = malloc(NN_RAD_REC(P)), *rad_d = nn_malloc(NN_RAD_REC(P)), *Hd = nn_malloc(2 * p3);
    const int64_t o[3] = {1000, 2000, 3000}; const int chunk[3] = {32, 32, 32};
    double worst_mean = 0; int worst_max = 0, ncase = 0, nwarp = 0;
    for (int op = -2; op < SA_N; op++) for (int sv = 0; sv < 4; sv++) {
        scan_aug_cfg c = {.strength = sv & 2 ? 2.f : 1.f, .drop_ignore = 1, .hist = g_t, .nhist = 1};
        if (op >= 0) c.p[op] = 1; else if (op == -1) for (int i = 0; i < SA_N; i++) c.p[i] = .45f;   /* (-2: no scanner op) */
        const uint64_t seed = 77 + 1009 * (uint64_t)(op + 2) + 31 * sv; const int hasax = sv & 1, warped = (op + sv) % 2 == 0;
        memcpy(cin, ct, p3); memset(ign, 0, p3); memcpy(cy, cy0, sizeof cy); memcpy(cx, cx0, sizeof cx);
        uint8_t *tg = malloc(2 * p3); memcpy(tg, tgt, 2 * p3);
        const unsigned on = scan_aug_plan(&c, seed, cin, tg, 2, ign, P, o, cy, cx, hasax, chunk, 1, pl);
        memcpy(cpu, cin, p3);
        if (on) scan_aug_exec(pl, cpu, scr);
        size_t nz; double sum, sq; u8_stats(cpu, p3, &nz, &sum, &sq);
        const double mean = sum / (double)p3, var = sq / (double)p3 - mean * mean, sd = sqrt(var > 0 ? var : 0) + 1e-3;
        sym sy = sym_of((int)((seed * 7) % 48)); spatial_aug sa = {0};
        if (warped) { spatial_aug_make(&sa, seed, P, 10, 1, 1, 1); spatial_aug_zoom(&sa, .8); nwarp++; }
        rng r; rseed(&r, seed);
        for (size_t k = 0; k < p3; k++) msk[k] = cpu[k] != 0 && !ign[k];
        if (warped) write_spatial_tm(cpu, tg, msk, nullptr, nullptr, P, o, (float)mean, (float)(1.0 / sd), hasax, cy, cx, sy, &sa, 1.f, 0.f, 0.f, &r, nrow, 1,
                                     nullptr, Hc, nullptr, nullptr, nullptr, nullptr, nullptr, app, 1, 1, TM);
        else write_x_ct(cpu, P, P, (float)mean, (float)(1.0 / sd), sy, 0, 1.f, 0.f, 0.f, &r, 1, Hc);
        rad_record(rad_h, o, P, P, hasax, sy, cy, cx, warped ? &sa : nullptr);
        nn_h2d(rad_d, rad_h, NN_RAD_REC(P));
        memset(rec, 0, sizeof *rec); rec->scan = on != 0; rec->xfmt = 1; if (on) rec->plan = *pl;
        memcpy(cin_p, cin, p3);
        nn_aug_ct(cin_p, rec, rad_d, P, Hd, 0);
        nn_aug_last_bytes(gpu, P); nn_d2h(Hg, Hd, 2 * p3);
        size_t zbad = 0, big = 0; double ad = 0; int mx = 0;
        for (size_t k = 0; k < p3; k++) {
            const int d = abs((int)cpu[k] - (int)gpu[k]); zbad += (cpu[k] == 0) != (gpu[k] == 0); ad += d; big += d > 2; if (d > mx) mx = d;
        }
        ad /= (double)p3;
        double hmax = 0; for (size_t k = 0; k < p3; k++) { const double d = fabs(h2f_host(Hc[k]) - h2f_host(Hg[k])); if (d > hmax) hmax = d; }
        const char *nm = op >= 0 ? scan_aug_names[op] : op == -1 ? "mixture" : "none";
        CHECK(!zbad, "%s s%d: %zu voxels with a different zero set", nm, sv, zbad);
        CHECK(ad < .15 && big < p3 / 200 && mx <= 8, "%s s%d (on %x): bytes mean |d| %.3f, %zu > 2, max %d", nm, sv, on, ad, big, mx);
        if (!on) CHECK(warped ? hmax < 2e-3 : hmax == 0, "%s s%d: unaugmented CT channel max |d| %.2g (%s)", nm, sv, hmax, warped ? "warp" : "symmetry");
        else CHECK(hmax <= (mx + 1.01) / sd + 2e-3, "%s s%d: CT channel max |d| %.3g beyond the bytes' %d / sd %.1f", nm, sv, hmax, mx, sd);
        if (ad > worst_mean) worst_mean = ad;
        if (mx > worst_max) worst_max = mx;
        ncase++;
        if (op == -2 && sv == 0) {   /* the jitter: (H - (ia H0 + ib)) / isg ~ N(0, 1) */
            memcpy(H0, Hg, 2 * p3);
            rec->jitter = 1; rec->ia = 1.1f; rec->ib = .05f; rec->isg = .08f; rec->nseed = 12345;
            nn_aug_ct(cin_p, rec, rad_d, P, Hd, 0); nn_d2h(Hg, Hd, 2 * p3);
            double s1 = 0, s2 = 0; for (size_t k = 0; k < p3; k++) { const double e = (h2f_host(Hg[k]) - (1.1 * h2f_host(H0[k]) + .05)) / .08; s1 += e; s2 += e * e; }
            const double m = s1 / (double)p3, v = s2 / (double)p3 - m * m;
            CHECK(fabs(m) < .02 && fabs(v - 1) < .05, "jitter noise mean %.4f var %.4f", m, v);
        }
        free(tg);
    }
    int e2e_n = 0, e2e_scan = 0, e2e_warp = 0; double e2e_worst = 0; size_t e2e_lbad[4] = {0};
    {   /* end to end: the --task unwrap2 sampler with --gpu-aug 0 and 1 (same seed, deterministic): phase / winding / packed targets / radial
           records byte for byte; the device's CT channel from ct8 + aug against the CPU's x16 within the two jitter noises */
        enum { PW = 64, NB = 16 }; const size_t q3 = (size_t)PW * PW * PW;
        char tmpd[] = "/tmp/ufsm-gpuaug-XXXXXX"; const char *dir = mkdtemp(tmpd);
        CHECK(dir && !make_wind_fixture(dir, 256, 2.f), "wind fixture");
        char path[1400]; snprintf(path, sizeof path, "%s/sources.json", dir);
        sources *S = sources_load(path); CHECK(S != nullptr, "sources");
        sample_cfg c = sample_cfg_default();
        c.P = PW; c.B = 1; c.nworkers = 2; c.nbuf = 3; c.deterministic = 1; c.seed = 11; c.xfmt = 1; c.gpu_input = 1; c.side = 2; c.wind4 = 1; c.pack_tm = 1;
        memset(c.level_p, 0, sizeof c.level_p); c.level_p[0] = 1; c.min_fg = 0;
        CHECK(!sample_aug_preset(&c, "all"), "preset");
        c.augment = 3; c.symmetry_p = .5f; c.rotate_p = .5f; c.zoom_p = .3f; c.elastic_p = .3f; c.zoom_out_p = 0;
        memset(c.scan.p, 0, sizeof c.scan.p); c.scan.strength = 1;
        const int sops[] = {SA_BLUR, SA_LOWRES, SA_NOISE, SA_CNOISE, SA_TONE, SA_PAGANIN, SA_GIN, SA_BIAS, SA_RING, SA_SLICE, SA_QUANT, SA_CHUNK, SA_CUTOUT, SA_DROPSLICE};
        for (size_t k = 0; k < sizeof sops / sizeof *sops; k++) c.scan.p[sops[k]] = .1f;   /* 0.9^14: ~1 in 4 windows without scanner ops */
        sample_cfg cg = c, cg2 = c; cg.gpu_aug = 1; cg2.gpu_aug = 2;
        CHECK(sample_gpu_aug_ok(&cg), "sample_gpu_aug_ok on the unwrap2 recipe");
        CHECK(sample_gpu_lwarp_ok(&cg2), "sample_gpu_lwarp_ok on the unwrap2 recipe");
        uint16_t *xa[NB]; uint8_t *lab[3][NB], *c8[NB], *c82[NB]; void *rr[3][NB]; scan_ct_rec *ra[NB];
        for (int run = 0; run < 3; run++) {   /* (--gpu-aug 0, 1, 2) */
            sampler *sp = sampler_start(S, run == 2 ? &cg2 : run ? &cg : &c); CHECK(sp != nullptr, "sampler %d", run);
            for (int i = 0; sp && i < NB; i++) {
                batch *b = sampler_next(sp);
                lab[run][i] = malloc(4 * q3); memcpy(lab[run][i], b->side, q3); memcpy(lab[run][i] + q3, b->w4, q3); memcpy(lab[run][i] + 2 * q3, b->wcomp, q3); memcpy(lab[run][i] + 3 * q3, b->tm, q3);
                rr[run][i] = malloc(NN_RAD_REC(PW)); memcpy(rr[run][i], b->rad, NN_RAD_REC(PW));
                if (!run) { CHECK(b->x16 && !b->ct8, "CPU batch layout"); xa[i] = malloc(2 * q3); memcpy(xa[i], b->x16, 2 * q3); }
                else if (run == 1) { CHECK(!b->x16 && b->ct8 && b->aug, "--gpu-aug batch layout"); c8[i] = nn_host_alloc(q3); memcpy(c8[i], b->ct8, q3); ra[i] = nn_host_alloc(sizeof *ra[i]); memcpy(ra[i], b->aug, sizeof *ra[i]); }
                else {   /* --gpu-aug 2: the same draws (the same bytes and records as --gpu-aug 1) */
                    CHECK(!b->x16 && b->ct8 && !memcmp(b->ct8, c8[i], q3), "window %d: --gpu-aug 2 CT bytes differ from --gpu-aug 1", i);
                    { const scan_ct_rec *g = b->aug; CHECK(!memcmp(g, ra[i], offsetof(scan_ct_rec, plan)), "window %d: --gpu-aug 2 record head differs (scan %d/%d nseed %llx/%llx)", i, g->scan, ra[i]->scan, (unsigned long long)g->nseed, (unsigned long long)ra[i]->nseed);
                      if (g->scan && memcmp(&g->plan, &ra[i]->plan, sizeof g->plan)) { size_t o = 0, nd = 0, last = 0; const uint8_t *x0 = (const uint8_t *)&g->plan, *x1 = (const uint8_t *)&ra[i]->plan; for (size_t q = 0; q < sizeof g->plan; q++) if (x0[q] != x1[q]) { if (!nd) o = q; nd++; last = q; }
                          CHECK(0, "window %d: --gpu-aug 2 scan plan differs (%zu bytes, offsets %zu..%zu)", i, nd, o, last); } }
                    c82[i] = nn_host_alloc(4 * q3); memcpy(c82[i], lab[2][i], 4 * q3);   /* (pinned: nn_label_warp's sources) */
                }
                sampler_release(sp, b);
            }
            if (sp) sampler_stop(sp);
        }
        void *rd = nn_malloc(NN_RAD_REC(PW)), *hd = nn_malloc(2 * q3); uint16_t *hg = malloc(2 * q3);
        for (int i = 0; i < NB; i++) {
            CHECK(!memcmp(lab[0][i], lab[1][i], 4 * q3), "window %d: phase / winding / packed targets differ between --gpu-aug 0 and 1", i);
            CHECK(!memcmp(rr[0][i], rr[1][i], NN_RAD_REC(PW)), "window %d: radial records differ", i);
            CHECK(!memcmp(rr[0][i], rr[2][i], NN_RAD_REC(PW)), "window %d: --gpu-aug 2 radial records differ", i);
            int32_t hh[8]; memcpy(hh, rr[1][i], sizeof hh); const int warped = hh[6] & 1;
            nn_h2d(rd, rr[1][i], NN_RAD_REC(PW));
            if (!warped) CHECK(!memcmp(lab[0][i], lab[2][i], 4 * q3), "window %d (unwarped): --gpu-aug 2 labels differ from the CPU's", i);
            else {   /* --gpu-aug 2: the device's warp of the source-grid labels against the CPU's warped labels */
                const float *cs, *sn; sample_phase_cssn(&cs, &sn);
                uint8_t *dl = nn_malloc(4 * q3), *gl = malloc(4 * q3);
                nn_label_warp(1, c82[i], nullptr, rd, PW, dl, nullptr, cs, sn, 0);
                nn_label_warp(2, c82[i] + q3, c82[i] + 2 * q3, rd, PW, dl + q3, dl + 2 * q3, cs, sn, 0);
                nn_label_warp(0, c82[i] + 3 * q3, nullptr, rd, PW, dl + 3 * q3, nullptr, cs, sn, 0);
                nn_d2h(gl, dl, 4 * q3);
                size_t bad[4] = {0};
                for (int pl = 0; pl < 4; pl++) for (size_t k = 0; k < q3; k++) bad[pl] += gl[pl * q3 + k] != lab[0][i][pl * q3 + k];
                for (int pl = 0; pl < 4; pl++) if (bad[pl] > e2e_lbad[pl]) e2e_lbad[pl] = bad[pl];
                CHECK(bad[0] <= q3 / 10000 && bad[1] <= q3 / 10000 && bad[2] <= q3 / 10000 && bad[3] <= q3 / 10000,
                      "window %d: device-warped labels differ from the CPU's at %zu phase / %zu w4 / %zu component / %zu packed voxels of %zu", i, bad[0], bad[1], bad[2], bad[3], q3);
                nn_free(dl); free(gl);
            }
            nn_host_free(c82[i]); free(lab[2][i]); free(rr[2][i]);
            nn_aug_ct(c8[i], ra[i], rd, PW, hd, 0); nn_d2h(hg, hd, 2 * q3);
            double s1 = 0, s2 = 0; size_t zbad = 0;
            for (size_t k = 0; k < q3; k++) {
                const double a = h2f_host(xa[i][k]), g = h2f_host(hg[k]), d = g - a; s1 += d; s2 += d * d;
                if (warped) zbad += (xa[i][k] == 0) != (hg[k] == 0) && (a == 0 ? fabs(g) : fabs(a)) > 1e-6;   /* (outside the warp: exactly 0 in both) */
            }
            const double m = s1 / (double)q3, sdv = sqrt(fmax(s2 / (double)q3 - m * m, 0)), lim = sqrt(2.) * ra[i]->isg + .08;
            CHECK(fabs(m) < .03 + ra[i]->isg * .05 && sdv < lim, "window %d (scan %d, warp %d, isg %.3f): CT channel difference mean %.4f sd %.4f (limit %.3f)", i, ra[i]->scan, warped, ra[i]->isg, m, sdv, lim);
            CHECK(zbad < q3 / 1000, "window %d: %zu voxels inside the warp in one channel, outside in the other", i, zbad);
            if (sdv - sqrt(2.) * ra[i]->isg > e2e_worst) e2e_worst = sdv - sqrt(2.) * ra[i]->isg;
            e2e_n++; e2e_scan += ra[i]->scan; e2e_warp += warped;
            free(xa[i]); free(lab[0][i]); free(lab[1][i]); free(rr[0][i]); free(rr[1][i]); nn_host_free(c8[i]); nn_host_free(ra[i]);
        }
        nn_free(rd); nn_free(hd); free(hg);
        CHECK(e2e_scan > 0 && e2e_scan < e2e_n && e2e_warp > 0, "the windows cover scanner ops on / off and warps (%d / %d scan, %d warped)", e2e_scan, e2e_n, e2e_warp);
        char rm[1500]; snprintf(rm, sizeof rm, "rm -rf %s", dir); if (system(rm)) {}
    }
    printf("test_gpu_aug: end to end %d unwrap2 windows (%d with scanner ops, %d warped), CT channel sd beyond the jitter noises <= %.3f\n", e2e_n, e2e_scan, e2e_warp, e2e_worst);
    printf("test_gpu_aug: --gpu-aug 2 device-warped labels, worst window: %zu phase / %zu w4 / %zu component / %zu packed voxels off the CPU's\n", e2e_lbad[0], e2e_lbad[1], e2e_lbad[2], e2e_lbad[3]);
    printf("test_gpu_aug: %d cases (%d warped) at P %d: worst bytes mean |d| %.3f, max %d: %s\n", ncase, nwarp, P, worst_mean, worst_max, fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
