/* sampbias (2026-11-14): the draw balance options of the sampler on the synthetic wind fixture.
   - edge_zboost_index: interior cells of a labelled box keep one draw, corner / edge cells get more (<= 4), "zboost" multiplies the low-z cells by K
   - sampler_vox_scale: 1 for sources at or above the weighted mean known share / K, the ratio below it, 1 while unmeasured
   - a sampler with --snap-jitter / --edge-boost / --stick-src / --vox-cap on delivers batches whose known-phase share is recorded
   - "hard" (hardspot 2026-11-27): the hard index of a map at another power-of-two um, the kept share of hard draws at frac 0.5, off = byte-identical */
#define WIND_FIXTURE_NO_MAIN
#include "make_wind_fixture.c"
#include "../src/sample.c"
#include "../src/hardmap.h"
#include <assert.h>
#include <sys/stat.h>

void *nn_host_alloc(size_t n) { return malloc(n); }
void nn_host_free(void *p) { free(p); }
static int failed;
static void check(const char *name, int ok) { printf("sampbias: %-72s %s\n", name, ok ? "ok" : "FAIL"); failed += !ok; }
int main(int argc, char **argv) {
    char tmpd[] = "/tmp/ufsm-sampbias-XXXXXX"; const char *dir = argc > 1 ? argv[1] : mkdtemp(tmpd);
    if (!dir || make_wind_fixture(dir, 256, 2.f)) { fprintf(stderr, "fixture failed\n"); return 1; }
    char path[1400]; snprintf(path, sizeof path, "%s/sources.json", dir);
    sources *S = sources_load(path); assert(S);
    sample_cfg c = sample_cfg_default();
    c.P = 64; c.B = 1; c.nworkers = 2; c.nbuf = 4; c.seed = 7; c.xfmt = 1; c.gpu_input = 1; c.side = 2; c.wind4 = 1;
    memset(c.level_p, 0, sizeof c.level_p); c.level_p[0] = 1; c.min_fg = 0;
    c.snap_jitter = 1; c.edge_boost = 1; c.stick = 4; c.vox_cap = 2;

    /* 1: the index. A 12^3 grid with the labelled box [3, 9)^3; window box n = P >> (lev - min_level) = 64 >> 4 = 4 cells (lev 4, min_level 0). */
    sampler fake = {0}; fake.cfg = c;
    source fs = S->src[0]; fs.band = (void *)1; fs.phase = (void *)1; fs.min_level = 0; fs.zb_below = 0; fs.zb_k = 0;
    const int64_t G = 12; uint32_t *idx = malloc(G * G * G * sizeof *idx); size_t n = 0;
    for (int64_t z = 3; z < 9; z++) for (int64_t y = 3; y < 9; y++) for (int64_t x = 3; x < 9; x++) idx[n++] = (uint32_t)((z * G + y) * G + x);
    occ_index o = {idx, n, 4, {G, G, G}};
    edge_zboost_index(&fake, &fs, &o);
    int cnt[12 * 12 * 12] = {0}; for (size_t k = 0; k < o.n; k++) cnt[o.idx[k]]++;
    const int interior = cnt[(6 * G + 6) * G + 6], corner = cnt[(3 * G + 3) * G + 3];
    check("edge-boost: an interior cell is drawn once", interior == 1);
    check("edge-boost: a corner cell is drawn 2-4 times", corner >= 2 && corner <= 4);
    free(o.idx);
    /* zboost: cells with z < 5 x3 */
    fake.cfg.edge_boost = 0; fs.zb_below = 5; fs.zb_k = 3;
    idx = malloc(G * G * G * sizeof *idx); n = 0;
    for (int64_t z = 3; z < 9; z++) for (int64_t y = 3; y < 9; y++) for (int64_t x = 3; x < 9; x++) idx[n++] = (uint32_t)((z * G + y) * G + x);
    o = (occ_index){idx, n, 4, {G, G, G}};
    edge_zboost_index(&fake, &fs, &o);
    memset(cnt, 0, sizeof cnt); for (size_t k = 0; k < o.n; k++) cnt[o.idx[k]]++;
    check("zboost: z < 5 cells x3, others x1", cnt[(3 * G + 5) * G + 5] == 3 && cnt[(4 * G + 5) * G + 5] == 3 && cnt[(5 * G + 5) * G + 5] == 1);
    free(o.idx);

    /* 2: the loss scale of a source with a small known share */
    sources fs2 = {0}; fs2.n = 3; fake.S = &fs2; double cum[3] = {1, 2, 3}; fake.cum = cum; fake.cfg.vox_cap = 2;
    check("vox scale: 1 while nothing is measured", sampler_vox_scale(&fake, 0) == 1.f);
    const double sh[3] = {0.66, 0.60, 0.056};
    for (int j = 0; j < 3; j++) { atomic_store(&fake.ksh_n[j], 10); atomic_store(&fake.ksh_sum[j], sh[j] * 10); }
    const double ref = (0.66 + 0.60 + 0.056) / 3;
    check("vox scale: a source at the mean keeps 1", sampler_vox_scale(&fake, 0) == 1.f);
    check("vox scale: 0.056 share -> K x share / mean", fabsf(sampler_vox_scale(&fake, 2) - (float)(2 * 0.056 / ref)) < 1e-4f);

    /* 3: a sampler with every option on delivers windows and records the known share */
    c.nworkers = 2; sampler *sp = sampler_start(S, &c); assert(sp);
    int got = 0; for (int i = 0; i < 6; i++) { batch *b = sampler_next(sp); if (!b) break; got++; sampler_release(sp, b); }
    check("sampler with --snap-jitter --edge-boost --stick-src --vox-cap delivers 6 batches", got == 6);
    check("known-phase share recorded for source 0", atomic_load(&sp->ksh_n[0]) >= 1);
    sampler_stop(sp);

    /* 4: "hard" (hardspot 2026-11-27): a map at 4.8 um (the source is 9.6: a power of two apart), lev 5 = 32 map voxels = 16 source voxels a
       cell, 16^3 cells; cells with z < 8 (source z < 128: half the fixture's level-7 draw cells) at weight 255 (8 repeats). Loaded through the json. */
    {
        char hm[1500], hj[1500], sj[1500]; snprintf(hm, sizeof hm, "%s/hard.raw", dir); snprintf(hj, sizeof hj, "%s/hard.raw.json", dir); snprintf(sj, sizeof sj, "%s/sources-hard.json", dir);
        uint8_t map[16 * 16 * 16] = {0}; for (int k = 0; k < 8 * 16 * 16; k++) map[k] = 255;
        FILE *f = fopen(hm, "wb"); fwrite(map, 1, sizeof map, f); fclose(f);
        f = fopen(hj, "w"); fprintf(f, "{\"grid\": [16, 16, 16], \"lev\": 5, \"um\": 4.8, \"ct\": \"ct\"}\n"); fclose(f);
        FILE *a = fopen(path, "rb"); char txt[8192]; size_t tn = fread(txt, 1, sizeof txt - 1, a); fclose(a); txt[tn] = 0;
        char *w = strstr(txt, "\"weight\": 1,"); assert(w);
        f = fopen(sj, "w"); fprintf(f, "%.*s\"hard\": {\"map\": \"%s\", \"frac\": 0.5}, %s", (int)(w - txt), txt, hm, w); fclose(f);
        sources *SH = sources_load(sj);
        check("hard: the json's map loads (grid, lev, um, frac)", SH && SH->src[0].hard && SH->src[0].hard_lev == 5 && SH->src[0].hard_grid[0] == 16 && SH->src[0].hard_frac == 0.5f);
        sample_cfg ch = sample_cfg_default();
        ch.P = 64; ch.B = 1; ch.nworkers = 2; ch.nbuf = 4; ch.seed = 7; ch.xfmt = 1; ch.gpu_input = 1; ch.side = 2; ch.wind4 = 1;
        memset(ch.level_p, 0, sizeof ch.level_p); ch.level_p[0] = 1; ch.min_fg = 0; ch.deterministic = 1;   /* (a fixed draw sequence: the share check cannot flake) */
        sampler *sh = SH ? sampler_start(SH, &ch) : nullptr; assert(sh);
        const int lev = sh->occ[0].lev; const int64_t *g = sh->occ[0].shape; const double cs = ldexp(1.0, lev);
        size_t want = 0; for (size_t k = 0; k < sh->occ[0].n; k++) { const uint32_t id = sh->occ[0].idx[k]; want += (id / (g[1] * g[2]) + 0.5) * cs < 128; }
        int allin = 1; for (size_t k = 0; k < sh->occ[0].hn; k++) allin &= (sh->occ[0].hidx[k] / (g[1] * g[2]) + 0.5) * cs < 128;
        printf("sampbias: hard: occupancy level %d, %zu draw cells, %zu with z < 128, %zu hard entries\n", lev, sh->occ[0].n, want, sh->occ[0].hn);
        check("hard: the index holds exactly the cells below z 128, 8 times each", want > 0 && sh->occ[0].hn == 8 * want && allin && sh->occ[0].hfrac == 0.5f);
        int got2 = 0; for (int i = 0; i < 80; i++) { batch *b = sampler_next(sh); if (!b) break; got2++; sampler_release(sh, b); }
        uint64_t hs[3]; sampler_hard_stats(sh, 0, hs); const double share = hs[1] ? (double)hs[0] / hs[1] : -1;
        printf("sampbias: hard: %llu of %llu kept windows from hard draws (%.3f, frac 0.5)\n", (unsigned long long)hs[0], (unsigned long long)hs[1], share);
        check("hard: frac 0.5 -> 0.3-0.7 of the kept windows (binomial sd ~0.055 at 80)", got2 == 80 && hs[1] >= 80 && share >= 0.3 && share <= 0.7);
        check("hard: every hard draw lands on a hard cell", hs[2] >= hs[0] && hs[2] <= hs[1]);
        sampler_stop(sh);
        /* off: --hard-frac -1 draws the same windows as the source without a map (deterministic batches, every plane's hash) */
        ch.deterministic = 1; ch.nworkers = 2;
        uint64_t ha[8][16] = {{0}}, hb[8][16] = {{0}}; int na = 0, nb = 0;
        sampler *s0 = sampler_start(S, &ch); assert(s0);
        for (int i = 0; i < 8; i++) { batch *b = sampler_next(s0); const char *nm[16]; na = sampler_batch_hash(s0, b, nm, ha[i], 16); sampler_release(s0, b); }
        sampler_stop(s0);
        ch.hard_frac = -1; sampler *s1 = sampler_start(SH, &ch); assert(s1);
        check("hard: --hard-frac -1 builds no hard index", s1->occ[0].hn == 0);
        for (int i = 0; i < 8; i++) { batch *b = sampler_next(s1); const char *nm[16]; nb = sampler_batch_hash(s1, b, nm, hb[i], 16); sampler_release(s1, b); }
        sampler_stop(s1);
        check("hard: off -> 8 batches byte-identical to the source without a map", na > 0 && na == nb && !memcmp(ha, hb, sizeof ha));
        ch.hard_frac = 0; ch.deterministic = 1; sampler *s2 = sampler_start(SH, &ch); assert(s2);
        int diff = 0; for (int i = 0; i < 8; i++) { batch *b = sampler_next(s2); const char *nm[16]; uint64_t hh[16] = {0}; sampler_batch_hash(s2, b, nm, hh, 16); diff |= memcmp(hh, ha[i], sizeof hh) != 0; sampler_release(s2, b); }
        sampler_stop(s2);
        check("hard: on (frac 0.5) changes the deterministic draws", diff);
        sources_free(SH);
    }
    /* 5: ufsm hard-map on a synthetic mf plane: 64^3 field cells of h 2 x 4.8 um (origin 0), out 2.4 um lev 5 (8 field cells a side, 8^3
       cells for 256^3 voxels). Field z < 32 material: mf 200 (flagged) at x < 32, 50 at x >= 32; z >= 32 off material (0). A box
       "0 0 224 32 32 256 0.5" sets cell (0, 0, 7) to 128. Expected: z < 4, x < 4 -> 255; (0, 0, 7) -> 128; the rest 0. */
    {
        char md[1500], fj[1500], ob[1500], bt[1500]; snprintf(md, sizeof md, "%s/hmf/mf", dir); snprintf(fj, sizeof fj, "%s/hmf/fields.json", dir);
        snprintf(ob, sizeof ob, "%s/hmf/hard.raw", dir); snprintf(bt, sizeof bt, "%s/hmf/boxes.tsv", dir);
        char hd[1500]; snprintf(hd, sizeof hd, "%s/hmf", dir); mkdir(hd, 0755);
        const int64_t g = 64; uint8_t *v = calloc((size_t)g * g * g, 1);
        for (int64_t z = 0; z < 32; z++) for (int64_t y = 0; y < g; y++) for (int64_t x = 0; x < g; x++) v[(z * g + y) * g + x] = x < 32 ? 200 : 50;
        const int64_t gs[3] = {g, g, g}; fx_volctx vc = {v, g}; const int wr = fx_write_arr(md, gs, 0.f, 0, fx_cut, &vc); free(v);
        FILE *f = fopen(fj, "w"); fprintf(f, "{\"origin\": [0, 0, 0], \"shape\": [64, 64, 64], \"h\": 2, \"um\": 4.8}\n"); fclose(f);
        f = fopen(bt, "w"); fprintf(f, "# z0 y0 x0 z1 y1 x1 w\n0 0 224 32 32 256 0.5\n"); fclose(f);
        char *av[] = {"ufsm", "hard-map", "--out", ob, "--um", "2.4", "--shape", "256,256,256", "--lev", "5", "--mf", md, "--mf-thr", "179", "--boxes", bt, "--threads", "2", nullptr};
        const int rc = wr ? -1 : cmd_hard_map(18, av);
        uint8_t hm[512] = {0}; f = rc ? nullptr : fopen(ob, "rb"); const size_t got3 = f ? fread(hm, 1, sizeof hm + 1, f) : 0; if (f) fclose(f);
        int ok = rc == 0 && got3 == 512;
        for (int z = 0; ok && z < 8; z++) for (int y = 0; y < 8; y++) for (int x = 0; x < 8; x++) {
            const int want = z == 0 && y == 0 && x == 7 ? 128 : z < 4 && x < 4 ? 255 : 0; ok &= hm[(z * 8 + y) * 8 + x] == want; }
        check("hard-map: mf share -> 255 / 0 by cell, a box at w 0.5 -> 128, 8^3 grid", ok);
    }
    printf("sampbias: %s\n", failed ? "FAILED" : "passed");
    return failed;
}
