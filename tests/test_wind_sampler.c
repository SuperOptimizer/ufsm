/* --task unwrap2 sampling end to end on a synthetic band + stored-phase source (tests/make_wind_fixture.c: no recto / ink targets):
   test_wind_sampler [DIR] (the fixture is written into DIR, default a temporary directory removed at the end).
   - packed targets (--pack-targets, what train now picks for target-less sources) give the same batches as unpacked ones: the code
     bytes decode to the unpacked target / mask bytes, the CT channel, radial records, phase, winding and component bytes are equal
     (warped windows: the packed fast warp vs the trilinear one on all-zero targets)
   - target-less sources are not rejected as empty (every draw that passes the CT tests is kept)
   - deterministic batches do not depend on the worker or slot count */
#define WIND_FIXTURE_NO_MAIN
#include "make_wind_fixture.c"
#include "../src/sample.c"
#include <assert.h>

void *nn_host_alloc(size_t n) { return malloc(n); }
void nn_host_free(void *p) { free(p); }
static int failed;
static void check(const char *name, int ok) { printf("wind sampler: %-72s %s\n", name, ok ? "ok" : "FAIL"); failed += !ok; }
enum { NB = 10, PW = 64 };
typedef struct { uint16_t *x; uint8_t *t, *m, *tm, *side, *w4, *wc; void *rad; int64_t corner[3]; } snap;
static void take(sampler *sp, snap *o, size_t p3) {
    for (int i = 0; i < NB; i++) {
        batch *b = sampler_next(sp); assert(b);
        o[i].x = malloc(p3 * 2); memcpy(o[i].x, b->x16, p3 * 2);
        o[i].rad = malloc(NN_RAD_REC(PW)); memcpy(o[i].rad, b->rad, NN_RAD_REC(PW));
        if (b->tm) { o[i].tm = malloc(p3); memcpy(o[i].tm, b->tm, p3); o[i].t = o[i].m = nullptr; }
        else { o[i].t = malloc(2 * p3); memcpy(o[i].t, b->t, 2 * p3); o[i].m = malloc(p3); memcpy(o[i].m, b->m, p3); o[i].tm = nullptr; }
        o[i].side = malloc(p3); memcpy(o[i].side, b->side, p3);
        o[i].w4 = malloc(p3); memcpy(o[i].w4, b->w4, p3); o[i].wc = malloc(p3); memcpy(o[i].wc, b->wcomp, p3);
        memcpy(o[i].corner, b->corner[0], sizeof o[i].corner);
        sampler_release(sp, b);
    }
}
static void drop(snap *o) { for (int i = 0; i < NB; i++) { free(o[i].x); free(o[i].rad); free(o[i].t); free(o[i].m); free(o[i].tm); free(o[i].side); free(o[i].w4); free(o[i].wc); } }
int main(int argc, char **argv) {
    char tmpd[] = "/tmp/ufsm-wind-XXXXXX"; const char *dir = argc > 1 ? argv[1] : mkdtemp(tmpd);
    if (!dir || make_wind_fixture(dir, 256, 2.f)) { fprintf(stderr, "fixture failed\n"); return 1; }
    char path[1400]; snprintf(path, sizeof path, "%s/sources.json", dir);
    sources *S = sources_load(path); assert(S);
    sample_cfg c = sample_cfg_default();
    c.P = PW; c.B = 1; c.nworkers = 3; c.nbuf = 5; c.deterministic = 1; c.seed = 7; c.xfmt = 1; c.gpu_input = 1; c.side = 2; c.wind4 = 1;
    memset(c.level_p, 0, sizeof c.level_p); c.level_p[0] = 1; c.min_fg = 0;
    assert(!sample_aug_preset(&c, "all"));
    c.augment = 3; c.symmetry_p = .5f; c.rotate_p = .5f; c.zoom_p = .3f; c.elastic_p = .3f; c.zoom_out_p = 0;
    const size_t p3 = (size_t)PW * PW * PW;
    snap a[NB], b[NB], d[NB];
    c.pack_tm = 0; sampler *sp = sampler_start(S, &c); assert(sp); take(sp, a, p3);
    uint64_t prod, rej; sampler_stats(sp, &prod, &rej); sampler_stop(sp);
    check("target-less source: no draw rejected (min_fg 0: no CT rejection either)", rej == 0);
    c.pack_tm = 1; sp = sampler_start(S, &c); assert(sp); take(sp, b, p3); sampler_stop(sp);
    int same = 1, warped = 0, known = 0;
    for (int i = 0; i < NB; i++) {
        int32_t hd[8]; memcpy(hd, b[i].rad, sizeof hd); warped += hd[6] & 1;
        same &= !memcmp(a[i].x, b[i].x, p3 * 2) && !memcmp(a[i].rad, b[i].rad, NN_RAD_REC(PW)) && !memcmp(a[i].side, b[i].side, p3) &&
                !memcmp(a[i].w4, b[i].w4, p3) && !memcmp(a[i].wc, b[i].wc, p3) && !memcmp(a[i].corner, b[i].corner, sizeof a[i].corner);
        for (size_t k = 0; k < p3; k++) {
            const uint8_t q = b[i].tm[k];
            same &= a[i].t[k] == ((q & SAMPLE_TM_T0) ? 255 : 0) && a[i].t[p3 + k] == ((q & SAMPLE_TM_T1) ? 255 : 0) && a[i].m[k] == ((q & SAMPLE_TM_M) ? 1 : 0);
            known += a[i].side[k] < PHASE_PERIOD;
        }
    }
    printf("wind sampler: %d of %d windows warped, %.3f of the voxels with a known phase\n", warped, NB, known / (double)(NB * p3));
    check("packed targets decode to the unpacked batches (all other arrays equal)", same && warped > 0 && warped < NB && known > 0);
    c.nworkers = 1; c.nbuf = 2; sp = sampler_start(S, &c); assert(sp); take(sp, d, p3); sampler_stop(sp);
    int det = 1;
    for (int i = 0; i < NB; i++) det &= !memcmp(d[i].x, b[i].x, p3 * 2) && !memcmp(d[i].tm, b[i].tm, p3) && !memcmp(d[i].w4, b[i].w4, p3) && !memcmp(d[i].side, b[i].side, p3);
    check("deterministic batches: 3 workers / 5 slots = 1 worker / 2 slots", det);
    drop(a); drop(b); drop(d); sources_free(S);
    {   /* sparse labels: a stored phase known in one of the 8 chunks of a 512 fixture's level-1 grid; windows are drawn around it
           (the phase-chunk occupancy index), none rejected for lack of CT, most of them labelled */
        char d2[1500]; snprintf(d2, sizeof d2, "%s/sparse", dir);
        if (make_wind_fixture(d2, 512, 4.f)) { fprintf(stderr, "fixture failed\n"); return 1; }
        const int h = 256; uint8_t *ph = malloc((size_t)h * h * h); assert(ph);
        for (int z = 0; z < h; z++) for (int y = 0; y < h; y++) for (int x = 0; x < h; x++) {
            const double w = fx_wind(2.0 * z + .5, 2.0 * y + .5, 2.0 * x + .5), f = w - floor(w);
            const int in = z >= 128 && y < 128 && x >= 128; int cc = (int)(f * PHASE_PERIOD); if (cc >= PHASE_PERIOD) cc = PHASE_PERIOD - 1;
            ph[((size_t)z * h + y) * h + x] = in ? (uint8_t)cc : PHASE_UNKNOWN;
        }
        char pp[1600]; snprintf(pp, sizeof pp, "%s/phase/1", d2); char cmd[1700]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", pp); if (system(cmd)) return 1;
        const int64_t hs[3] = {h, h, h}; fx_volctx cv = {ph, h};
        if (fx_write_arr(pp, hs, 0.f, PHASE_UNKNOWN, fx_cut, &cv)) return 1;
        free(ph);
        snprintf(path, sizeof path, "%s/sources.json", d2);
        sources *S2 = sources_load(path); assert(S2);
        sample_cfg c2 = c; c2.nworkers = 2; c2.nbuf = 4; c2.min_fg = .3; c2.geometry_augment = 0; memset(c2.scan.p, 0, sizeof c2.scan.p);
        sampler *sq = sampler_start(S2, &c2); assert(sq);
        int inbox = 1; size_t kn = 0;
        for (int i = 0; i < NB; i++) {
            batch *bb = sampler_next(sq); assert(bb);
            const int64_t *o = bb->corner[0];   /* the window meets the labelled chunk's level-0 extent [256, 512) x [0, 256) x [256, 512) (origins snap to the CT chunks) */
            if (getenv("WS_DEBUG")) fprintf(stderr, "corner %lld %lld %lld\n", (long long)o[0], (long long)o[1], (long long)o[2]);
            inbox &= o[0] + PW + 128 > 256 && o[0] < 512 && o[1] < 256 && o[2] + PW + 128 > 256 && o[2] < 512;   /* (+ 128: the snap down) */
            for (size_t k = 0; k < p3; k++) kn += bb->side[k] < PHASE_PERIOD;
            sampler_release(sq, bb);
        }
        uint64_t pr2, rj2; sampler_stats(sq, &pr2, &rj2); sampler_stop(sq); sources_free(S2);
        printf("wind sampler: sparse labels: %.3f of the voxels with a known phase, %llu rejected\n", kn / (double)(NB * p3), (unsigned long long)rj2);
        check("sparse labels: windows drawn around the labelled phase chunks", inbox && kn > NB * p3 / 4);
    }
    if (argc < 2) { char cmd[1500]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir); if (system(cmd)) failed++; }
    return failed != 0;
}
