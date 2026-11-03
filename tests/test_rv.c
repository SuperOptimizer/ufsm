/* train --task unwrap2 --rv 1 (the recto / verso side model): (1) nn_rv_targets on hand-made phase / papyrus codes; (2) a 16-wide
   from-scratch net on the synthetic spiral fixture (tests/make_wind_fixture.c: sheets CT ~150, gaps ~45) trains a few steps on head
   slabs with the papyrus plane (--rv-iso 100), logs a finite "rv bce", names "rv" in the checkpoint header, resumes; (3) predict writes
   channels 0 (recto side) and 1 (verso side); (4) --rv with --ink-head is refused.
   Usage: test_rv <ufsm binary> <make_wind_fixture binary> */
#include "nn.h"
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static int sh(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static int sh(const char *fmt, ...) {
    char cmd[8192]; va_list a; va_start(a, fmt); vsnprintf(cmd, sizeof cmd, fmt, a); va_end(a);
    const int r = system(cmd);
    if (r) fprintf(stderr, "command failed (%d): %s\n", r, cmd);
    return r;
}
static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *s = malloc((size_t)n + 1); if (!s) { fclose(f); return nullptr; }
    s[fread(s, 1, (size_t)n, f)] = 0; fclose(f); return s;
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: test_rv <ufsm binary> <make_wind_fixture binary>\n"); return 2; }
    const char *U = argv[1];
    if (nn_init(0)) { fprintf(stderr, "test_rv: no GPU\n"); return 1; }
    {   /* (1) the targets: recto side = phase < 126, verso 126..251, unknown phase 252..255; void (papyrus < 127) 0 / 0 known; papyrus 255 unknown */
        enum { N = 10 };
        const uint8_t ph[N]  = {0, 100, 125, 126, 200, 251, 252, 255, 30, 200};
        const uint8_t pap[N] = {254, 254, 127, 254, 200, 254, 254, 254, 0, 255};
        const uint8_t wt0[N] = {255, 255, 255, 0, 0, 0, 0, 0, 0, 0}, wt1[N] = {0, 0, 0, 255, 255, 255, 0, 0, 0, 0}, wm[N] = {1, 1, 1, 1, 1, 1, 0, 0, 1, 0};
        uint8_t *dp = nn_malloc(N), *dq = nn_malloc(N), *dt = nn_malloc(2 * N), *dm = nn_malloc(N), t[2 * N], m[N];
        nn_h2d(dp, ph, N); nn_h2d(dq, pap, N); nn_rv_targets(dp, dq, N, dt, dm); nn_sync(); nn_d2h(t, dt, 2 * N); nn_d2h(m, dm, N);
        for (int i = 0; i < N; i++) CHECK(m[i] == wm[i] && (!wm[i] || (t[i] == wt0[i] && t[N + i] == wt1[i])),
                                          "voxel %d (phase %d, papyrus %d): targets %d / %d mask %d, want %d / %d mask %d", i, ph[i], pap[i], t[i], t[N + i], m[i], wt0[i], wt1[i], wm[i]);
        nn_free(dp); nn_free(dq); nn_free(dt); nn_free(dm);
    }
    char d[] = "/tmp/test_rv_XXXXXX"; if (!mkdtemp(d)) return 1;
    CHECK(!sh("%s %s/fx 256 2 > %s/fx.log 2>&1", argv[2], d, d), "fixture");
    const char *common = "--task unwrap2 --rv 1 --rv-iso 100 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 2 --val-every 1000 --log-every 1 "
                         "--ckpt-every 1000 --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64";
    /* (2) from scratch, then a resume */
    CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources.json --out %s/b --steps 6 %s > %s/b.log 2>&1", U, d, d, common, d), "rv net (%s/b.log)", d);
    {   char p[600]; snprintf(p, sizeof p, "%s/b.log", d); char *lg = slurp(p);
        CHECK(lg && strstr(lg, "head slabs:"), "the rv net runs on head slabs");
        const char *k = lg ? strstr(lg, " rv bce ") : nullptr; const double v = k ? atof(k + 8) : -1;
        CHECK(k && isfinite(v) && v > 0, "a finite rv bce in the training log (%g)", v);
        free(lg);
        snprintf(p, sizeof p, "%s/b/last.ckpt", d); FILE *f = fopen(p, "rb"); char hdr[16384] = {0};
        if (f) { if (!fgets(hdr, sizeof hdr, f)) hdr[0] = 0; fclose(f); }
        CHECK(strstr(hdr, "\"unwrap\":{\"version\":2") && strstr(hdr, "\"rv\":{\"version\":1,\"recto\":0,\"verso\":1,\"iso\":100"), "checkpoint header: unwrap 2 with the rv block");
    }
    CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources.json --out %s/b --steps 8 --resume %s/b/last.ckpt %s > %s/b2.log 2>&1", U, d, d, d, common, d), "resume (%s/b2.log)", d);
    {   char p[600]; snprintf(p, sizeof p, "%s/b2.log", d); char *lg = slurp(p);
        CHECK(lg && strstr(lg, "resumed") && strstr(lg, " rv bce "), "resume trains the rv head on");
        free(lg);
    }
    /* (3) predict both sides */
    const char *pr = "--um 9.6 --level 0 --window 160 --halo 16 --shard 128 --levels 1 --threads 2";
    CHECK(!sh("%s predict %s/b/last.ckpt %s/fx ct %s/pr --channel 0 %s > %s/p0.log 2>&1", U, d, d, d, pr, d), "predict recto side (%s/p0.log)", d);
    CHECK(!sh("%s predict %s/b/last.ckpt %s/fx ct %s/pv --channel 1 %s > %s/p1.log 2>&1", U, d, d, d, pr, d), "predict verso side (%s/p1.log)", d);
    CHECK(sh("diff -rq -x zarr.json %s/pr %s/pv > /dev/null 2>&1", d, d) != 0, "recto and verso volumes are identical");
    CHECK(!sh("%s predict %s/b/last.ckpt %s/fx ct %s/pf --channel rv %s > %s/pf.log 2>&1", U, d, d, d, pr, d), "predict --channel rv (%s/pf.log)", d);
    CHECK(sh("diff -rq -x zarr.json %s/pr %s/pf > /dev/null 2>&1", d, d) != 0 && sh("diff -rq -x zarr.json %s/pv %s/pf > /dev/null 2>&1", d, d) != 0, "the fused volume equals one side");
    {   char p[600]; snprintf(p, sizeof p, "%s/pf/zarr.json", d); char *z = slurp(p);
        CHECK(z && strstr(z, "recto / verso side"), "the fused volume's content attribute"); free(z); }
    CHECK(sh("%s predict %s/b/last.ckpt %s/fx ct %s/pz --fit-zarr %s/fz %s > %s/pz.log 2>&1", U, d, d, d, d, pr, d) != 0, "--fit-zarr accepted an rv checkpoint");
    /* (5) side-face on the fixture (its wraps sit on the sheets' inner faces): faces found, boundaries at void, output readable */
    CHECK(!sh("%s side-face %s/fx/phase 1 %s/fx/ct 19.2 %s/sf 19.2 --iso 100 --per-voxel 1 --threads 2 --block 64 --shard 128 > %s/sf.log 2> %s/sf.err", U, d, d, d, d, d), "side-face (%s/sf.err)", d);
    {   char p[600]; snprintf(p, sizeof p, "%s/sf.log", d); char *lg = slurp(p);
        const char *f = lg ? strstr(lg, "face found ") : nullptr; const double ff = f ? atof(f + 11) : -1;
        const char *b = lg ? strstr(lg, "face-anchored verso->recto boundaries") : nullptr; const char *v = b ? strstr(b, "void ") : nullptr; const double bv = v ? atof(v + 5) : -1;
        CHECK(ff > 0.95 && bv > 0.95, "side-face: face found %g, anchored boundaries at void %g (want > 0.95)", ff, bv);
        free(lg);
        CHECK(!sh("test -e %s/sf/19.2/zarr.json", d), "side-face output array");
    }
    /* (4) refused combination */
    CHECK(sh("%s train %s/fx/sources.json --out %s/c --steps 1 --ink-head 1 %s > %s/c.log 2>&1", U, d, d, common, d) != 0, "--rv with --ink-head accepted");
    if (!fails) sh("rm -rf %s", d);
    if (fails) { fprintf(stderr, "test_rv: %d FAILED (kept %s)\n", fails, d); return 1; }
    printf("test_rv: ok\n");
    return 0;
}
