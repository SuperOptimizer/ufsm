/* train --density-lambda (unwrap2): the density channel's BCE weight split from --normal-lambda. On the synthetic spiral fixture
   (tests/make_wind_fixture.c), for a 32-wide net (full-volume head gradient) and a 16-wide one (head slabs):
   (1) a plain unwrap2 donor (3 steps); (2) --init-from it with --density-lambda 0, --count-lambda 0, --wd 0: the density head row
   (channel 8, weights + bias) stays exactly the donor's while a normal row (channel 2) trains; (3) the same at --density-lambda 1
   moves the density row (the check is live); the checkpoint header and the log carry the density lambda.
   Usage: test_density_lambda <ufsm binary> <make_wind_fixture binary> */
#include "nn.h"
#include "unet.h"
#include <math.h>
#include <stdarg.h>
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
static float *load(const char *ck, unet_cfg *cfg, size_t *np, size_t *hw, size_t *hb) {
    int step; if (unet_peek(ck, cfg, &step)) return nullptr;
    unet *u = unet_create(cfg); if (unet_load(u, ck) < 0) { unet_free(u); return nullptr; }
    *np = unet_nparams(u); unet_head_offsets(u, hw, hb);
    float *h = malloc(*np * 4); nn_d2h(h, unet_live_params(u), *np * 4); unet_free(u);
    return h;
}
/* changed head entries of row c (weights + bias) between two checkpoints of the same layout */
static size_t row_diff(const float *a, const float *b, size_t hw, size_t hb, size_t cin, int c) {
    size_t n = 0; for (size_t i = 0; i < cin; i++) n += a[hw + c * cin + i] != b[hw + c * cin + i];
    return n + (a[hb + c] != b[hb + c]);
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: test_density_lambda <ufsm binary> <make_wind_fixture binary>\n"); return 2; }
    const char *U = argv[1];
    char d[] = "/tmp/test_dlam_XXXXXX"; if (!mkdtemp(d)) return 1;
    if (nn_init(0)) { fprintf(stderr, "test_density_lambda: no GPU\n"); return 1; }
    CHECK(!sh("%s %s/fx 256 2 1 > %s/fx.log 2>&1", argv[2], d, d), "fixture");
    const char *common = "--task unwrap2 --P 64 --B 1 --gpus 0 --workers 1 --sampler-slots 4 --val-batches 1 --val-every 1000 --log-every 1 "
                         "--ckpt-every 1000 --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --levels 1 --density-code 1 --wd 0";
    const char *W[2] = {"32,64,128", "16,32,64"};
    for (int k = 0; k < 2; k++) {
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources.json --out %s/donor%d --widths %s --steps 3 %s > %s/donor%d.log 2>&1", U, d, d, k, W[k], common, d, k),
              "donor (%s/donor%d.log)", d, k);
        for (int z = 0; z < 2; z++)   /* z 0: --density-lambda 0; z 1: 1 (the control) */
            CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources.json --out %s/r%d%d --widths %s --steps 3 --density-lambda %d --init-from %s/donor%d/last.ckpt %s > %s/r%d%d.log 2>&1",
                      U, d, d, k, z, W[k], z, d, k, common, d, k, z), "warm run (%s/r%d%d.log)", d, k, z);
        char a[600], b[600], c[600]; snprintf(a, sizeof a, "%s/donor%d/last.ckpt", d, k); snprintf(b, sizeof b, "%s/r%d0/last.ckpt", d, k);
        snprintf(c, sizeof c, "%s/r%d1/last.ckpt", d, k);
        unet_cfg ca, cb, cc; size_t na = 0, nb = 0, nc = 0, hwa, hba, hwb, hbb, hwc, hbc;
        float *pa = load(a, &ca, &na, &hwa, &hba), *pb = load(b, &cb, &nb, &hwb, &hbb), *pc = load(c, &cc, &nc, &hwc, &hbc);
        CHECK(pa && pb && pc, "load checkpoints (%s)", W[k]);
        if (pa && pb && pc) {
            CHECK(na == nb && na == nc && hwa == hwb && hwa == hwc && hba == hbb, "layouts differ (%s)", W[k]);
            const size_t cin = (hba - hwa) / (size_t)ca.cout;
            const size_t d0 = row_diff(pa, pb, hwa, hba, cin, 8), n0 = row_diff(pa, pb, hwa, hba, cin, 2), d1 = row_diff(pa, pc, hwa, hba, cin, 8);
            CHECK(d0 == 0, "%s: --density-lambda 0 changed %zu entries of the density head row", W[k], d0);
            CHECK(n0 > cin / 2, "%s: a normal head row barely trained (%zu of %zu changed)", W[k], n0, cin + 1);
            CHECK(d1 > cin / 2, "%s: --density-lambda 1 left the density row (%zu of %zu changed): the check is dead", W[k], d1, cin + 1);
        }
        free(pa); free(pb); free(pc);
        FILE *f = fopen(b, "rb"); char hdr[8192] = {0};
        if (f) { if (!fgets(hdr, sizeof hdr, f)) hdr[0] = 0; fclose(f); }
        CHECK(strstr(hdr, "\"normal_lambda\":1,\"density_lambda\":0,"), "%s: checkpoint header density_lambda 0", W[k]);
        char p[600]; snprintf(p, sizeof p, "%s/r%d0.log", d, k); char *lg = slurp(p);
        CHECK(lg && strstr(lg, "8 density (lambda 0, step"), "%s: the log names the density lambda", W[k]);
        if (k == 1) CHECK(lg && strstr(lg, "head slabs:"), "the 16-wide net runs on head slabs");
        free(lg);
    }
    if (!fails) sh("rm -rf %s", d);
    if (fails) { fprintf(stderr, "test_density_lambda: %d FAILED (kept %s)\n", fails, d); return 1; }
    printf("test_density_lambda: ok\n");
    return 0;
}
