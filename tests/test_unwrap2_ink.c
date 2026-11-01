/* train --task unwrap2 --ink-head 1 (the v2 universal model: winding heads + the ink logit, channel 11) and predict writing fields
   and the ink volume from one pass. On the synthetic spiral fixture (tests/make_wind_fixture.c with its ink target, prob254):
   (1) a plain unwrap2 donor (11 outputs, 3 steps); (2) --init-from it with --ink-head 1 at learning rate 0: the trunk and head rows
   0-10 are the donor's, row 11 zero weights with bias --ink-bias; (3) a 16-wide from-scratch ink net trains a few steps (head slabs,
   ink bce logged and finite, the checkpoint names "ink":11) and resumes; (4) predict --fit-zarr --channel ink writes the fields and an
   ink volume byte-identical to predict --channel 11 alone; with --feather the fields agree with the plain --fit-zarr run.
   Usage: test_unwrap2_ink <ufsm binary> <make_wind_fixture binary> */
#include "nn.h"
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static int sh(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
#include <stdarg.h>
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
/* the live weights of a checkpoint and its head offsets */
static float *load(const char *ck, unet_cfg *cfg, size_t *np, size_t *hw, size_t *hb) {
    int step; if (unet_peek(ck, cfg, &step)) return nullptr;
    unet *u = unet_create(cfg); if (unet_load(u, ck) < 0) { unet_free(u); return nullptr; }
    *np = unet_nparams(u); unet_head_offsets(u, hw, hb);
    float *h = malloc(*np * 4); nn_d2h(h, unet_live_params(u), *np * 4); unet_free(u);
    return h;
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: test_unwrap2_ink <ufsm binary> <make_wind_fixture binary>\n"); return 2; }
    const char *U = argv[1];
    char d[] = "/tmp/test_u2ink_XXXXXX"; if (!mkdtemp(d)) return 1;
    if (nn_init(0)) { fprintf(stderr, "test_unwrap2_ink: no GPU\n"); return 1; }
    CHECK(!sh("%s %s/fx 256 2 1 > %s/fx.log 2>&1", argv[2], d, d), "fixture");
    const char *common = "--task unwrap2 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 2 --val-every 1000 --log-every 1 "
                         "--ckpt-every 1000 --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --levels 1";   /* (level 0 only: the fixture's ink is a level-0 array) */
    /* (1) the donor: a plain unwrap2 net */
    CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources.json --out %s/donor --widths 32,64,128 --steps 3 %s > %s/donor.log 2>&1", U, d, d, common, d), "donor (%s/donor.log)", d);
    /* (2) the warm start at learning rate 0 */
    CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-ink.json --out %s/warm --widths 32,64,128 --steps 1 --lr 0 --muon-lr 0 --ink-head 1 --ink-bias -1.5 "
              "--init-from %s/donor/last.ckpt %s > %s/warm.log 2>&1", U, d, d, d, common, d), "warm start (%s/warm.log)", d);
    {   char a[600], b[600]; snprintf(a, sizeof a, "%s/donor/last.ckpt", d); snprintf(b, sizeof b, "%s/warm/last.ckpt", d);
        unet_cfg ca, cb; size_t na = 0, nb = 0, hwa, hba, hwb, hbb;
        float *pa = load(a, &ca, &na, &hwa, &hba), *pb = load(b, &cb, &nb, &hwb, &hbb);
        CHECK(pa && pb, "load the donor / warm checkpoints");
        if (pa && pb) {
            CHECK(ca.cout == 11 && cb.cout == 12, "outputs %d / %d, want 11 / 12", ca.cout, cb.cout);
            const size_t cin = (hba - hwa) / (size_t)ca.cout;
            CHECK(hwa == hwb && cin == (hbb - hwb) / (size_t)cb.cout, "head layout (%zu %zu, %zu inputs)", hwa, hwb, cin);
            size_t bad = 0; for (size_t i = 0; i < hwa && i < hwb; i++) bad += pa[i] != pb[i];
            CHECK(!bad, "%zu trunk weights before the head differ from the donor's", bad);
            bad = 0; for (size_t i = 0; i < 11 * cin; i++) bad += pa[hwa + i] != pb[hwb + i];
            for (int c = 0; c < 11; c++) bad += pa[hba + c] != pb[hbb + c];
            CHECK(!bad, "%zu head weights of rows 0-10 differ from the donor's", bad);
            float mx = 0; for (size_t i = 0; i < cin; i++) mx = fmaxf(mx, fabsf(pb[hwb + 11 * cin + i]));
            CHECK(mx == 0.f && fabsf(pb[hbb + 11] + 1.5f) < 1e-6f, "ink row: max |w| %g (want 0), bias %g (want -1.5)", mx, pb[hbb + 11]);
            const size_t ta = hba + 11, tb = hbb + 12;   /* (whatever follows the head) */
            bad = 0; for (size_t i = 0; ta + i < na && tb + i < nb; i++) bad += pa[ta + i] != pb[tb + i];
            CHECK(!bad && na - ta == nb - tb, "%zu weights after the head differ (tails %zu / %zu)", bad, na - ta, nb - tb);
        }
        free(pa); free(pb);
        char p[600]; snprintf(p, sizeof p, "%s/warm.log", d); char *lg = slurp(p);
        CHECK(lg && strstr(lg, "row 11 (ink) fresh"), "warm start log names the fresh ink row");
        free(lg);
    }
    /* (3) a 16-wide from-scratch net with the ink head (head slabs with a 16-channel level 0), then a resume */
    CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-ink.json --out %s/b --widths 16,32,64 --steps 6 --ink-head 1 --ink-lambda 2 --trunk-lr-mult 0.5 --trunk-lr-steps 3 %s > %s/b.log 2>&1",
              U, d, d, common, d), "from-scratch ink net (%s/b.log)", d);
    {   char p[600]; snprintf(p, sizeof p, "%s/b.log", d); char *lg = slurp(p);
        CHECK(lg && strstr(lg, "head slabs:"), "the ink net runs on head slabs");
        CHECK(lg && strstr(lg, "trunk learning rate x1"), "--trunk-lr-steps switched the trunk back");
        const char *k = lg ? strstr(lg, " ink bce ") : nullptr; double v = k ? atof(k + 9) : -1;
        CHECK(k && isfinite(v) && v > 0, "a finite ink bce in the training log (%g)", v);
        const char *q = lg ? strstr(lg, "% of windows") : nullptr;
        CHECK(q && atof(q - 3) > 0, "windows with known ink");
        free(lg);
        snprintf(p, sizeof p, "%s/b/last.ckpt", d); FILE *f = fopen(p, "rb"); char hdr[8192] = {0};
        if (f) { if (!fgets(hdr, sizeof hdr, f)) hdr[0] = 0; fclose(f); }
        CHECK(strstr(hdr, "\"unwrap\":{\"version\":2") && strstr(hdr, "\"ink\":11"), "checkpoint header: unwrap version 2 with \"ink\":11");
    }
    CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-ink.json --out %s/b --widths 16,32,64 --steps 8 --ink-head 1 --ink-lambda 2 --resume %s/b/last.ckpt %s > %s/b2.log 2>&1",
              U, d, d, d, common, d), "resume (%s/b2.log)", d);
    {   char p[600]; snprintf(p, sizeof p, "%s/b2.log", d); char *lg = slurp(p);
        CHECK(lg && strstr(lg, "resumed") && strstr(lg, " ink bce "), "resume trains the ink head on");
        free(lg);
    }
    /* (4) predict: fields + ink from one pass vs the ink channel alone; --feather with --fit-zarr */
    const char *pr = "--um 9.6 --level 0 --window 160 --halo 16 --shard 128 --levels 2 --threads 2 --fit-h 4";
    CHECK(!sh("%s predict %s/b/last.ckpt %s/fx ct %s/pi --channel ink --fit-zarr %s/fi %s > %s/p1.log 2>&1", U, d, d, d, d, pr, d), "predict fields + ink (%s/p1.log)", d);
    CHECK(!sh("%s predict %s/b/last.ckpt %s/fx ct %s/po --channel 11 %s > %s/p2.log 2>&1", U, d, d, d, pr, d), "predict channel 11 (%s/p2.log)", d);
    CHECK(!sh("%s predict %s/b/last.ckpt %s/fx ct %s/pz --fit-zarr %s/fz %s > %s/p3.log 2>&1", U, d, d, d, d, pr, d), "predict fields only (%s/p3.log)", d);
    CHECK(!sh("%s predict %s/b/last.ckpt %s/fx ct %s/pf --channel ink --fit-zarr %s/ff %s --feather 8 > %s/p4.log 2>&1", U, d, d, d, d, pr, d), "predict --feather + fields (%s/p4.log)", d);
    CHECK(!sh("test -d %s/fi/pc && test -d %s/fi/dq && test -d %s/pi", d, d, d), "fields and the ink volume exist");
    CHECK(!sh("diff -rq -x zarr.json %s/pi %s/po > /dev/null", d, d), "the ink volume of the fields pass differs from --channel 11 alone");
    CHECK(!sh("diff -rq -x zarr.json %s/fi %s/fz > /dev/null", d, d), "--channel ink changed the fields");
    CHECK(sh("diff -rq -x zarr.json %s/pi %s/pz > /dev/null 2>&1", d, d) != 0, "the ink volume equals channel 0's (no channel offset)");
    CHECK(!sh("diff -rq -x zarr.json %s/ff %s/fz > /dev/null", d, d), "--feather changed the fields");
    {   char p[600]; snprintf(p, sizeof p, "%s/pi/zarr.json", d); char *z = slurp(p);
        CHECK(z && strstr(z, "ink probability"), "the ink volume's content attribute");
        free(z);
    }
    if (!fails) sh("rm -rf %s", d);
    if (fails) { fprintf(stderr, "test_unwrap2_ink: %d FAILED (kept %s)\n", fails, d); return 1; }
    printf("test_unwrap2_ink: ok\n");
    return 0;
}
