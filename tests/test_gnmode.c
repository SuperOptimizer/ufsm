/* test_gnmode (fix0332 2026-11-18): predict's GroupNorm choice for gn_ema checkpoints (src/gnmode.c): an unwarmed checkpoint (ge_n <
   ge_warm) defaults to each window's own statistics with a DEFAULTED flag, a warmed one to the running ones, --gn-live wins, --gn-frozen
   and non-gn_ema nets never default to live */
#include "gnmode.h"
#include "checkpoint.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)
int main(void) {
    const char *unw = "{\"step\":8690,\"gn_ema\":1,\"ge_n\":8690,\"ge_warm\":1000000000,\"ge_mom\":0.01}", *warm = "{\"gn_ema\":1,\"ge_n\":2000,\"ge_warm\":500}";
    int d; long n, w;
    CHECK(gn_live_choice(unw, nullptr, 1, 0, &d, &n, &w) == 1 && d == 1 && n == 8690 && w == 1000000000, "unwarmed: live, defaulted (%d %ld %ld)", d, n, w);
    CHECK(gn_live_choice(unw, "0", 1, 0, &d, &n, &w) == 0 && d == 0, "unwarmed, --gn-live 0: running");
    CHECK(gn_live_choice(unw, "1", 1, 0, &d, &n, &w) == 1 && d == 0, "unwarmed, --gn-live 1: live, not defaulted");
    CHECK(gn_live_choice(warm, nullptr, 1, 0, &d, &n, &w) == 0 && d == 0, "warmed: running");
    CHECK(gn_live_choice(warm, "1", 1, 0, &d, &n, &w) == 1 && d == 0, "warmed, --gn-live 1: live");
    CHECK(gn_live_choice(unw, nullptr, 1, 1, &d, &n, &w) == 0 && d == 0, "unwarmed, --gn-frozen: not live");
    CHECK(gn_live_choice(unw, nullptr, 0, 0, &d, &n, &w) == 0 && d == 0, "no gn_ema: not live");
    CHECK(gn_live_choice(nullptr, nullptr, 1, 0, &d, &n, &w) == 0 && d == 0 && n == -1 && w == -1, "no header: running (old checkpoints)");
    CHECK(gn_live_choice("{\"gn_ema\":1}", nullptr, 1, 0, &d, &n, &w) == 0 && d == 0, "header without ge_n / ge_warm: running");
    /* train's validation (gn_live_choice_n): live before the warm-up step, running after it, --val-gn-live wins */
    CHECK(gn_live_choice_n(nullptr, 1, 0, 300, 500, &d) == 1 && d == 1, "val before warm: live, defaulted");
    CHECK(gn_live_choice_n(nullptr, 1, 0, 500, 500, &d) == 0 && d == 0, "val at warm: running");
    CHECK(gn_live_choice_n("0", 1, 0, 300, 500, &d) == 0 && d == 0, "val --val-gn-live 0 before warm: running");
    CHECK(gn_live_choice_n("1", 1, 0, 9000, 500, &d) == 1 && d == 0, "val --val-gn-live 1 after warm: live");
    CHECK(gn_live_choice_n(nullptr, 0, 0, 0, 500, &d) == 0 && d == 0, "val, no gn_ema: not live");
    /* predict --ema / train --val-live auto (ema_choice): the EMA unless |ema - p| / |p| > EMA_REL_MAX; an explicit value wins */
    CHECK(ema_choice(nullptr, 0.049, &d) == 1 && d == 0, "ema auto, u1-like lag 0.049: EMA");
    CHECK(ema_choice("auto", 0.117, &d) == 0 && d == 1, "ema auto, pab-A-like lag 0.117: live, defaulted");
    CHECK(ema_choice(nullptr, -1, &d) == 1 && d == 0, "ema auto, unreadable (-1): EMA");
    CHECK(ema_choice("1", 0.5, &d) == 1 && d == 0, "--ema 1 wins");
    CHECK(ema_choice("0", 0.0, &d) == 0 && d == 0, "--ema 0 wins");
    /* checkpoint_ema_rel on a synthetic file: header line, p, ema (and moments after them, never read) */
    {
        const char *fn = "test_gnmode.ckpt.tmp"; FILE *f = fopen(fn, "wb"); const int np = 3000000; float *x = malloc((size_t)np * 4);
        fprintf(f, "UFSM{\"nparams\":%d,\"step\":5}\n", np);
        for (int i = 0; i < np; i++) x[i] = (i % 7) - 3.f;   /* |p|^2 per 7: 28 */
        fwrite(x, 4, np, f); for (int i = 0; i < np; i++) x[i] *= 1.1f; fwrite(x, 4, np, f); fwrite(x, 4, np, f); fclose(f); free(x);
        const double r = checkpoint_ema_rel(fn); remove(fn);
        CHECK(fabs(r - 0.1) < 1e-5, "checkpoint_ema_rel: ema = 1.1 p reads 0.1 (%.6f)", r);
        CHECK(checkpoint_ema_rel("no-such-file.ckpt") == -1, "checkpoint_ema_rel: missing file -1");
    }
    printf("test_gnmode: %s\n", fails ? "FAILED" : "ok"); return fails != 0;
}
