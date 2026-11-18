/* test_gnmode (fix0332 2026-11-18): predict's GroupNorm choice for gn_ema checkpoints (src/gnmode.c): an unwarmed checkpoint (ge_n <
   ge_warm) defaults to each window's own statistics with a DEFAULTED flag, a warmed one to the running ones, --gn-live wins, --gn-frozen
   and non-gn_ema nets never default to live */
#include "gnmode.h"
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
    printf("test_gnmode: %s\n", fails ? "FAILED" : "ok"); return fails != 0;
}
