/* Label-noise tools: nn_phase_trunc (the robust phase loss of train --phase-trunc: confident predictions far off their label leave the
   mask, unsure or agreeing ones stay) and ufsm phase-conflict (two label phase rasters -> 1 where both are known and differ by > tol). */
#include "nn.h"
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_phase_conflict(int argc, char **argv);
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static uint16_t f2h(float f) {   /* IEEE half from float (normal range, round to nearest) */
    uint32_t x; memcpy(&x, &f, 4); const uint32_t s = (x >> 16) & 0x8000; int e = (int)((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffff;
    if (e <= 0) return (uint16_t)s; if (e >= 31) return (uint16_t)(s | 0x7c00);
    m += 0x1000; if (m & 0x800000) { m = 0; e++; } return (uint16_t)(s | (uint32_t)e << 10 | m >> 13);
}
static float logit(float p) { return logf(p / (1 - p)); }

int main(void) {
    nn_init(0);
    {   /* five voxels: confident and right / confident and 0.4 turn off / unsure and off / unlabelled / confident, 0.2 turn off */
        enum { N = 5 };
        const float pa[N] = {0.f, 0.4f, 0.4f, 0.f, 0.2f}, ta = 0.f, conf[N] = {0.9f, 0.9f, 0.2f, 0.9f, 0.9f};
        uint16_t lg[2 * N]; uint8_t t[2 * N], m[N];
        for (int i = 0; i < N; i++) {
            const float c = conf[i] * cosf(2 * (float)M_PI * pa[i]), s = conf[i] * sinf(2 * (float)M_PI * pa[i]);
            lg[i] = f2h(logit(0.5f + 0.5f * c)); lg[N + i] = f2h(logit(0.5f + 0.5f * s));
            t[i] = (uint8_t)lroundf(127.5f + 127.5f * cosf(2 * (float)M_PI * ta)); t[N + i] = (uint8_t)lroundf(127.5f + 127.5f * sinf(2 * (float)M_PI * ta));
            m[i] = i == 3 ? 0 : 1;
        }
        void *dl = nn_malloc(sizeof lg); uint8_t *dt = nn_malloc(sizeof t), *dm = nn_malloc(sizeof m);
        nn_h2d(dl, lg, sizeof lg); nn_h2d(dt, t, sizeof t); nn_h2d(dm, m, sizeof m);
        unsigned long long c0[2]; nn_phase_trunc_counts(c0);
        nn_phase_trunc(dl, (const char *)dl + N * 2, dt, dm, N, 0.25f, 0.5f);
        uint8_t mo[N]; nn_d2h(mo, dm, sizeof mo); unsigned long long cc[2]; nn_phase_trunc_counts(cc);
        printf("phase trunc: mask %d %d %d %d %d, counts %llu labelled %llu dropped\n", mo[0], mo[1], mo[2], mo[3], mo[4], cc[0], cc[1]);
        CHECK(mo[0] == 1 && mo[1] == 0 && mo[2] == 1 && mo[3] == 0 && mo[4] == 1, "phase trunc mask %d %d %d %d %d (want 1 0 1 0 1)", mo[0], mo[1], mo[2], mo[3], mo[4]);
        CHECK(cc[0] == 4 && cc[1] == 1, "phase trunc counts %llu / %llu (want 4 / 1)", cc[0], cc[1]);
        nn_free(dl); nn_free(dt); nn_free(dm);
    }
    {   /* phase-conflict on two 128^3 rasters: agree, disagree by 0.4 turn, one unknown */
        char dir[] = "/tmp/ufsm-robust-XXXXXX"; if (!mkdtemp(dir)) { fprintf(stderr, "FAIL: tmp dir\n"); return 1; }
        const int64_t sh[3] = {128, 128, 128}; const size_t n = 128 * 128 * 128; uint8_t *a = malloc(n), *b = malloc(n);
        for (size_t i = 0; i < n; i++) { a[i] = (uint8_t)(i % 200); b[i] = a[i]; }
        for (size_t i = 0; i < 1000; i++) b[i] = (uint8_t)((a[i] + 100) % 252);   /* 100 / 252 = 0.40 turn */
        for (size_t i = 1000; i < 1100; i++) b[i] = 255;                            /* unknown in b */
        for (size_t i = 1100; i < 1200; i++) b[i] = (uint8_t)((a[i] + 30) % 252);  /* 0.12 turn: within tol */
        char pa_[300], pb_[300], po[300], cmd[1000];
        snprintf(pa_, sizeof pa_, "%s/a/9.6", dir); snprintf(pb_, sizeof pb_, "%s/b/9.6", dir); snprintf(po, sizeof po, "%s/c", dir);
        z3w *wa = z3w_create(pa_, sh, 128, 0.f, 255, "{}"), *wb = z3w_create(pb_, sh, 128, 0.f, 255, "{}");
        CHECK(wa && wb && !z3w_write_shard(wa, 0, 0, 0, a, 1) && !z3w_write_shard(wb, 0, 0, 0, b, 1), "write rasters");
        z3w_close(wa); z3w_close(wb);
        char ra[300], rb[300]; snprintf(ra, sizeof ra, "%s/a", dir); snprintf(rb, sizeof rb, "%s/b", dir);
        char *av[] = {"ufsm", "phase-conflict", ra, "9.6", rb, "9.6", po, "--tol", "0.25"};
        CHECK(cmd_phase_conflict(9, av) == 0, "phase-conflict run");
        z3 *zc = z3_open(store_open(dir), "c", nullptr); uint8_t *c = calloc(n, 1); const int64_t o[3] = {0, 0, 0};
        CHECK(zc && !z3_read(zc, o, sh, c, 1), "read the conflict mask");
        size_t ones = 0, wrong = 0; for (size_t i = 0; i < n; i++) { ones += c[i] != 0; wrong += (c[i] != 0) != (i < 1000); }
        printf("phase-conflict: %zu flagged, %zu misflagged\n", ones, wrong);
        CHECK(ones == 1000 && wrong == 0, "phase-conflict flags %zu (want 1000), misflagged %zu", ones, wrong);
        z3_close(zc); free(a); free(b); free(c);
        snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir); if (system(cmd)) fprintf(stderr, "(could not remove %s)\n", dir);
    }
    if (fails) { fprintf(stderr, "test_robust: %d failure(s)\n", fails); return 1; }
    printf("test_robust: OK\n");
    return 0;
}
