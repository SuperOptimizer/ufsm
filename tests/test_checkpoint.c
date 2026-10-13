/* A resumed optimizer's next update must match an uninterrupted update. */
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;
static void check(const char *name, int ok) { printf("  %-48s %s\n", name, ok ? "ok" : "FAIL"); failures += !ok; }
static void gradient(unet *u, unsigned seed) {
    size_t n = unet_nparams(u);
    float *h = malloc(n * 4);
    for (size_t i = 0; i < n; i++) { seed = seed * 1664525u + 1013904223u; h[i] = ((seed >> 8) / 16777216.f - 0.5f) * 0.1f; }
    unet_grad_h2d(u, h); free(h);
}
static void update(unet *u, int muon, int step) {
    if (muon) unet_muon(u, 0.01f, 0.95f, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, step);
    else unet_adamw(u, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, step);
    unet_ema(u, 0.9f);
}
static int same_file(const char *a, const char *b) {
    FILE *fa = fopen(a, "rb"), *fb = fopen(b, "rb");
    if (!fa || !fb) { if (fa) fclose(fa); if (fb) fclose(fb); return 0; }
    int x, y, ok = 1;
    do { x = fgetc(fa); y = fgetc(fb); if (x != y) { ok = 0; break; } } while (x != EOF);
    fclose(fa); fclose(fb); return ok;
}
int main(void) {
    if (nn_init(0)) return 1;
    unet_cfg cfg = {2, {4, 8}, 4, 2, 2, 0};
    char base[96], a[96], b[96], old[96], cut[96];
    snprintf(base, sizeof base, "/tmp/ufsm_checkpoint_%d_base", (int)getpid());
    snprintf(a, sizeof a, "/tmp/ufsm_checkpoint_%d_a", (int)getpid());
    snprintf(b, sizeof b, "/tmp/ufsm_checkpoint_%d_b", (int)getpid());
    snprintf(old, sizeof old, "/tmp/ufsm_checkpoint_%d_old", (int)getpid());
    snprintf(cut, sizeof cut, "/tmp/ufsm_checkpoint_%d_cut", (int)getpid());
    for (int muon = 0; muon <= 1; muon++) {
        unet *live = unet_create(&cfg), *resumed = unet_create(&cfg);
        unet_init(live, 37); unet_init(resumed, 91);
        gradient(live, 123); update(live, muon, 1);
        check("save trained state", unet_save(live, base, 1, nullptr) == 0);
        check("load saved step", unet_load(resumed, base) == 1);
        check("save loaded state", unet_save(resumed, b, 1, nullptr) == 0);
        check(muon ? "Muon state round trip" : "AdamW state round trip", same_file(base, b));
        gradient(live, 456); gradient(resumed, 456);
        update(live, muon, 2); update(resumed, muon, 2);
        check("save uninterrupted next step", unet_save(live, a, 2, nullptr) == 0);
        check("save resumed next step", unet_save(resumed, b, 2, nullptr) == 0);
        check(muon ? "Muon resumed next update is identical" : "AdamW resumed next update is identical", same_file(a, b));
        if (muon) {
            /* Preserve the first four arrays but omit the optional header field and momentum payload. */
            FILE *f = fopen(base, "rb"), *legacy = fopen(old, "wb"), *truncated = fopen(cut, "wb");
            char line[4096];
            if (!f || !legacy || !truncated || !fgets(line, sizeof line, f)) return 1;
            fputs(line, truncated);
            char *field = strstr(line, "\"muon_mom\":1,");
            check("Muon flag saved", field != nullptr);
            if (!field) return 1;
            memmove(field, field + strlen("\"muon_mom\":1,"), strlen(field + strlen("\"muon_mom\":1,")) + 1);
            fputs(line, legacy);
            size_t n = 4 * unet_nparams(live), bytes = n * sizeof(float);
            float *payload = malloc(bytes);
            check("read legacy payload", fread(payload, sizeof(float), n, f) == n);
            fwrite(payload, 1, bytes, legacy); fwrite(payload, 1, bytes, truncated);
            fclose(f); fclose(legacy); fclose(truncated); free(payload);
            check("missing declared Muon momentum rejected", unet_load(resumed, cut) < 0);
            check("old checkpoint remains readable", unet_load(resumed, old) == 1);
            /* Loading old state into a used object must also clear its previous momentum. */
            unet *fresh = unet_create(&cfg);
            check("old checkpoint loads in fresh object", unet_load(fresh, old) == 1);
            gradient(resumed, 789); gradient(fresh, 789);
            update(resumed, 1, 2); update(fresh, 1, 2);
            unet_save(resumed, a, 2, nullptr); unet_save(fresh, b, 2, nullptr);
            check("legacy checkpoint clears stale Muon momentum", same_file(a, b));
            unet_free(fresh);
        }
        unet_free(live); unet_free(resumed);
    }
    {   /* warm start into more inputs and outputs (train --cond: 4 -> 7 inputs, --head-map): the grown net computes the donor's
           outputs on the mapped rows whatever the new inputs hold */
        unet_cfg dc = cfg; dc.cout = 3; unet_cfg gc = dc; gc.cin = 7; gc.cout = 5;
        unet *donor = unet_create(&dc), *grown = unet_create(&gc);
        unet_init(donor, 5); unet_init(grown, 6);
        check("save donor", unet_save(donor, base, 7, nullptr) == 0);
        const int rows[5] = {0, 1, 2, -1, -1};
        check("warm start 4 -> 7 inputs, 3 -> 5 outputs", unet_init_from_map(grown, base, rows) > 0);
        enum { P = 16 }; const size_t p3 = (size_t)P * P * P;
        float *h = malloc(7 * p3 * 4); unsigned s = 77;
        for (size_t k = 0; k < 7 * p3; k++) { s = s * 1664525u + 1013904223u; h[k] = ((s >> 8) / 16777216.f - 0.5f) * 2.f; }
        float *dx = nn_malloc(7 * p3 * 4), *od = malloc(3 * p3 * 4), *og = malloc(5 * p3 * 4);
        nn_h2d(dx, h, 4 * p3 * 4);
        const float *ld = unet_forward_x(donor, dx, (shape5){1, 4, P, P, P}, 0, 0); nn_sync(); nn_d2h(od, ld, 3 * p3 * 4);
        double emax = 0;
        for (int pass = 0; pass < 2; pass++) {   /* new inputs zero, then random */
            float *g7 = malloc(7 * p3 * 4); memcpy(g7, h, 4 * p3 * 4);
            if (pass) memcpy(g7 + 4 * p3, h + 4 * p3, 3 * p3 * 4); else memset(g7 + 4 * p3, 0, 3 * p3 * 4);
            nn_h2d(dx, g7, 7 * p3 * 4); free(g7);
            const float *lg = unet_forward_x(grown, dx, (shape5){1, 7, P, P, P}, 0, 0); nn_sync(); nn_d2h(og, lg, 5 * p3 * 4);
            for (size_t k = 0; k < 3 * p3; k++) { const double e = fabs((double)og[k] - od[k]); if (e > emax) emax = e; }
        }
        printf("  grown net vs donor: max |logit difference| %.3g\n", emax);
        check("grown net computes the donor's mapped outputs", emax < 1e-3);
        free(h); free(od); free(og); nn_free(dx); unet_free(donor); unet_free(grown);
    }
    const char *e = nn_check(); check("CUDA checks", !e); if (e) printf("%s\n", e);
    unlink(base); unlink(a); unlink(b); unlink(old); unlink(cut);
    return failures != 0;
}
