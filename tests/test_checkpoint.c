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
    const char *e = nn_check(); check("CUDA checks", !e); if (e) printf("%s\n", e);
    unlink(base); unlink(a); unlink(b); unlink(old); unlink(cut);
    return failures != 0;
}
