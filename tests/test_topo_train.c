/* train --topo-lambda (the wrap-count loss, src/nn_loss.cu nn_topo_slab) in the trainer, on the synthetic spiral fixture
   (tests/make_wind_fixture.c), --det 1, 4 steps, the latent w24 layout (--lat 4 --lat-guide 3 --rv 2: streamed head slabs, fused
   gradient pass) and the plain whole-cell unwrap2 head:
   (1) --topo-lambda 0 logs exactly the losses of a run without the flag (the off path runs no topo code);
   (2) --topo-lambda 1 trains: the topo term is logged, finite, on every window, with rays, and it moves the logged loss;
   (3) layouts without the phase pair are refused (--rv 1: the side slot holds recto / verso).
   Usage: test_topo_train <ufsm binary> <make_wind_fixture binary> */
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
/* per step line the text from " loss " through the gradient norm ("gn X"), ';'-separated */
static char *loss_terms(const char *log) {
    size_t cap = 1 << 16, n = 0; char *o = malloc(cap); o[0] = 0;
    for (const char *l = log; l && *l; ) {
        const char *e = strchr(l, '\n'); const size_t len = e ? (size_t)(e - l) : strlen(l);
        if (len > 5 && !strncmp(l, "step ", 5)) {
            char ln[4096]; snprintf(ln, sizeof ln, "%.*s", (int)(len < 4000 ? len : 4000), l);
            char *a = strstr(ln, " loss "), *g = a ? strstr(a, " gn ") : nullptr;
            if (a && g) { char *ge = g + 4; while (*ge && *ge != ' ') ge++; n += (size_t)snprintf(o + n, cap - n, "%d:%.*s;", atoi(ln + 5), (int)(ge - a), a); }
        }
        l = e ? e + 1 : nullptr;
        if (n + 4200 > cap) { cap *= 2; o = realloc(o, cap); }
    }
    return o;
}
/* the step lines' " topo X (count errors E, dT T turn, P% of windows)" groups: count, all finite, every window, the means */
static int topo_terms(const char *log, double *loss, double *err, double *pct) {
    int n = 0; *loss = *err = 0; *pct = 100;
    for (const char *p = log; (p = strstr(p, " topo ")); p++) {
        double a, b, c, w;
        if (sscanf(p, " topo %lf (count errors %lf, dT %lf turn, %lf%% of windows)", &a, &b, &c, &w) != 4) continue;
        if (!isfinite(a) || !isfinite(b) || !isfinite(c)) return -1;
        *loss += a; *err += b; if (w < *pct) *pct = w; n++;
    }
    return n;
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: test_topo_train <ufsm binary> <make_wind_fixture binary>\n"); return 2; }
    const char *U = argv[1];
    char d[] = "/tmp/test_topo_train_XXXXXX"; if (!mkdtemp(d)) return 1;
    CHECK(!sh("%s %s/fx 256 2 1 > %s/fx.log 2>&1", argv[2], d, d), "fixture");
    const char *common = "--task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 2 --val-train-ok 1 --val-every 1000 --log-every 1 "
                         "--ckpt-every 1000 --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --det 1 --steps 4 --packed-weight 8";
    static const struct { const char *name, *flags; } lay[] = {
        {"lat", "--rv 2 --rv-iso 100 --lat 4 --lat-guide 3"},
        {"plain", ""},
    };
    for (int k = 0; k < 2; k++) {
        const char *nm = lay[k].name, *fl = lay[k].flags;
        char p[1024];
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources.json --out %s/%s-a %s %s > %s/%s-a.log 2>&1", U, d, d, nm, common, fl, d, nm), "%s: run without the flag (%s/%s-a.log)", nm, d, nm);
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources.json --out %s/%s-b %s %s --topo-lambda 0 > %s/%s-b.log 2>&1", U, d, d, nm, common, fl, d, nm), "%s: --topo-lambda 0 (%s/%s-b.log)", nm, d, nm);
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources.json --out %s/%s-c %s %s --topo-lambda 1 --topo-len 8 --topo-stride 2 > %s/%s-c.log 2>&1", U, d, d, nm, common, fl, d, nm), "%s: --topo-lambda 1 (%s/%s-c.log)", nm, d, nm);
        snprintf(p, sizeof p, "%s/%s-a.log", d, nm); char *la = slurp(p);
        snprintf(p, sizeof p, "%s/%s-b.log", d, nm); char *lb = slurp(p);
        snprintf(p, sizeof p, "%s/%s-c.log", d, nm); char *lc = slurp(p);
        if (!la || !lb || !lc) { CHECK(0, "%s: missing logs", nm); continue; }
        char *ta = loss_terms(la), *tb = loss_terms(lb), *tc = loss_terms(lc);
        printf("%s off   : %s\n%s lambda0: %s\n%s topo 1 : %s\n", nm, ta, nm, tb, nm, tc);
        CHECK(strlen(ta) > 20 && strchr(ta, '4'), "%s: no step lines", nm);
        CHECK(!strcmp(ta, tb), "%s: --topo-lambda 0 logs other losses than no flag", nm);
        CHECK(!strstr(la, " topo ") && !strstr(lb, " topo "), "%s: the off runs log a topo term", nm);
        double tl, te, tp; const int nt = topo_terms(lc, &tl, &te, &tp);
        printf("%s topo terms: %d steps, mean loss %.4f, count errors %.4f, min %.0f%% of windows\n", nm, nt, nt > 0 ? tl / nt : 0, nt > 0 ? te / nt : 0, tp);
        CHECK(nt == 4 && tp == 100 && tl > 0, "%s: topo terms %d (want 4), loss sum %.4f, windows %.0f%%", nm, nt, tl, tp);
        CHECK(strstr(lc, "--topo-lambda 1: wrap-count rays") != nullptr, "%s: no topo banner", nm);
        CHECK(strcmp(ta, tc) != 0, "%s: the topo run logs the off run's losses", nm);
        free(la); free(lb); free(lc); free(ta); free(tb); free(tc);
    }
    /* (3) refused: --rv 1 */
    const int rc = sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources.json --out %s/rv1 %s --rv 1 --rv-iso 100 --topo-lambda 1 > %s/rv1.log 2>&1", U, d, d, common, d);
    CHECK(rc != 0, "--rv 1 --topo-lambda 1 was accepted");
    if (fails) { fprintf(stderr, "test_topo_train: %d FAILED (logs in %s)\n", fails, d); return 1; }
    sh("rm -rf %s", d);
    printf("test_topo_train: OK\n");
    return 0;
}
