/* train --task unwrap2 head registry (src/train.c aux planes): (1) every single-head layout that existed before the registry (plain
   unwrap2, --ink-head, --rv 1, --fib-normal-lambda, and the latent --lat 4 --lat-guide 3 runs of the plain / rv / ink heads) trains 4
   steps with --det 1 on the synthetic spiral fixture (tests/make_wind_fixture.c, ink target) and logs exactly the losses of the reference
   (tests/heads_ref.txt: the step lines' loss terms and gradient norms, 4 decimals, produced by the binary before the registry; the
   weights themselves are not bit-reproducible run to run, the logged losses are); (2) the combined heads --rv 2 (recto / verso on own
   channels next to the phase) + --ink-head 1 + --fib-normal-lambda (the fibre plane from the sources' "fibre" target), whole-window
   and latent, train with finite rv / ink / fibn losses and name every head in the checkpoint header.
   References are kept per GPU name (the losses differ between architectures); none for this GPU: (1) only checks that the runs work.
   UFSM_HEADS_REF_WRITE=1 appends this GPU's lines to the reference file instead (run with the pre-change binary).
   Usage: test_heads <ufsm binary> <make_wind_fixture binary> <tests/heads_ref.txt> */
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
/* the loss terms of a training log: per step line the text from " loss " through the gradient norm, then the validation tail */
static char *loss_terms(const char *log) {
    size_t cap = 1 << 16, n = 0; char *o = malloc(cap); o[0] = 0;
    for (const char *l = log; l && *l; ) {
        const char *e = strchr(l, '\n'); const size_t len = e ? (size_t)(e - l) : strlen(l);
        if (len > 5 && !strncmp(l, "step ", 5)) {
            char ln[4096]; snprintf(ln, sizeof ln, "%.*s", (int)(len < 4000 ? len : 4000), l);
            char *a = strstr(ln, " loss "), *g = a ? strstr(a, " gn ") : nullptr;
            if (a && g) {
                char *ge = g + 4; while (*ge && *ge != ' ') ge++;
                const char *v = strstr(ge, " val ");
                n += (size_t)snprintf(o + n, cap - n, "%d:%.*s%s%s;", atoi(ln + 5), (int)(ge - a), a, v ? v : "", "");
            }
        }
        l = e ? e + 1 : nullptr;
        if (n + 4200 > cap) { cap *= 2; o = realloc(o, cap); }
    }
    for (char *c = o; *c; c++) if (*c == '|' || *c == '\n') *c = ' ';
    return o;
}
static void gpu_name(char *out, size_t n) {
    const char *cv = getenv("CUDA_VISIBLE_DEVICES"); char cmd[256];
    snprintf(cmd, sizeof cmd, "nvidia-smi --query-gpu=name --format=csv,noheader -i %s 2>/dev/null", cv && *cv && strchr("0123456789", *cv) ? cv : "0");
    for (char *c = cmd; *c; c++) if (*c == ',' && c > strstr(cmd, "-i ")) { *c = 0; strcat(cmd, " 2>/dev/null"); break; }
    FILE *p = popen(cmd, "r"); out[0] = 0;
    if (p) { if (!fgets(out, (int)n, p)) out[0] = 0; pclose(p); }
    out[strcspn(out, "\n")] = 0; if (!out[0]) snprintf(out, n, "unknown");
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: test_heads <ufsm binary> <make_wind_fixture binary> <heads_ref.txt>\n"); return 2; }
    const char *U = argv[1], *ref = argv[3]; const int wr = getenv("UFSM_HEADS_REF_WRITE") && atoi(getenv("UFSM_HEADS_REF_WRITE"));
    char gpu[256]; gpu_name(gpu, sizeof gpu);
    char d[] = "/tmp/test_heads_XXXXXX"; if (!mkdtemp(d)) return 1;
    CHECK(!sh("%s %s/fx 256 2 1 > %s/fx.log 2>&1", argv[2], d, d), "fixture");
    /* the fibre plane of the combined runs: the fixture's ink raster again, as the sources' "fibre" target next to "ink" */
    CHECK(!sh("sed 's|\"ink\": *{\\([^}]*\\)}|\"ink\": {\\1}, \"fibre\": {\\1}|' %s/fx/sources-ink.json > %s/fx/sources-fib.json && grep -q fibre %s/fx/sources-fib.json", d, d, d), "sources with a fibre target");
    const char *common = "--task unwrap2 --levels 1 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 2 --val-every 1000 --log-every 1 "
                         "--ckpt-every 1000 --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --widths 16,32,64 --det 1 --steps 4";
    static const struct { const char *name, *src, *flags; } old[] = {
        {"plain", "sources.json", ""},
        {"ink", "sources-ink.json", "--ink-head 1"},
        {"rv", "sources.json", "--rv 1 --rv-iso 100"},
        {"fibn", "sources-ink.json", "--fib-normal-lambda 0.1"},
        {"lat_plain", "sources.json", "--lat 4 --lat-guide 3"},
        {"lat_rv", "sources.json", "--rv 1 --rv-iso 100 --lat 4 --lat-guide 3"},
        {"lat_ink", "sources-ink.json", "--ink-head 1 --lat 4 --lat-guide 3"},
    };
    char *refs = slurp(ref); FILE *rw = wr ? fopen(ref, "a") : nullptr;
    if (wr && !rw) { fprintf(stderr, "cannot append to %s\n", ref); return 1; }
    int have = 0, cmp = 0;
    for (size_t k = 0; k < sizeof old / sizeof *old; k++) {
        if (sh("UFSM_GRAD_MX8=1 %s train %s/fx/%s --out %s/%s %s %s > %s/%s.log 2>&1", U, d, old[k].src, d, old[k].name, common, old[k].flags, d, old[k].name)) { CHECK(0, "%s run (%s/%s.log)", old[k].name, d, old[k].name); continue; }
        char p[600]; snprintf(p, sizeof p, "%s/%s.log", d, old[k].name); char *lg = slurp(p), *t = lg ? loss_terms(lg) : nullptr;
        CHECK(t && strstr(t, "4:"), "%s: four logged steps", old[k].name);
        if (rw && t) fprintf(rw, "%s|%s|%s\n", gpu, old[k].name, t);
        else if (refs && t) {   /* the reference line of this GPU and configuration */
            char key[400]; snprintf(key, sizeof key, "%s|%s|", gpu, old[k].name); const char *r = strstr(refs, key);
            if (r) {
                have++; r += strlen(key); const size_t rl = strcspn(r, "\n");
                const int same = strlen(t) == rl && !strncmp(t, r, rl); cmp += same;
                CHECK(same, "%s: the logged losses differ from the reference\n  got  %s\n  want %.*s", old[k].name, t, (int)rl, r);
            }
        }
        free(t); free(lg);
    }
    if (rw) { fclose(rw); printf("test_heads: reference lines for %s appended to %s\n", gpu, ref); }
    else if (!have) fprintf(stderr, "test_heads: no reference for GPU \"%s\" in %s: the old layouts ran, their losses not compared\n", gpu, ref);
    else fprintf(stderr, "test_heads: %d of %d old layouts log the reference losses (%s)\n", cmp, have, gpu);
    free(refs);
    /* (2) the combined heads: rv on its own channels beside the phase, ink and the fibre term together, whole-window and latent */
    static const struct { const char *name, *flags; } comb[] = {
        {"all", "--rv 2 --rv-iso 100 --ink-head 1 --fib-normal-lambda 0.1"},
        {"lat_all", "--rv 2 --rv-iso 100 --ink-head 1 --fib-normal-lambda 0.1 --lat 4 --lat-guide 3"},
    };
    for (size_t k = 0; !wr && k < sizeof comb / sizeof *comb; k++) {
        if (sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-fib.json --out %s/%s %s %s > %s/%s.log 2>&1", U, d, d, comb[k].name, common, comb[k].flags, d, comb[k].name)) { CHECK(0, "%s run (%s/%s.log)", comb[k].name, d, comb[k].name); continue; }
        char p[600]; snprintf(p, sizeof p, "%s/%s.log", d, comb[k].name); char *lg = slurp(p);
        const char *last = nullptr; for (const char *q = lg; q && (q = strstr(q, "\nstep ")); q++) last = q + 1;
        double rv = -1, ink = -1, fib = -1, phase = -1;
        if (last) {
            const char *a = strstr(last, " rv bce "), *b = strstr(last, " ink bce "), *c = strstr(last, " fibn "), *e = strstr(last, " phase bce ");
            rv = a ? atof(a + 8) : -1; ink = b ? atof(b + 9) : -1; fib = c ? atof(c + 6) : -1; phase = e ? atof(e + 11) : -1;
        }
        CHECK(last && isfinite(rv) && rv > 0 && isfinite(ink) && ink > 0 && isfinite(fib) && fib > -1 && isfinite(phase) && phase > 0,
              "%s: finite rv %g / ink %g / fibn %g / phase %g losses in the last step line", comb[k].name, rv, ink, fib, phase);
        CHECK(lg && strstr(lg, "head slabs:"), "%s: head slabs", comb[k].name);
        free(lg);
        snprintf(p, sizeof p, "%s/%s/last.ckpt", d, comb[k].name); FILE *f = fopen(p, "rb"); char hdr[16384] = {0};
        if (f) { if (!fgets(hdr, sizeof hdr, f)) hdr[0] = 0; fclose(f); }
        CHECK(strstr(hdr, "\"rv\":{\"version\":2,\"recto\":12,\"verso\":13") && strstr(hdr, "\"ink\":11"), "%s: checkpoint header names rv on channels 12-13 and ink on 11", comb[k].name);
        /* (3) predict --channel rv on the --rv 2 checkpoint: 1 + 254 (1 + p_r - p_v) / 2 of its channels 12 / 13 (lossless stores; within one code) */
        const char *pr = "--um 9.6 --level 0 --window 160 --halo 16 --shard 128 --levels 1 --threads 2 --q 0";
        static const char *ch[3] = {"rv", "12", "13"}; int ok = 1;
        for (int c = 0; c < 3 && ok; c++)
            ok = !sh("%s predict %s/%s/last.ckpt %s/fx ct %s/p%s_%s --channel %s %s > %s/p%s_%s.log 2>&1 && %s read %s/p%s_%s 9.6 64 64 64 128 128 128 %s/p%s_%s.raw > /dev/null 2>&1",
                     U, d, comb[k].name, d, d, comb[k].name, ch[c], ch[c], pr, d, comb[k].name, ch[c], U, d, comb[k].name, ch[c], d, comb[k].name, ch[c]);
        CHECK(ok, "%s: predict --channel rv / 12 / 13 (%s/p%s_*.log)", comb[k].name, d, comb[k].name);
        if (ok) {
            enum { NV = 128 * 128 * 128 }; uint8_t *v[3]; int got = 1;
            for (int c = 0; c < 3; c++) { snprintf(p, sizeof p, "%s/p%s_%s.raw", d, comb[k].name, ch[c]); FILE *rf = fopen(p, "rb"); v[c] = malloc(NV); got &= rf && fread(v[c], 1, NV, rf) == NV; if (rf) fclose(rf); }
            double worst = 0, sum = 0; size_t n = 0;
            for (size_t q = 0; got && q < NV; q++) if (v[0][q]) { const double e = fabs(v[0][q] - (1 + 254 * (1 + v[1][q] / 255. - v[2][q] / 255.) / 2)); if (e > worst) worst = e; sum += e; n++; }
            CHECK(got && n > NV / 2 && worst < 1.5, "%s: --channel rv against channels 12 / 13: worst %g codes, mean %g over %zu voxels", comb[k].name, worst, n ? sum / n : 0.0, n);
            for (int c = 0; c < 3; c++) free(v[c]);
        }
    }
    /* (4) the sources' "heads" key: every head at 0 trains nothing (gradient norm 0); an unknown head name is refused */
    if (!wr) {
        CHECK(!sh("sed 's|\"weight\": 1,|\"weight\": 1, \"heads\": {\"phase\": 0, \"normal\": 0, \"density\": 0, \"wind4\": 0, \"rv\": 0, \"ink\": {\"on\": 0}, \"fibre\": 0},|' %s/fx/sources-fib.json > %s/fx/sources-off.json"
                  " && grep -q heads %s/fx/sources-off.json && sed 's|\"rv\": 0|\"rvv\": 0|' %s/fx/sources-off.json > %s/fx/sources-typo.json", d, d, d, d, d), "sources with heads off");
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-off.json --out %s/off %s %s > %s/off.log 2>&1", U, d, d, common, comb[0].flags, d), "heads-off run (%s/off.log)", d);
        char p[600]; snprintf(p, sizeof p, "%s/off.log", d); char *lg = slurp(p); int nz = 0, ns = 0;
        for (const char *q = lg; q && (q = strstr(q, "\nstep ")); q++) { const char *g = strstr(q, " gn "); ns++; nz += g && atof(g + 4) != 0; }
        CHECK(ns == 4 && nz == 0, "heads off: %d of %d steps with a nonzero gradient norm", nz, ns);
        free(lg);
        CHECK(sh("%s train %s/fx/sources-typo.json --out %s/typo %s %s > %s/typo.log 2>&1", U, d, d, common, comb[0].flags, d) != 0, "an unknown head name was accepted");
    }
    /* (5) --um-rungs with --film 2 on two sources whose labels allow one level each (phase at level 1 / level 2): both rungs train, the
       checkpoint names "film":2 and resumes; --levels with --um-rungs is refused */
    if (!wr) {
        CHECK(!sh("{ printf '{\"sources\": ['; sed -n 's/^{\"sources\": \\[//p; /targets/p' %s/fx/sources-l1.json | tr -d '\\n' | sed 's/]}$//'; printf ',\\n'; "
                  "sed -n 's/^{\"sources\": \\[//p; /targets/p' %s/fx/sources2.json | grep -v holdout | tr -d '\\n'; printf '\\n'; } > %s/fx/sources-mix.json && grep -c phase2 %s/fx/sources-mix.json > /dev/null", d, d, d, d), "two-source fixture");
        const char *rf = "--lat 4 --lat-guide 3 --rv 2 --rv-iso 100 --um-rungs 19.2,38.4 --rung-shift-p 0.3 --film 2";
        char cm[2048]; snprintf(cm, sizeof cm, "%s", common); for (int q = 0; q < 2; q++) { const char *cut = q ? " --steps 4" : "--levels 1 "; char *ls = strstr(cm, cut); if (ls) memmove(ls, ls + strlen(cut), strlen(ls + strlen(cut)) + 1); }   /* (the rungs pick the levels; steps below) */
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-mix.json --out %s/rung %s %s --steps 4 > %s/rung.log 2>&1", U, d, d, cm, rf, d), "rungs + film run (%s/rung.log)", d);
        char p[600]; snprintf(p, sizeof p, "%s/rung.log", d); char *lg = slurp(p);
        CHECK(lg && strstr(lg, "levels by physical voxel size") && strstr(lg, "\nstep      4 "), "rungs + film: 4 steps");
        free(lg);
        snprintf(p, sizeof p, "%s/rung/last.ckpt", d); FILE *f = fopen(p, "rb"); char hdr[16384] = {0};
        if (f) { if (!fgets(hdr, sizeof hdr, f)) hdr[0] = 0; fclose(f); }
        CHECK(strstr(hdr, "\"film\":2"), "rungs + film: checkpoint header names film 2");
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-mix.json --out %s/rung --resume %s/rung/last.ckpt %s %s --steps 6 > %s/rung2.log 2>&1 && grep -q 'step      6 ' %s/rung2.log", U, d, d, d, cm, rf, d, d), "rungs + film resume (%s/rung2.log)", d);
        CHECK(sh("%s train %s/fx/sources-mix.json --out %s/rung3 %s %s > %s/rung3.log 2>&1", U, d, d, common, rf, d) != 0, "--levels with --um-rungs accepted");
    }
    /* (6) --uw 1 + --grad-cos 2 (latent, all heads): the cosine lines name every head pair, the weights move off 1, the header carries
       "uw" with t 4 and a resume continues from it (t 6) */
    if (!wr) {
        const char *uf = "--lat 4 --lat-guide 3 --rv 2 --rv-iso 100 --ink-head 1 --uw 1 --uw-lr 0.05 --grad-cos 2";
        char cm[2048]; snprintf(cm, sizeof cm, "%s", common); { char *ls = strstr(cm, " --steps 4"); if (ls) memmove(ls, ls + 10, strlen(ls + 10) + 1); }
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-ink.json --out %s/uw %s %s --steps 4 > %s/uw.log 2>&1", U, d, d, cm, uf, d), "uw + grad-cos run (%s/uw.log)", d);
        CHECK(!sh("grep -q 'grad-cos step 4 (head input): phase/normal [-+]0[.][0-9]*.*rv/ink [-+]0[.]' %s/uw.log && grep -q 'uw phase [0-9.]* normal 0[.]' %s/uw.log", d, d), "uw + grad-cos: the cosine and weight lines");
        CHECK(!sh("head -1 %s/uw/last.ckpt | grep -q '\"uw\":{\"t\":4,'", d), "uw: header t 4");
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-ink.json --out %s/uw --resume %s/uw/last.ckpt %s %s --steps 6 > %s/uw2.log 2>&1 && grep -q '(resumed)' %s/uw2.log && head -1 %s/uw/last.ckpt | grep -q '\"uw\":{\"t\":6,'", U, d, d, d, cm, uf, d, d, d), "uw resume (%s/uw2.log)", d);
    }
    /* (7) --fstem 16 (latent, all heads): the stem's projection starts at zero, so step 1 logs the stem-less run's losses; the header names
       "fstem":[16,7]; a resume continues; a resume without --fstem is refused; predict --channel rv runs on it */
    if (!wr) {
        const char *ff = "--lat 4 --lat-guide 3 --rv 2 --rv-iso 100 --ink-head 1";
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-ink.json --out %s/fs0 %s %s > %s/fs0.log 2>&1 && UFSM_GRAD_MX8=1 %s train %s/fx/sources-ink.json --out %s/fs1 %s %s --fstem 16 > %s/fs1.log 2>&1",
                  U, d, d, common, ff, d, U, d, d, common, ff, d), "fstem runs (%s/fs0.log, fs1.log)", d);
        CHECK(!sh("[ \"$(grep '^step      1 ' %s/fs0.log | sed 's/ gn .*//')\" = \"$(grep '^step      1 ' %s/fs1.log | sed 's/ gn .*//')\" ] && grep -q 'fine stem: 16' %s/fs1.log", d, d, d),
              "fstem: step 1 losses differ from the stem-less run's");   /* (the gradient norm counts the stem's own gradients) */
        CHECK(!sh("head -1 %s/fs1/last.ckpt | grep -q '\"fstem\":\\[16,7\\]'", d), "fstem: header");
        char cm[2048]; snprintf(cm, sizeof cm, "%s", common); { char *ls = strstr(cm, " --steps 4"); if (ls) memmove(ls, ls + 10, strlen(ls + 10) + 1); }
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-ink.json --out %s/fs1 --resume %s/fs1/last.ckpt %s %s --fstem 16 --steps 6 > %s/fs2.log 2>&1 && grep -q 'step      6 ' %s/fs2.log", U, d, d, d, cm, ff, d, d), "fstem resume (%s/fs2.log)", d);
        CHECK(sh("%s train %s/fx/sources-ink.json --out %s/fs1 --resume %s/fs1/last.ckpt %s %s --steps 8 > %s/fs3.log 2>&1", U, d, d, d, cm, ff, d) != 0, "fstem: a resume without --fstem accepted");
        CHECK(!sh("%s predict %s/fs1/last.ckpt %s/fx ct %s/pfs --channel rv --um 9.6 --level 0 --window 160 --halo 16 --shard 128 --levels 1 --threads 2 --q 0 > %s/pfs.log 2>&1", U, d, d, d, d), "fstem: predict (%s/pfs.log)", d);
    }
    /* (8) --ssl 8 (latent, src/ssl.h): (a) without an unlabelled source the side head gets no gradient and the run logs lat_plain's
       losses; (b) with a "ct_only" copy of the fixture source at --ssl-share 0.5 the step lines carry a finite ssl term over the
       unlabelled windows, the teacher targets about the mask share of the cells, the header names the side head, a resume continues */
    if (!wr) {
        CHECK(!sh("sed 's|}}}}]}|}}}}, {\"name\": \"wind-fixture-u\", \"root\": \"%s/fx\", \"ct\": \"ct\", \"um\": 9.6, \"axis\": \"%s/fx/axis.json\", \"weight\": 1, \"ct_only\": 1}]}|' %s/fx/sources.json > %s/fx/sources-ssl.json"
                  " && grep -q ct_only %s/fx/sources-ssl.json", d, d, d, d, d), "sources with an unlabelled copy");
        const char *sf = "--lat 4 --lat-guide 3 --ssl 8 --ssl-level 1 --ssl-block 16";
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources.json --out %s/ssl0 %s %s > %s/ssl0.log 2>&1", U, d, d, common, sf, d), "ssl without unlabelled sources (%s/ssl0.log)", d);
        char p[600]; snprintf(p, sizeof p, "%s/lat_plain.log", d); char *l0 = slurp(p); snprintf(p, sizeof p, "%s/ssl0.log", d); char *l1 = slurp(p);
        char *t0 = l0 ? loss_terms(l0) : nullptr, *t1 = l1 ? loss_terms(l1) : nullptr;
        CHECK(t0 && t1 && !strcmp(t0, t1), "ssl without unlabelled windows: the losses differ from lat_plain's\n  got  %s\n  want %s", t1 ? t1 : "-", t0 ? t0 : "-");
        free(t0); free(t1); free(l0); free(l1);
        char cm[2048]; snprintf(cm, sizeof cm, "%s", common); { char *ls = strstr(cm, " --steps 4"); if (ls) memmove(ls, ls + 10, strlen(ls + 10) + 1); }
        CHECK(!sh("UFSM_SSL_CHECK=1 UFSM_GRAD_MX8=1 %s train %s/fx/sources-ssl.json --out %s/ssl %s %s --ssl-share 0.5 --steps 8 > %s/ssl.log 2>&1", U, d, d, cm, sf, d), "ssl run (%s/ssl.log)", d);
        snprintf(p, sizeof p, "%s/ssl.log", d); char *lg = slurp(p); int nfin = 0, nchk = 0, nbad = 0;
        for (const char *q = lg; q && (q = strstr(q, "\nstep ")); q++) { const char *a = strstr(q, " ssl "), *e = strchr(q + 1, '\n'); if (a && (!e || a < e)) { const double v = atof(a + 5); nfin += isfinite(v) && v > 0; } }
        for (const char *q = lg; q && (q = strstr(q, "ssl check (GPU ")); q++) { const char *a = strstr(q, "): "); const double t = a ? atof(a + 3) : -1; nchk++; nbad += !(t > 10 && t < 75); }
        CHECK(nfin >= 2 && nchk >= 2 && !nbad, "ssl: %d step lines with a finite ssl term, %d checks (%d with a targeted share outside 10..75%%)", nfin, nchk, nbad);
        free(lg);
        CHECK(!sh("head -1 %s/ssl/last.ckpt | grep -q '\"emb_lev\":1,\"emb_ch\":8'", d), "ssl: header names the side head");
        /* (d) a run without the side head resumes with --ssl (the head's block inserted fresh, the rest and the step the checkpoint's) */
        CHECK(!sh("cp -r %s/lat_plain %s/lp2 && UFSM_GRAD_MX8=1 %s train %s/fx/sources-ssl.json --out %s/lp2 --resume %s/lp2/last.ckpt %s %s --ssl-share 0.5 --steps 8 > %s/lp2.log 2>&1"
                  " && grep -q 'has no side head' %s/lp2.log && grep -q 'step      8 ' %s/lp2.log && grep -Eq 'std 0[.](0[1-9]|[1-9])' %s/lp2.log && head -1 %s/lp2/last.ckpt | grep -q '\"emb_lev\":1,\"emb_ch\":8'", d, d, U, d, d, d, cm, sf, d, d, d, d, d), "ssl: resume of a run without the side head (%s/lp2.log)", d);
        /* (e) --ssl-ink (latent, ink head): the ink-ssl term on the unlabelled windows is finite and covers the unmasked cells (conf 0, no papyrus gate) */
        CHECK(!sh("sed 's|}}}}]}|}}}}, {\"name\": \"wind-fixture-u\", \"root\": \"%s/fx\", \"ct\": \"ct\", \"um\": 9.6, \"axis\": \"%s/fx/axis.json\", \"weight\": 1, \"ct_only\": 1}]}|' %s/fx/sources-ink.json > %s/fx/sources-ink-ssl.json"
                  " && grep -q ct_only %s/fx/sources-ink-ssl.json", d, d, d, d, d), "ink sources with an unlabelled copy (%s/fx/sources-ink.json's ending)", d);
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-ink-ssl.json --out %s/sslink %s %s --ink-head 1 --ssl-share 0.5 --ssl-ink 1 --ssl-ink-conf 0 --ssl-ink-pap 0 --steps 8 > %s/sslink.log 2>&1", U, d, d, cm, sf, d), "ssl-ink run (%s/sslink.log)", d);
        { snprintf(p, sizeof p, "%s/sslink.log", d); char *li = slurp(p); int nk = 0, nbad = 0;
          for (const char *q = li; q && (q = strstr(q, " ssl-ink ")); q++) { const double v = atof(q + 9); const char *c = strchr(q, '('); const double sh_ = c ? atof(c + 1) : 0; nk++; nbad += !(isfinite(v) && v > 0 && sh_ > 30 && sh_ < 90); }
          CHECK(nk >= 2 && !nbad, "ssl-ink: %d step lines with the term, %d with a non-finite loss or a cell share outside 30..90%%", nk, nbad); free(li); }
        /* (f) --ssl-heads over every self-trainable head of a --rv 2 ink net (confidence 0, no papyrus gate): each head's term finite on the
           unlabelled windows and covering the unmasked cells; a head the net lacks is refused */
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-ink-ssl.json --out %s/sslh %s %s --ink-head 1 --rv 2 --rv-iso 64 --ssl-share 0.5 --ssl-heads ink,rv,dens,normal,phase:0.5"
                  " --ssl-conf 0 --ssl-ink-conf 0 --ssl-norm-conf 0 --ssl-phase-conf 0 --ssl-pap 0 --ssl-ink-pap 0 --steps 8 > %s/sslh.log 2>&1", U, d, d, cm, sf, d), "ssl-heads run (%s/sslh.log)", d);
        { snprintf(p, sizeof p, "%s/sslh.log", d); char *li = slurp(p); const char *hn[] = {" ssl-ink ", " ssl-rv ", " ssl-dens ", " ssl-normal ", " ssl-phase "};
          for (int h = 0; h < 5; h++) { int nk = 0, nbad = 0;
              for (const char *q = li; q && (q = strstr(q, hn[h])); q++) { const double v = atof(q + strlen(hn[h])); const char *c = strchr(q, '('); const double sh_ = c ? atof(c + 1) : 0; nk++; nbad += !(isfinite(v) && v > 0 && sh_ > 30 && sh_ < 90); }
              CHECK(nk >= 2 && !nbad, "ssl-heads%s: %d step lines with the term, %d with a non-finite loss or a cell share outside 30..90%%", hn[h], nk, nbad); }
          free(li); }
        CHECK(sh("%s train %s/fx/sources-ink-ssl.json --out %s/sslhx %s %s --ink-head 1 --ssl-heads fib --steps 1 > %s/sslhx.log 2>&1", U, d, d, cm, sf, d) && !sh("grep -q 'the net has no such head' %s/sslhx.log", d),
              "ssl-heads: a head the net lacks is refused (%s/sslhx.log)", d);
        /* (c) unlabelled windows skip the head slabs: the same losses as with the slabs run on zero masks (UFSM_SSL_SLABS=1) */
        CHECK(!sh("UFSM_SSL_SLABS=1 UFSM_GRAD_MX8=1 %s train %s/fx/sources-ssl.json --out %s/sslb %s %s --ssl-share 0.5 --steps 8 > %s/sslb.log 2>&1", U, d, d, cm, sf, d), "ssl run with slabs (%s/sslb.log)", d);
        snprintf(p, sizeof p, "%s/ssl.log", d); l0 = slurp(p); snprintf(p, sizeof p, "%s/sslb.log", d); l1 = slurp(p);
        t0 = l0 ? loss_terms(l0) : nullptr; t1 = l1 ? loss_terms(l1) : nullptr;
        CHECK(t0 && t1 && !strcmp(t0, t1), "ssl: skipping the head slabs changed the losses\n  skip   %s\n  slabs  %s", t0 ? t0 : "-", t1 ? t1 : "-");
        free(t0); free(t1); free(l0); free(l1);
        CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-ssl.json --out %s/ssl --resume %s/ssl/last.ckpt %s %s --ssl-share 0.5 --steps 10 > %s/ssl2.log 2>&1 && grep -q 'step     10 ' %s/ssl2.log", U, d, d, d, cm, sf, d, d), "ssl resume (%s/ssl2.log)", d);
    }
    if (!fails) sh("rm -rf %s", d);
    if (fails) { fprintf(stderr, "test_heads: %d FAILED (kept %s)\n", fails, d); return 1; }
    printf("test_heads: ok\n");
    return 0;
}
