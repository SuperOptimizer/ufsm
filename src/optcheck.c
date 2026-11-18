#include "optcheck.h"
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* the names asked for: an open-addressing set of copied strings (names can be built in stack buffers: train's --aug-%s-p), growable
   (train alone asks for ~320 names; the old fixed 512-slot list dropped names past the cap: mini audit opts 7) */
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static char **g_set; static size_t g_cap, g_n;   /* mgpu: main */
static char *g_pre[64]; static int g_npre;   /* mgpu: main */
static char **g_warned; static size_t g_nw, g_cw;   /* mgpu: main */

static uint64_t h64(const char *s) { uint64_t h = 1469598103934665603ull; for (; *s; s++) h = (h ^ (uint8_t)*s) * 1099511628211ull; return h; }
static int set_has(const char *s) {
    if (!g_cap) return 0;
    for (size_t i = h64(s) & (g_cap - 1);; i = (i + 1) & (g_cap - 1)) { if (!g_set[i]) return 0; if (!strcmp(g_set[i], s)) return 1; }
}
static void set_put(char *s) {
    if (2 * (g_n + 1) > g_cap) {   /* grow: load factor <= 1/2 */
        const size_t nc = g_cap ? 2 * g_cap : 1024; char **ns = calloc(nc, sizeof *ns); if (!ns) return;
        for (size_t k = 0; k < g_cap; k++) if (g_set[k]) { size_t i = h64(g_set[k]) & (nc - 1); while (ns[i]) i = (i + 1) & (nc - 1); ns[i] = g_set[k]; }
        free(g_set); g_set = ns; g_cap = nc;
    }
    size_t i = h64(s) & (g_cap - 1); while (g_set[i]) i = (i + 1) & (g_cap - 1);
    g_set[i] = s; g_n++;
}
const char *opt_n(const char *name) {
    if (!name || name[0] != '-' || name[1] != '-' || !name[2]) return name;   /* only "--x" names (helpers also compare positional words) */
    pthread_mutex_lock(&g_mu);
    if (!set_has(name)) { char *c = strdup(name); if (c) set_put(c); }
    pthread_mutex_unlock(&g_mu);
    return name;
}
const char *opt_p(const char *prefix) {
    if (!prefix || prefix[0] != '-' || prefix[1] != '-' || !prefix[2]) return prefix;
    pthread_mutex_lock(&g_mu);
    int have = 0; for (int k = 0; k < g_npre && !have; k++) have = !strcmp(g_pre[k], prefix);
    if (!have && g_npre < 64) g_pre[g_npre++] = strdup(prefix);
    pthread_mutex_unlock(&g_mu);
    return prefix;
}
static int seen_l(const char *a) {
    if (set_has(a)) return 1;
    for (int k = 0; k < g_npre; k++) if (g_pre[k] && !strncmp(a, g_pre[k], strlen(g_pre[k]))) return 1;
    return 0;
}
static int warned_l(const char *a) { for (size_t k = 0; k < g_nw; k++) if (!strcmp(g_warned[k], a)) return 1; return 0; }
int opt_unread(const char *who, int argc, char **argv) {
    int n = 0;
    pthread_mutex_lock(&g_mu);
    for (int i = 2; i < argc; i++) {   /* argv[1] is the command */
        const char *a = argv[i];
        if (!a || a[0] != '-' || a[1] != '-' || !a[2]) continue;   /* values, positionals, negative numbers ("-1") */
        if (seen_l(a)) { if (i + 1 < argc && strncmp(argv[i + 1], "--", 2)) i++; continue; }   /* its value (if it has one) */
        if (warned_l(a)) continue;
        fprintf(stderr, "%s: WARNING: option %s is not one %s reads (a typo, or an env-var switch such as UFSM_SAMPLER_ONLY?)\n", who, a, who); n++;
        if (g_nw == g_cw) { const size_t nc = g_cw ? 2 * g_cw : 16; char **w = realloc(g_warned, nc * sizeof *w); if (!w) continue; g_warned = w; g_cw = nc; }
        g_warned[g_nw++] = strdup(a);
    }
    pthread_mutex_unlock(&g_mu);
    return n;
}

[[noreturn]] static void bad(const char *name, const char *v, const char *what) {
    fprintf(stderr, "option %s: \"%s\" is not %s\n", name ? name : "?", v, what);
    exit(2);
}
/* the end of a number: trailing blanks, then nothing or a list (",..."): list-valued options (--onfly-gpu A,B, --gpus, --widths ...)
   are read for their first element here, the code that splits the list reads the rest (2026-11-14 sweep: test_phase_onfly's
   "--onfly-gpu 0,0" was refused). Junk right after the number ("2e4" for an integer, "288x", "1.5x") still is an error. */
static const char *num_end(const char *e) { while (*e == ' ' || *e == '\t' || *e == '\n') e++; return *e == ',' ? e + strlen(e) : e; }
long long opt_atoll(const char *name, const char *v) {
    if (!v || !*v) return 0;
    char *e; errno = 0; const long long r = strtoll(v, &e, 10);
    if (e == v || *num_end(e)) bad(name, v, "an integer");
    if (errno == ERANGE) bad(name, v, "an integer in range");
    return r;
}
long opt_atol(const char *name, const char *v) { const long long r = opt_atoll(name, v); if (r < LONG_MIN || r > LONG_MAX) bad(name, v, "an integer in range"); return (long)r; }
int opt_atoi(const char *name, const char *v) { const long long r = opt_atoll(name, v); if (r < INT_MIN || r > INT_MAX) bad(name, v, "an integer in int range"); return (int)r; }
double opt_atof(const char *name, const char *v) {
    if (!v || !*v) return 0;
    char *e; const double r = strtod(v, &e);
    if (e == v || *num_end(e)) bad(name, v, "a number");
    return r;
}

/* opt_match: '%' [flags / width / precision / length] conversion matches any run of characters (incl. empty); the rest literally */
int opt_match(const char *p, const char *s) {
    for (; *p; p++, s++) {
        if (*p == '%' && p[1] && p[1] != '%') {
            const char *q = p + 1; while (*q && strchr("-+ #0123456789.hlLqjzt", *q)) q++;
            if (!*q) return 0;
            for (const char *t = s; ; t++) { if (opt_match(q + 1, t)) return 1; if (!*t) return 0; }
        }
        if (*p == '%' && p[1] == '%') p++;
        if (*s != *p) return 0;
    }
    return !*s;
}
int opt_unknown(const char *who, int argc, char **argv, const char *const *names, const char *const *prefixes) {
    int n = 0;
    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        if (!a || a[0] != '-' || a[1] != '-' || !a[2]) continue;   /* values, positionals, "-1", a bare "--" */
        char nm[256]; size_t l = strcspn(a, "="); if (l >= sizeof nm) l = sizeof nm - 1; memcpy(nm, a, l); nm[l] = 0;
        int ok = 0;
        for (int k = 0; names && names[k] && !ok; k++) ok = strchr(names[k], '%') ? opt_match(names[k], nm) : !strcmp(names[k], nm);
        for (int k = 0; prefixes && prefixes[k] && !ok; k++) ok = !strncmp(nm, prefixes[k], strlen(prefixes[k]));
        if (!ok) { fprintf(stderr, "%s: ERROR: unknown option %s (no command of this binary reads it: a typo, a newer binary's flag, or an env-var switch?"
                                   " UFSM_OPT_LAX=1 runs anyway)\n", who, nm); n++; }
    }
    return n;
}
