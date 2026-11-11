/* ufsm annotate-serve: a local (127.0.0.1) HTTP viewer + annotation tool for manual winding labels. One self-contained page
   (src/annot_page.html, embedded with #embed) shows CT slices on the three axis planes at the CT pyramid's levels, with our wraps
   (plane cuts of each wrap's vertex grid), a label array (phase / codes) and fibres (UFIB points near the plane) as overlays, and
   records five annotation types in a per-scroll store (src/annot.h). Decoded CT chunks are kept in an LRU (--cache-gb), so panning
   and stepping through neighbouring slices reads nothing new. Wraps load in a background thread (the .sfc decoder is not
   thread-safe), so the page is usable at once and the wrap overlay fills in. */
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <dirent.h>
#include <arpa/inet.h>
#include "annot.h"
#include "imgops.h"
#include "json.h"
#include "store.h"
#include "tiffw.h"
#include "zarr3.h"

/* ---- a growable string -------------------------------------------------------------------------------------------------------- */
typedef struct { char *s; size_t n, cap; } sb;
static void sb_put(sb *b, const char *t, size_t n) {
    if (b->n + n + 1 > b->cap) { b->cap = (b->n + n + 1) * 2 + 64; b->s = realloc(b->s, b->cap); }
    memcpy(b->s + b->n, t, n); b->n += n; b->s[b->n] = 0;
}
static void sb_str(sb *b, const char *t) { sb_put(b, t, strlen(t)); }
static void sb_fmt(sb *b, const char *f, ...) __attribute__((format(printf, 2, 3)));
static void sb_fmt(sb *b, const char *f, ...) {
    char tmp[512]; va_list ap; va_start(ap, f); int k = vsnprintf(tmp, sizeof tmp, f, ap); va_end(ap);
    if (k < 0) return;
    if ((size_t)k < sizeof tmp) { sb_put(b, tmp, (size_t)k); return; }
    char *big = malloc((size_t)k + 1); va_start(ap, f); vsnprintf(big, (size_t)k + 1, f, ap); va_end(ap); sb_put(b, big, (size_t)k); free(big);
}
static void sb_jstr(sb *b, const char *t) {   /* a JSON string literal */
    sb_put(b, "\"", 1);
    for (const unsigned char *p = (const unsigned char *)t; *p; p++) {
        if (*p == '"' || *p == '\\') { char e[2] = {'\\', (char)*p}; sb_put(b, e, 2); }
        else if (*p < 0x20) sb_fmt(b, "\\u%04x", *p);
        else sb_put(b, (const char *)p, 1);
    }
    sb_put(b, "\"", 1);
}
static void sb_num(sb *b, double v) {
    if (!isfinite(v)) { sb_str(b, "null"); return; }
    if (v == floor(v) && fabs(v) < 1e15) sb_fmt(b, "%.0f", v); else sb_fmt(b, "%.6g", v);
}
/* a parsed JSON node back to text (keys in skip[] of the top object left out) */
static void jdump(sb *b, const json *j, const char *const *skip, int nskip) {
    if (!j) { sb_str(b, "null"); return; }
    switch (j->type) {
    case J_NULL: sb_str(b, "null"); break;
    case J_BOOL: sb_str(b, j->num ? "true" : "false"); break;
    case J_NUM: sb_num(b, j->num); break;
    case J_STR: sb_jstr(b, j->str); break;
    case J_ARR: sb_str(b, "[");
        for (size_t i = 0; i < j->n; i++) { if (i) sb_str(b, ","); jdump(b, j->items[i], nullptr, 0); }
        sb_str(b, "]"); break;
    case J_OBJ: { sb_str(b, "{"); int first = 1;
        for (size_t i = 0; i < j->n; i++) {
            int sk = 0; for (int k = 0; k < nskip; k++) if (!strcmp(j->keys[i], skip[k])) sk = 1;
            if (sk) continue;
            if (!first) sb_str(b, ","); first = 0; sb_jstr(b, j->keys[i]); sb_str(b, ":"); jdump(b, j->items[i], nullptr, 0);
        }
        sb_str(b, "}"); break; }
    }
}

/* ---- the annotation store ----------------------------------------------------------------------------------------------------- */
typedef struct { int id, dead; char *text; } aitem;
typedef struct { int op, id; } aundo;   /* op 0: an add, 1: a del */
struct annot_store {
    char *path, *scroll, *scan; int next_id;
    aitem *it; int n, cap;
    aundo *u; int nu, ucap;
    pthread_mutex_t mu;
};
static int is_vec3(const json *a) { return a && a->type == J_ARR && a->n == 3 && a->items[0]->type == J_NUM && a->items[1]->type == J_NUM && a->items[2]->type == J_NUM; }
/* the item's fields check out for its type: "" or the reason */
static const char *annot_check(const json *o) {
    if (!o || o->type != J_OBJ) return "not an object";
    const char *t = json_str(json_get(o, "type"), "");
    if (!strcmp(t, "point")) return is_vec3(json_get(o, "p")) ? "" : "point needs p [z,y,x]";
    if (!strcmp(t, "pair")) return is_vec3(json_get(o, "a")) && is_vec3(json_get(o, "b")) && json_get(o, "dw") && json_get(o, "dw")->type == J_NUM ? "" : "pair needs a, b [z,y,x] and dw";
    if (!strcmp(t, "line")) { const json *p = json_get(o, "pts"); if (!p || p->type != J_ARR || p->n < 2) return "line needs >= 2 pts";
        for (size_t i = 0; i < p->n; i++) if (!is_vec3(p->items[i])) return "line pts must be [z,y,x]"; return ""; }
    if (!strcmp(t, "region")) { const json *bx = json_get(o, "box"); const char *v = json_str(json_get(o, "verdict"), "");
        if (!bx || bx->type != J_ARR || bx->n != 6) return "region needs box [z0,y0,x0,z1,y1,x1]";
        return !strcmp(v, "good") || !strcmp(v, "bad") || !strcmp(v, "unknown") ? "" : "region verdict good|bad|unknown"; }
    if (!strcmp(t, "wraperr")) return is_vec3(json_get(o, "p")) && json_get(o, "wrap") ? "" : "wraperr needs p and wrap";
    return "type must be point|pair|line|region|wraperr";
}
static void st_push(annot_store *s, int id, char *text) {
    if (s->n == s->cap) { s->cap = s->cap ? 2 * s->cap : 64; s->it = realloc(s->it, (size_t)s->cap * sizeof *s->it); }
    s->it[s->n++] = (aitem){id, 0, text};
}
static void st_log(annot_store *s, const char *line) {
    char p[4200]; snprintf(p, sizeof p, "%s.log", s->path); FILE *f = fopen(p, "a");
    if (!f) { fprintf(stderr, "annot: cannot append to %s: %s\n", p, strerror(errno)); return; }
    fprintf(f, "%s\n", line); fflush(f); fsync(fileno(f)); fclose(f);
}
static char *st_snapshot_locked(annot_store *s) {
    sb b = {0}; sb_str(&b, "{\"version\":1,\"scroll\":"); sb_jstr(&b, s->scroll ? s->scroll : ""); sb_str(&b, ",\"scan\":");
    sb_jstr(&b, s->scan ? s->scan : ""); sb_fmt(&b, ",\"next_id\":%d,\"items\":[", s->next_id); int first = 1;
    for (int i = 0; i < s->n; i++) if (!s->it[i].dead) { if (!first) sb_str(&b, ","); first = 0; sb_str(&b, s->it[i].text); }
    sb_str(&b, "]}\n"); return b.s;
}
static void st_save(annot_store *s) {   /* snapshot to <path>.tmp, renamed over <path> */
    char *t = st_snapshot_locked(s); char p[4200]; snprintf(p, sizeof p, "%s.tmp", s->path); FILE *f = fopen(p, "w");
    if (f) { fputs(t, f); fflush(f); fsync(fileno(f)); fclose(f); if (rename(p, s->path)) fprintf(stderr, "annot: rename %s: %s\n", p, strerror(errno)); }
    else fprintf(stderr, "annot: cannot write %s: %s\n", p, strerror(errno));
    free(t);
}
annot_store *annot_open(const char *path, const char *scroll, const char *scan) {
    annot_store *s = calloc(1, sizeof *s); s->path = strdup(path); s->scroll = scroll ? strdup(scroll) : nullptr; s->scan = scan ? strdup(scan) : nullptr;
    s->next_id = 1; pthread_mutex_init(&s->mu, nullptr);
    FILE *f = fopen(path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); char *t = malloc((size_t)n + 1);
        if (fread(t, 1, (size_t)n, f) != (size_t)n) n = 0; t[n] = 0; fclose(f);
        json *J = json_parse(t, (size_t)n); free(t);
        if (!J || J->type != J_OBJ) { fprintf(stderr, "annot: %s is not an annotation store\n", path); json_free(J); annot_close(s); return nullptr; }
        const json *it = json_get(J, "items"); int mx = 0;
        for (size_t i = 0; it && i < it->n; i++) {
            const int id = (int)json_num(json_get(it->items[i], "id"), 0); if (id <= 0) continue;
            sb b = {0}; jdump(&b, it->items[i], nullptr, 0); st_push(s, id, b.s); if (id > mx) mx = id;
        }
        s->next_id = (int)json_num(json_get(J, "next_id"), mx + 1); if (s->next_id <= mx) s->next_id = mx + 1;
        if (!s->scroll && json_get(J, "scroll")) s->scroll = strdup(json_str(json_get(J, "scroll"), ""));
        json_free(J);
    }
    return s;
}
void annot_close(annot_store *s) {
    if (!s) return;
    for (int i = 0; i < s->n; i++) free(s->it[i].text);
    free(s->it); free(s->u); free(s->path); free(s->scroll); free(s->scan); pthread_mutex_destroy(&s->mu); free(s);
}
int annot_add(annot_store *s, const char *item, const char *author, char *err, size_t errn) {
    json *o = json_parse(item, strlen(item)); const char *why = annot_check(o);
    if (*why) { snprintf(err, errn, "%s", why); json_free(o); return -1; }
    pthread_mutex_lock(&s->mu);
    const int id = s->next_id++; sb b = {0};
    sb_fmt(&b, "{\"id\":%d,\"t\":%lld,\"author\":", id, (long long)time(nullptr)); sb_jstr(&b, author ? author : "");
    static const char *const skip[] = {"id", "t", "author"}; sb t = {0}; jdump(&t, o, skip, 3); json_free(o);
    if (t.n > 2) { sb_str(&b, ","); sb_put(&b, t.s + 1, t.n - 1); } else sb_str(&b, "}");
    free(t.s); st_push(s, id, b.s);
    if (s->nu == s->ucap) { s->ucap = s->ucap ? 2 * s->ucap : 64; s->u = realloc(s->u, (size_t)s->ucap * sizeof *s->u); }
    s->u[s->nu++] = (aundo){0, id};
    sb l = {0}; sb_str(&l, "{\"op\":\"add\",\"item\":"); sb_str(&l, b.s); sb_str(&l, "}"); st_log(s, l.s); free(l.s);
    st_save(s); pthread_mutex_unlock(&s->mu); return id;
}
static aitem *st_find(annot_store *s, int id) { for (int i = 0; i < s->n; i++) if (s->it[i].id == id) return &s->it[i]; return nullptr; }
int annot_del(annot_store *s, int id) {
    pthread_mutex_lock(&s->mu); aitem *a = st_find(s, id);
    if (!a || a->dead) { pthread_mutex_unlock(&s->mu); return -1; }
    a->dead = 1;
    if (s->nu == s->ucap) { s->ucap = s->ucap ? 2 * s->ucap : 64; s->u = realloc(s->u, (size_t)s->ucap * sizeof *s->u); }
    s->u[s->nu++] = (aundo){1, id};
    char l[96]; snprintf(l, sizeof l, "{\"op\":\"del\",\"id\":%d}", id); st_log(s, l); st_save(s); pthread_mutex_unlock(&s->mu); return 0;
}
int annot_undo(annot_store *s) {
    pthread_mutex_lock(&s->mu);
    if (!s->nu) { pthread_mutex_unlock(&s->mu); return -1; }
    const aundo u = s->u[--s->nu]; aitem *a = st_find(s, u.id);
    if (a) a->dead = u.op == 0;   /* undo an add: dead; undo a del: alive again */
    char l[128]; snprintf(l, sizeof l, "{\"op\":\"undo\",\"of\":\"%s\",\"id\":%d}", u.op ? "del" : "add", u.id); st_log(s, l);
    st_save(s); pthread_mutex_unlock(&s->mu); return 0;
}
char *annot_snapshot(annot_store *s) { pthread_mutex_lock(&s->mu); char *t = st_snapshot_locked(s); pthread_mutex_unlock(&s->mu); return t; }
int annot_count(annot_store *s) { pthread_mutex_lock(&s->mu); int c = 0; for (int i = 0; i < s->n; i++) c += !s->it[i].dead; pthread_mutex_unlock(&s->mu); return c; }

/* ---- wrap / plane cut --------------------------------------------------------------------------------------------------------- */
static const int PA[3][2] = {{1, 2}, {0, 2}, {0, 1}};   /* the in-plane axes (rows, columns) of each plane */
static void seg_put(float **seg, size_t *cap, size_t *n, const float s4[4]) {
    if (*n + 4 > *cap) { *cap = (*n + 4) * 2 + 1024; *seg = realloc(*seg, *cap * sizeof **seg); }
    memcpy(*seg + *n, s4, 4 * sizeof *s4); *n += 4;
}
/* one triangle (3 vertices, zyx) against the plane */
static int tri_cut(const float *v[3], int axis, double pos, const double lo[2], const double hi[2], float **seg, size_t *cap, size_t *n) {
    double d[3]; for (int k = 0; k < 3; k++) d[k] = v[k][axis] - pos;
    float pt[4]; int m = 0; const int a0 = PA[axis][0], a1 = PA[axis][1];
    for (int e = 0; e < 3 && m < 4; e++) {
        const int i = e, j = (e + 1) % 3;
        if ((d[i] < 0) == (d[j] < 0)) continue;
        const double t = d[i] / (d[i] - d[j]);
        pt[m++] = (float)(v[i][a0] + t * (v[j][a0] - v[i][a0])); pt[m++] = (float)(v[i][a1] + t * (v[j][a1] - v[i][a1]));
    }
    if (m != 4) return 0;
    if (hi[0] > lo[0] && ((pt[0] < lo[0] && pt[2] < lo[0]) || (pt[0] >= hi[0] && pt[2] >= hi[0]))) return 0;
    if (hi[1] > lo[1] && ((pt[1] < lo[1] && pt[3] < lo[1]) || (pt[1] >= hi[1] && pt[3] >= hi[1]))) return 0;
    seg_put(seg, cap, n, pt); return 1;
}
size_t annot_cut_wrap(const float *zyx, int w, int h, int axis, double pos, const double lo[2], const double hi[2], float **seg, size_t *cap, size_t *n) {
    size_t c = 0;
    for (int r = 0; r + 1 < h; r++) {
        /* row band reject: the two vertex rows' range along the axis */
        float mn = 1e30f, mx = -1e30f;
        for (int q = 0; q < 2; q++) for (int i = 0; i < w; i++) { const float *p = zyx + 3 * ((size_t)(r + q) * w + i); if (p[0] < 0) continue; const float a = p[axis]; if (a < mn) mn = a; if (a > mx) mx = a; }
        if (mn > pos || mx < pos) continue;
        for (int i = 0; i + 1 < w; i++) {
            const float *p00 = zyx + 3 * ((size_t)r * w + i), *p01 = p00 + 3, *p10 = p00 + 3 * (size_t)w, *p11 = p10 + 3;
            if (p00[0] < 0 || p01[0] < 0 || p10[0] < 0 || p11[0] < 0) continue;
            const float *t1[3] = {p00, p01, p11}, *t2[3] = {p00, p11, p10};
            c += tri_cut(t1, axis, pos, lo, hi, seg, cap, n); c += tri_cut(t2, axis, pos, lo, hi, seg, cap, n);
        }
    }
    return c;
}

/* ---- server state ------------------------------------------------------------------------------------------------------------ */
typedef struct { int idx; int w, h; float *zyx; float lo[3], hi[3]; } wrap_t;
typedef struct { float z, y, x; char fam; } fpt;
typedef struct { char name[64]; double z, y, x, size, score; char note[160]; } region_t;
typedef struct { int lvl; int64_t cz, cy, cx; uint8_t *buf; int64_t n[3]; uint64_t tick; } chunk_t;
static struct {
    annot_cfg c; store *ct; z3 *lv[8]; int nlev; store *ls; z3 *lab;
    wrap_t *wr; atomic_int nwr, wloaded, wtotal; pthread_mutex_t wmu;
    fpt *fp; size_t nfp; region_t *rg; int nrg;
    annot_store *st;
    chunk_t *ch; int nch, chcap; uint64_t tick; size_t chbytes, chmax; pthread_mutex_t cmu;
    int lfd; pthread_t acc, wth; atomic_int stop;
    char *page; size_t pagen;
} G;

/* a decoded chunk of level l (or the label array when l < 0); nullptr on error. The caller holds no lock afterwards: entries are
   never freed while a request might use them (eviction frees only entries older than every request in flight: tick-based, the
   cache is sized for many slices and a request touches at most a few hundred chunks, so the oldest are safe in practice). */
static const chunk_t *get_chunk(int l, int64_t cz, int64_t cy, int64_t cx) {
    z3 *z = l < 0 ? G.lab : G.lv[l]; if (!z) return nullptr;
    pthread_mutex_lock(&G.cmu);
    for (int i = 0; i < G.nch; i++) { chunk_t *e = &G.ch[i]; if (e->lvl == l && e->cz == cz && e->cy == cy && e->cx == cx) { e->tick = ++G.tick; pthread_mutex_unlock(&G.cmu); return e; } }
    pthread_mutex_unlock(&G.cmu);
    const z3_meta *m = z3_meta_of(z); const int64_t cs[3] = {m->chunk[0], m->chunk[1], m->chunk[2]};
    int64_t o[3] = {cz * cs[0], cy * cs[1], cx * cs[2]}, n[3];
    for (int d = 0; d < 3; d++) { n[d] = cs[d]; if (o[d] + n[d] > m->shape[d]) n[d] = m->shape[d] - o[d]; if (n[d] <= 0) return nullptr; }
    uint8_t *b = malloc((size_t)(n[0] * n[1] * n[2]));
    if (!b || z3_read(z, o, n, b, 1)) { free(b); return nullptr; }
    pthread_mutex_lock(&G.cmu);
    for (int i = 0; i < G.nch; i++) { chunk_t *e = &G.ch[i]; if (e->lvl == l && e->cz == cz && e->cy == cy && e->cx == cx) { e->tick = ++G.tick; free(b); pthread_mutex_unlock(&G.cmu); return e; } }
    const size_t bytes = (size_t)(n[0] * n[1] * n[2]);
    while (G.chbytes + bytes > G.chmax && G.nch > 64) {   /* evict the least recently used */
        int k = 0; for (int i = 1; i < G.nch; i++) if (G.ch[i].tick < G.ch[k].tick) k = i;
        G.chbytes -= (size_t)(G.ch[k].n[0] * G.ch[k].n[1] * G.ch[k].n[2]); free(G.ch[k].buf); G.ch[k] = G.ch[--G.nch];
    }
    if (G.nch == G.chcap) { G.chcap = G.chcap ? 2 * G.chcap : 1024; G.ch = realloc(G.ch, (size_t)G.chcap * sizeof *G.ch); }
    /* (realloc moves entries: callers copy what they need before the next get_chunk, see slice_read) */
    chunk_t *e = &G.ch[G.nch++]; *e = (chunk_t){l, cz, cy, cx, b, {n[0], n[1], n[2]}, ++G.tick}; G.chbytes += bytes;
    pthread_mutex_unlock(&G.cmu); return e;
}
/* the plane (axis ax, index p at level l) region rows [r0, r0+nr) x cols [c0, c0+nc) -> out (0 outside the array) */
static int slice_read(int l, int ax, int64_t p, int64_t r0, int64_t c0, int nr, int nc, uint8_t *out) {
    z3 *z = l < 0 ? G.lab : G.lv[l]; if (!z) return -1;
    const z3_meta *m = z3_meta_of(z); memset(out, 0, (size_t)nr * nc);
    if (p < 0 || p >= m->shape[ax]) return 0;
    const int ra = PA[ax][0], ca = PA[ax][1]; const int64_t cs[3] = {m->chunk[0], m->chunk[1], m->chunk[2]};
    for (int64_t cr = (r0 < 0 ? 0 : r0) / cs[ra]; cr * cs[ra] < r0 + nr && cr * cs[ra] < m->shape[ra]; cr++)
        for (int64_t cc = (c0 < 0 ? 0 : c0) / cs[ca]; cc * cs[ca] < c0 + nc && cc * cs[ca] < m->shape[ca]; cc++) {
            int64_t ci[3]; ci[ax] = p / cs[ax]; ci[ra] = cr; ci[ca] = cc;
            const chunk_t *e0 = get_chunk(l, ci[0], ci[1], ci[2]); if (!e0) continue;
            pthread_mutex_lock(&G.cmu);   /* copy under the lock (the table may be reallocated by another request) */
            const chunk_t *e = nullptr; for (int i = 0; i < G.nch; i++) if (G.ch[i].lvl == l && G.ch[i].cz == ci[0] && G.ch[i].cy == ci[1] && G.ch[i].cx == ci[2]) { e = &G.ch[i]; break; }
            if (e) {
                const int64_t o[3] = {ci[0] * cs[0], ci[1] * cs[1], ci[2] * cs[2]}; const int64_t lp = p - o[ax];
                for (int64_t r = 0; r < e->n[ra]; r++) {
                    const int64_t orow = o[ra] + r - r0; if (orow < 0 || orow >= nr) continue;
                    for (int64_t c = 0; c < e->n[ca]; c++) {
                        const int64_t ocol = o[ca] + c - c0; if (ocol < 0 || ocol >= nc) continue;
                        int64_t q[3]; q[ax] = lp; q[ra] = r; q[ca] = c;
                        out[orow * nc + ocol] = e->buf[(q[0] * e->n[1] + q[1]) * e->n[2] + q[2]];
                    }
                }
            }
            pthread_mutex_unlock(&G.cmu);
        }
    return 0;
}

/* ---- loading wraps, fibres, regions ------------------------------------------------------------------------------------------- */
static int ends(const char *s, const char *t) { size_t a = strlen(s), b = strlen(t); return a >= b && !strcmp(s + a - b, t); }
static int cmpstr(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }
static void *wrap_loader(void *arg) {
    (void)arg; DIR *d = opendir(G.c.wraps); if (!d) { fprintf(stderr, "annot: cannot open wraps %s\n", G.c.wraps); return nullptr; }
    char **names = nullptr; int nn = 0, nc = 0; struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        char p[4200]; snprintf(p, sizeof p, "%s/%s", G.c.wraps, e->d_name); struct stat sbuf; if (stat(p, &sbuf)) continue;
        if (!(ends(e->d_name, ".sfc") || S_ISDIR(sbuf.st_mode))) continue;
        if (nn == nc) { nc = nc ? 2 * nc : 128; names = realloc(names, (size_t)nc * sizeof *names); }
        names[nn++] = strdup(e->d_name);
    }
    closedir(d); qsort(names, (size_t)nn, sizeof *names, cmpstr);
    G.wr = calloc((size_t)nn + 1, sizeof *G.wr);   /* never reallocated: requests read entries [0, nwr) without the lock */
    atomic_store(&G.wtotal, nn);
    const int s = G.c.wrap_stride > 0 ? G.c.wrap_stride : 2;
    for (int k = 0; k < nn && !atomic_load(&G.stop); k++) {
        char p[4300], q[4400]; snprintf(p, sizeof p, "%s/%s", G.c.wraps, names[k]);
        snprintf(q, sizeof q, "%s/meta.json", p); char *meta = tfx_meta_read(q);
        double vum = 0; int idx = -1;
        if (meta) { json *J = json_parse(meta, strlen(meta)); vum = json_num(json_get(J, "voxel_size_um"), 0); idx = (int)json_num(json_get(J, "winding"), -1); json_free(J); free(meta); }
        if (idx < 0) { const char *t = names[k]; while (*t && (*t < '0' || *t > '9')) t++; idx = *t ? atoi(t) : k; }
        if (!(vum > 0)) vum = G.c.um0 * ldexp(1.0, G.c.base);
        int w, h, w2, h2, w3, h3; snprintf(q, sizeof q, "%s/x.tif", p); float *X = img_read_f32(q, &w, &h);
        snprintf(q, sizeof q, "%s/y.tif", p); float *Y = img_read_f32(q, &w2, &h2); snprintf(q, sizeof q, "%s/z.tif", p); float *Z = img_read_f32(q, &w3, &h3);
        if (!X || !Y || !Z || w2 != w || w3 != w || h2 != h || h3 != h) { free(X); free(Y); free(Z); atomic_fetch_add(&G.wloaded, 1); continue; }
        const int ws = (w + s - 1) / s, hs = (h + s - 1) / s; const double f = vum / G.c.um0;
        wrap_t W = {idx, ws, hs, malloc((size_t)ws * hs * 3 * sizeof(float)), {1e30f, 1e30f, 1e30f}, {-1e30f, -1e30f, -1e30f}};
        for (int r = 0; r < hs; r++) for (int i = 0; i < ws; i++) {
            const size_t a = (size_t)(r * s) * w + (size_t)(i * s); float *o = W.zyx + 3 * ((size_t)r * ws + i);
            if (X[a] < 0 || Y[a] < 0 || Z[a] < 0) { o[0] = o[1] = o[2] = -1; continue; }
            o[0] = (float)(Z[a] * f); o[1] = (float)(Y[a] * f); o[2] = (float)(X[a] * f);
            for (int dd = 0; dd < 3; dd++) { if (o[dd] < W.lo[dd]) W.lo[dd] = o[dd]; if (o[dd] > W.hi[dd]) W.hi[dd] = o[dd]; }
        }
        free(X); free(Y); free(Z);
        G.wr[atomic_load(&G.nwr)] = W; atomic_fetch_add(&G.nwr, 1);   /* (the entry is complete before the count publishes it) */
        atomic_fetch_add(&G.wloaded, 1);
    }
    for (int k = 0; k < nn; k++) free(names[k]); free(names);
    fprintf(stderr, "annot: %d wraps loaded from %s\n", atomic_load(&G.nwr), G.c.wraps);
    return nullptr;
}
static int load_ufib(const char *path) {   /* "UFIB" int32 1, n, npts, off[n+1], fam[n], float32 pts[3 npts] (z y x) */
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    char m[4]; int32_t h[3]; if (fread(m, 1, 4, f) != 4 || memcmp(m, "UFIB", 4) || fread(h, 4, 3, f) != 3 || h[0] != 1) { fclose(f); return -1; }
    int32_t *off = malloc((size_t)(h[1] + 1) * 4); char *fam = malloc((size_t)h[1] + 1); float *p = malloc((size_t)h[2] * 12 + 1);
    if (fread(off, 4, (size_t)h[1] + 1, f) != (size_t)h[1] + 1 || fread(fam, 1, (size_t)h[1], f) != (size_t)h[1] || fread(p, 12, (size_t)h[2], f) != (size_t)h[2]) { fclose(f); free(off); free(fam); free(p); return -1; }
    fclose(f); G.fp = malloc((size_t)h[2] * sizeof *G.fp + 1); G.nfp = 0;
    for (int r = 0; r < h[1]; r++) for (int k = off[r]; k < off[r + 1] && k < h[2]; k++) G.fp[G.nfp++] = (fpt){p[3 * k], p[3 * k + 1], p[3 * k + 2], fam[r]};
    free(off); free(fam); free(p); return 0;
}
static int load_regions(const char *path) {   /* name z y x size score note (level-0) */
    FILE *f = fopen(path, "r"); if (!f) return -1; char line[1024];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        region_t r = {0}; char nm[64] = "", note[160] = "";
        const int k = sscanf(line, "%63s %lf %lf %lf %lf %lf %159[^\n]", nm, &r.z, &r.y, &r.x, &r.size, &r.score, note);
        if (k < 5) continue;
        snprintf(r.name, sizeof r.name, "%s", nm); snprintf(r.note, sizeof r.note, "%s", note);
        G.rg = realloc(G.rg, (size_t)(G.nrg + 1) * sizeof *G.rg); G.rg[G.nrg++] = r;
    }
    fclose(f); return 0;
}

/* ---- HTTP ---------------------------------------------------------------------------------------------------------------------- */
#if defined(__has_embed)
static const char PAGE_EMBED[] = {
#embed "annot_page.html"
, 0};
#else
static const char PAGE_EMBED[] = "<html><body>this build has no embedded page (needs #embed, gcc >= 15): run with --page FILE</body></html>";
#endif

static void reply(int fd, int code, const char *type, const void *body, size_t n) {
    char h[512]; const int k = snprintf(h, sizeof h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
                                        code, code == 200 ? "OK" : code == 404 ? "Not Found" : "Bad Request", type, n);
    ssize_t w = write(fd, h, (size_t)k); (void)w;
    for (size_t o = 0; o < n;) { ssize_t r = write(fd, (const char *)body + o, n - o); if (r <= 0) break; o += (size_t)r; }
}
static void reply_json(int fd, int code, const char *t) { reply(fd, code, "application/json", t, strlen(t)); }
static const char *qget(const char *q, const char *key, char *buf, size_t n) {
    const size_t kl = strlen(key);
    for (const char *p = q; p && *p;) {
        if (!strncmp(p, key, kl) && p[kl] == '=') { const char *v = p + kl + 1; size_t i = 0; while (v[i] && v[i] != '&' && i + 1 < n) { buf[i] = v[i]; i++; } buf[i] = 0; return buf; }
        p = strchr(p, '&'); if (p) p++;
    }
    return nullptr;
}
static double qnum(const char *q, const char *key, double d) { char b[64]; const char *v = qget(q, key, b, sizeof b); return v ? atof(v) : d; }
static void handle(int fd) {
    char *req = malloc(65536); size_t n = 0; char *hend = nullptr;
    while (n < 65535) { ssize_t r = read(fd, req + n, 65535 - n); if (r <= 0) break; n += (size_t)r; req[n] = 0; if ((hend = strstr(req, "\r\n\r\n"))) break; }
    if (!hend) { free(req); return; }
    char method[8] = "", url[2048] = ""; sscanf(req, "%7s %2047s", method, url);
    size_t clen = 0; { const char *c = strcasestr(req, "\r\nContent-Length:"); if (c) clen = (size_t)atol(c + 17); }
    if (clen > (1 << 20)) { reply_json(fd, 400, "{\"error\":\"body too large\"}"); free(req); return; }
    const size_t have = n - (size_t)(hend + 4 - req); char *body = malloc(clen + 1); memcpy(body, hend + 4, have < clen ? have : clen);
    for (size_t o = have; o < clen;) { ssize_t r = read(fd, body + o, clen - o); if (r <= 0) { clen = o; break; } o += (size_t)r; }
    body[clen] = 0; free(req);
    char *q = strchr(url, '?'); if (q) *q++ = 0; const char *Q = q ? q : "";
    const int post = !strcmp(method, "POST");
    if (!strcmp(url, "/")) reply(fd, 200, "text/html; charset=utf-8", G.page, G.pagen);
    else if (!strcmp(url, "/info")) {
        sb b = {0}; sb_str(&b, "{\"scroll\":"); sb_jstr(&b, G.c.scroll ? G.c.scroll : ""); sb_fmt(&b, ",\"um0\":%.6g,\"base\":%d,\"levels\":[", G.c.um0, G.c.base);
        int first = 1; for (int l = 0; l < 8; l++) if (G.lv[l]) { const z3_meta *m = z3_meta_of(G.lv[l]); sb_fmt(&b, "%s{\"l\":%d,\"shape\":[%lld,%lld,%lld]}", first ? "" : ",", l, (long long)m->shape[0], (long long)m->shape[1], (long long)m->shape[2]); first = 0; }
        sb_fmt(&b, "],\"wraps\":{\"total\":%d,\"loaded\":%d,\"n\":%d},\"fibres\":%zu,\"labels\":%d,\"regions\":%d,\"items\":%d,\"author\":", atomic_load(&G.wtotal), atomic_load(&G.wloaded), atomic_load(&G.nwr), G.nfp, G.lab != nullptr, G.nrg, annot_count(G.st));
        sb_jstr(&b, G.c.author ? G.c.author : ""); sb_str(&b, ",\"store\":"); sb_jstr(&b, G.c.store); sb_str(&b, "}"); reply_json(fd, 200, b.s); free(b.s);
    } else if (!strcmp(url, "/slice") || !strcmp(url, "/lab")) {
        const int lab = !strcmp(url, "/lab"); const int l = (int)qnum(Q, "l", G.c.base), ax = (int)qnum(Q, "ax", 0);
        const int nr = (int)qnum(Q, "nr", 512), nc = (int)qnum(Q, "nc", 512);
        if (ax < 0 || ax > 2 || l < 0 || l > 7 || (!lab && !G.lv[l]) || (lab && (!G.lab || l != G.c.base)) || nr <= 0 || nc <= 0 || nr > 4096 || nc > 4096) { reply_json(fd, 404, "{\"error\":\"no such slice\"}"); free(body); return; }
        const int64_t p = (int64_t)floor(qnum(Q, "pos", 0) / ldexp(1.0, l));
        uint8_t *o = malloc((size_t)nr * nc); slice_read(lab ? -1 : l, ax, p, (int64_t)qnum(Q, "r0", 0), (int64_t)qnum(Q, "c0", 0), nr, nc, o);
        reply(fd, 200, "application/octet-stream", o, (size_t)nr * nc); free(o);
    } else if (!strcmp(url, "/wraps")) {
        const int ax = (int)qnum(Q, "ax", 0); const double pos = qnum(Q, "pos", 0);
        const double lo[2] = {qnum(Q, "r0", 0), qnum(Q, "c0", 0)}, hi[2] = {qnum(Q, "r1", 0), qnum(Q, "c1", 0)};
        sb b = {0}; sb_fmt(&b, "{\"loaded\":%d,\"total\":%d,\"wraps\":[", atomic_load(&G.wloaded), atomic_load(&G.wtotal)); int first = 1;
        const int nw = atomic_load(&G.nwr); const wrap_t *W = G.wr;
        float *seg = nullptr; size_t cap = 0;
        for (int k = 0; k < nw && ax >= 0 && ax <= 2; k++) {
            if (W[k].lo[ax] > pos || W[k].hi[ax] < pos) continue;
            size_t ns = 0; annot_cut_wrap(W[k].zyx, W[k].w, W[k].h, ax, pos, lo, hi, &seg, &cap, &ns); if (!ns) continue;
            sb_fmt(&b, "%s{\"w\":%d,\"s\":[", first ? "" : ",", W[k].idx); first = 0;
            for (size_t i = 0; i < ns; i++) sb_fmt(&b, "%s%.1f", i ? "," : "", seg[i]);
            sb_str(&b, "]}");
        }
        free(seg); sb_str(&b, "]}"); reply_json(fd, 200, b.s); free(b.s);
    } else if (!strcmp(url, "/fib")) {
        const int ax = (int)qnum(Q, "ax", 0); const double pos = qnum(Q, "pos", 0), tol = qnum(Q, "tol", 2);
        const double r0 = qnum(Q, "r0", 0), c0 = qnum(Q, "c0", 0), r1 = qnum(Q, "r1", 1e30), c1 = qnum(Q, "c1", 1e30);
        sb h = {0}, v = {0}, u = {0}; const int ra = PA[ax < 0 || ax > 2 ? 0 : ax][0], ca = PA[ax < 0 || ax > 2 ? 0 : ax][1];
        for (size_t i = 0; i < G.nfp && ax >= 0 && ax <= 2; i++) {
            const float c[3] = {G.fp[i].z, G.fp[i].y, G.fp[i].x};
            if (fabs(c[ax] - pos) > tol || c[ra] < r0 || c[ra] >= r1 || c[ca] < c0 || c[ca] >= c1) continue;
            sb *t = G.fp[i].fam == 'H' ? &h : G.fp[i].fam == 'V' ? &v : &u; sb_fmt(t, "%s%.1f,%.1f", t->n ? "," : "", c[ra], c[ca]);
        }
        sb b = {0}; sb_str(&b, "{\"H\":["); if (h.s) sb_str(&b, h.s); sb_str(&b, "],\"V\":["); if (v.s) sb_str(&b, v.s); sb_str(&b, "],\"O\":["); if (u.s) sb_str(&b, u.s); sb_str(&b, "]}");
        reply_json(fd, 200, b.s); free(b.s); free(h.s); free(v.s); free(u.s);
    } else if (!strcmp(url, "/ann") && !post) { char *t = annot_snapshot(G.st); reply_json(fd, 200, t); free(t); }
    else if (!strcmp(url, "/ann/add") && post) {
        char err[256] = ""; const int id = annot_add(G.st, body, G.c.author, err, sizeof err);
        if (id < 0) { sb b = {0}; sb_str(&b, "{\"error\":"); sb_jstr(&b, err); sb_str(&b, "}"); reply_json(fd, 400, b.s); free(b.s); }
        else { char t[64]; snprintf(t, sizeof t, "{\"id\":%d}", id); reply_json(fd, 200, t); }
    } else if (!strcmp(url, "/ann/del") && post) reply_json(fd, annot_del(G.st, (int)qnum(Q, "id", 0)) ? 404 : 200, "{}");
    else if (!strcmp(url, "/ann/undo") && post) reply_json(fd, annot_undo(G.st) ? 404 : 200, "{}");
    else if (!strcmp(url, "/regions")) {
        sb b = {0}; sb_str(&b, "[");
        for (int i = 0; i < G.nrg; i++) { const region_t *r = &G.rg[i]; sb_fmt(&b, "%s{\"name\":", i ? "," : ""); sb_jstr(&b, r->name);
            sb_fmt(&b, ",\"z\":%.0f,\"y\":%.0f,\"x\":%.0f,\"size\":%.0f,\"score\":%.6g,\"note\":", r->z, r->y, r->x, r->size, r->score); sb_jstr(&b, r->note); sb_str(&b, "}"); }
        sb_str(&b, "]"); reply_json(fd, 200, b.s); free(b.s);
    } else reply_json(fd, 404, "{\"error\":\"not found\"}");
    free(body);
}
static atomic_int g_live;
static void *conn_thread(void *arg) { const int fd = (int)(intptr_t)arg; handle(fd); close(fd); atomic_fetch_sub(&g_live, 1); return nullptr; }
static void *accept_thread(void *arg) {
    (void)arg;
    while (!atomic_load(&G.stop)) {
        const int fd = accept(G.lfd, nullptr, nullptr); if (fd < 0) { if (atomic_load(&G.stop)) break; continue; }
        while (atomic_load(&g_live) >= (G.c.threads > 0 ? G.c.threads : 4) * 4) usleep(1000);
        atomic_fetch_add(&g_live, 1); pthread_t t; pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
        if (pthread_create(&t, &a, conn_thread, (void *)(intptr_t)fd)) { close(fd); atomic_fetch_sub(&g_live, 1); }
        pthread_attr_destroy(&a);
    }
    return nullptr;
}
int annot_serve_start(const annot_cfg *c) {
    memset(&G, 0, sizeof G); G.c = *c; pthread_mutex_init(&G.cmu, nullptr); pthread_mutex_init(&G.wmu, nullptr);
    G.chmax = (size_t)((c->cache_gb > 0 ? c->cache_gb : 2.0) * 1e9);
    G.ct = store_open(c->ct_root); if (!G.ct) { fprintf(stderr, "annot: cannot open %s\n", c->ct_root); return -1; }
    for (int l = 0; l < 8; l++) { char k[2200]; snprintf(k, sizeof k, "%s/%d", c->ct_key, l); if (l < c->base) continue;
        char zj[2300]; snprintf(zj, sizeof zj, "%s/zarr.json", k); if (store_size(G.ct, zj) <= 0) continue;
        G.lv[l] = z3_open(G.ct, k, c->cache); if (G.lv[l]) G.nlev++; }
    if (!G.lv[c->base]) { fprintf(stderr, "annot: %s/%s has no level %d\n", c->ct_root, c->ct_key, c->base); return -1; }
    if (c->lab_root && c->lab_key) { G.ls = store_open(c->lab_root); G.lab = G.ls ? z3_open(G.ls, c->lab_key, nullptr) : nullptr;
        if (!G.lab) fprintf(stderr, "annot: labels %s %s not opened (overlay off)\n", c->lab_root, c->lab_key);
        else { const z3_meta *a = z3_meta_of(G.lab), *b = z3_meta_of(G.lv[c->base]);
            if (a->shape[0] != b->shape[0] || a->shape[1] != b->shape[1] || a->shape[2] != b->shape[2]) { fprintf(stderr, "annot: the label array's shape differs from CT level %d: overlay off\n", c->base); z3_close(G.lab); G.lab = nullptr; } } }
    if (c->fibers && load_ufib(c->fibers)) fprintf(stderr, "annot: cannot read fibres %s (overlay off)\n", c->fibers);
    if (c->regions && load_regions(c->regions)) fprintf(stderr, "annot: cannot read regions %s\n", c->regions);
    G.st = annot_open(c->store, c->scroll, c->ct_key); if (!G.st) return -1;
    if (c->page) { FILE *f = fopen(c->page, "rb"); if (f) { fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); G.page = malloc((size_t)n + 1);
        G.pagen = fread(G.page, 1, (size_t)n, f); fclose(f); } }
    if (!G.page) { G.pagen = strlen(PAGE_EMBED); G.page = malloc(G.pagen + 1); memcpy(G.page, PAGE_EMBED, G.pagen + 1); }
    G.lfd = socket(AF_INET, SOCK_STREAM, 0); const int one = 1; setsockopt(G.lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons((uint16_t)c->port)}; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(G.lfd, (struct sockaddr *)&a, sizeof a) || listen(G.lfd, 64)) { fprintf(stderr, "annot: cannot listen on 127.0.0.1:%d: %s\n", c->port, strerror(errno)); return -1; }
    socklen_t al = sizeof a; getsockname(G.lfd, (struct sockaddr *)&a, &al);
    if (c->wraps) pthread_create(&G.wth, nullptr, wrap_loader, nullptr);
    pthread_create(&G.acc, nullptr, accept_thread, nullptr);
    return ntohs(a.sin_port);
}
void annot_serve_stop(void) {
    atomic_store(&G.stop, 1); shutdown(G.lfd, SHUT_RDWR); close(G.lfd); pthread_join(G.acc, nullptr);
    if (G.c.wraps) pthread_join(G.wth, nullptr);
    while (atomic_load(&g_live) > 0) usleep(1000);
    for (int i = 0; i < G.nch; i++) free(G.ch[i].buf); free(G.ch);
    for (int k = 0; k < atomic_load(&G.nwr); k++) free(G.wr[k].zyx); free(G.wr); free(G.fp); free(G.rg); free(G.page);
    for (int l = 0; l < 8; l++) if (G.lv[l]) z3_close(G.lv[l]);
    if (G.lab) z3_close(G.lab); if (G.ls) store_close(G.ls); store_close(G.ct); annot_close(G.st);
}

static const char *aopt(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
int cmd_annotate_serve(int argc, char **argv) {
    if (argc < 4 || !*aopt(argc, argv, "--store", "")) {
        fprintf(stderr, "usage: ufsm annotate-serve <ct-root> <ct-key> --um U0 --store ANN.json [--level L (finest shown, 2)] [--port 8765]\n"
                        "       [--wraps DIR] [--wrap-stride 2] [--labels ROOT KEY (phase / codes on level L's grid)] [--fibers F.fib (UFIB level-0)]\n"
                        "       [--regions REGIONS.tsv (annot-regions)] [--scroll NAME] [--author NAME] [--cache DIR] [--cache-gb 2] [--threads 4] [--page FILE]\n"
                        "  a local viewer (127.0.0.1 only) of CT slices with wraps / labels / fibres and a manual winding annotation store\n");
        return 2;
    }
    annot_cfg c = {.ct_root = argv[2], .ct_key = argv[3], .um0 = atof(aopt(argc, argv, "--um", "0")), .base = atoi(aopt(argc, argv, "--level", "2")),
                   .wraps = aopt(argc, argv, "--wraps", nullptr), .wrap_stride = atoi(aopt(argc, argv, "--wrap-stride", "2")), .fibers = aopt(argc, argv, "--fibers", nullptr),
                   .store = aopt(argc, argv, "--store", ""), .regions = aopt(argc, argv, "--regions", nullptr), .scroll = aopt(argc, argv, "--scroll", nullptr),
                   .author = aopt(argc, argv, "--author", getenv("USER") ? getenv("USER") : "user"), .page = aopt(argc, argv, "--page", nullptr),
                   .cache = aopt(argc, argv, "--cache", nullptr), .port = atoi(aopt(argc, argv, "--port", "8765")), .threads = atoi(aopt(argc, argv, "--threads", "4")),
                   .cache_gb = atof(aopt(argc, argv, "--cache-gb", "2"))};
    for (int i = 2; i + 2 < argc; i++) if (!strcmp(argv[i], "--labels")) { c.lab_root = argv[i + 1]; c.lab_key = argv[i + 2]; }
    if (!(c.um0 > 0) || c.base < 0 || c.base > 7) { fprintf(stderr, "annotate-serve: --um U0 (> 0) and --level 0..7\n"); return 2; }
    const int port = annot_serve_start(&c); if (port < 0) return 1;
    printf("annotate-serve: http://127.0.0.1:%d/  (store %s, %d items)\n", port, c.store, annot_count(G.st)); fflush(stdout);
    for (;;) pause();
}
