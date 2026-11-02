/* ufsm patch-id WRAPS OUT --axis F --axis-um U0 --dir +-1 --sheet ROOT KEY --grid-um U
                 [--thr 1] [--zr Z0,Z1] [--block 32] [--band 64] [--min-comp 8] [--range 3] [--min-link 16] [--ratio 0.3]
                 [--min-votes 500] [--min-share 0.5] [--cells 32,32] [--gt ROOT KEY [--gt-range 4]] [--threads N]
   Sheet identity from physically connected sheet patches (TRACE, 11-01). A sheet prediction (--sheet: uint8 on the grid, voxels
   >= --thr are sheet, e.g. the published surface-m7 mask at 9.6 um) is cut into blocks of --block voxels; each block's 6-connected
   components (>= --min-comp voxels; blocks within 2 blocks of the axis skipped) are the atoms. Atoms of neighbouring blocks are
   joined across the shared face only when they are each other's best partner on it (>= --min-link touching voxel pairs, the
   second-best partner of either under --ratio of that): a component bridging two sheets has two partners on the far side and stays
   cut, so merges do not chain through touching sheets. The joins form a union-find with turn offsets (each atom unwraps theta
   around its block's centre angle; crossing theta = +-pi adds a turn), so a patch carries one continuous sheet angle; a join that
   would close a loop with a non-zero turn (a patch around the axis = a merge of two sheets) is refused and counted.
   Every wrap vertex (wNNN.sfc, q = k + dir theta / 2 pi as surf-labels) looks along its horizontal radial ray for the nearest sheet
   voxel within --range grid voxels (two different atoms at the same distance: none) and votes its wrap id, unwrapped into the patch's
   angle frame, J = k - dir m (theta_patch = theta + 2 pi m). A patch with >= --min-votes votes takes the majority J*, if it holds
   >= --min-share of them; each vote then gets the correction dk = J* - J (its wrap is dk turns off on this sheet). Per cell of
   --cells R,S (R wrap rows x 1/S turn, sector = floor((theta / 2 pi + 0.5) S): the V2LABEL anchor grid) the votes' modal dk.
   Outputs: OUT/cells.tsv (wrap row0 sector patch dk conf brk n npatch: patch = the cell's main patch id, conf = the modal dk's share
   of the patched votes, brk 1 = the cell's patched votes disagree (conf < 0.8) or come from >1 patch), OUT/wraps.tsv per wrap.
   --gt ROOT KEY (winding_mod14 codes on the grid, e.g. P4 q-9p6): each vertex also takes the nearest GT code along its ray within
   --gt-range voxels, e = q_gt - q (integer turns + one global gauge): the share on the right sheet before / after the per-vote and
   per-cell corrections, the patches' GT purity (the share of a patch's votes on its modal GT sheet: the ceiling of any patch-level
   relabelling), overall and per wrap. Bands of --band grid z rows (a multiple of --block); --zr limits the grid z range. */
#include "pairacc.h"
#include "sources.h"
#include "store.h"
#include "surfcomp.h"
#include "sysinfo.h"
#include "zarr3.h"
#include <dirent.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define PI_NONE UINT32_MAX
#define PI_BAD 0xFFFF   /* lab value of a rejected atom */
enum { PI_DKR = 8 };   /* corrections kept in [-PI_DKR, PI_DKR] */

static const char *pi_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static int pi_idx(int argc, char **argv, const char *k) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], k)) return i; return -1; }
static double pi_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static inline double pi_ang(double a) { a = fmod(a + M_PI, 2 * M_PI); if (a <= 0) a += 2 * M_PI; return a - M_PI; }   /* -> (-pi, pi] */

typedef struct {
    int id, w, h; char path[1400]; sfc_reader *r; sfc_cache *ca; uint32_t ch; double um;
    float *rz0, *rz1; long long nv;
    int b0, bn; float *g; uint8_t *v;   /* band rows: grid z, y, x, theta per vertex */
} pi_wrap;

typedef struct { uint32_t cell, comp; int16_t jv; uint8_t eb, pad; uint32_t n; } pi_rec;   /* eb: GT turn offset 0..13, 255 none */
typedef struct { pi_rec *r; size_t n, cap; } pi_recs;
typedef struct { float p[3]; uint32_t cell, comp; int16_t jv; uint8_t eb, gc; } pi_smp;   /* a pair-metric sample (GT-coded vertex), p in mm */
typedef struct { pi_smp *s; size_t n, cap; } pi_smps;
static inline uint64_t pi_mix(uint64_t x) { x += 0x9e3779b97f4a7c15ull; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31); }
static void pi_push(pi_recs *R, pi_rec x) { if (R->n == R->cap) { R->cap = R->cap ? 2 * R->cap : 1 << 16; R->r = realloc(R->r, R->cap * sizeof *R->r); if (!R->r) abort(); } R->r[R->n++] = x; }
static int pi_cmp_rec(const void *a, const void *b) {
    const pi_rec *x = a, *y = b;
    if (x->cell != y->cell) return x->cell < y->cell ? -1 : 1;
    if (x->comp != y->comp) return x->comp < y->comp ? -1 : 1;
    if (x->jv != y->jv) return x->jv < y->jv ? -1 : 1;
    return (x->eb > y->eb) - (x->eb < y->eb);
}
static void pi_reduce(pi_recs *R) {   /* sort + merge equal keys */
    if (!R->n) return;
    qsort(R->r, R->n, sizeof *R->r, pi_cmp_rec);
    size_t o = 0;
    for (size_t i = 1; i < R->n; i++) {
        pi_rec *a = &R->r[o], *b = &R->r[i];
        if (a->cell == b->cell && a->comp == b->comp && a->jv == b->jv && a->eb == b->eb) a->n += b->n; else R->r[++o] = *b;
    }
    R->n = o + 1;
}

/* union-find with turn offsets: theta_parent = theta_c + 2 pi off[c]. Every set is kept a height field: per cell (z block, angle bin
   of 1/NA turn in the root's unwrapped frame) the radius range of its atoms' centroids; a join that would put two atoms more than
   tol voxels apart in radius into one cell (two sheets stacked on one ray: a merge) is refused. Small-to-large over atom lists. */
typedef struct { uint64_t *k; float *lo, *hi; size_t cap, used; } pi_hash;   /* key 0 empty, 1 tombstone */
static inline uint64_t pi_hkey(uint32_t root, int32_t zb, int32_t ab) { return ((uint64_t)(root + 1) << 32) | ((uint64_t)(zb & 0xFFF) << 20) | (uint64_t)(ab & 0xFFFFF); }
static size_t pi_hslot(const pi_hash *H, uint64_t key, int *found) {   /* the key's slot, or the first free / tombstone slot */
    size_t i = (size_t)(pi_mix(key) & (H->cap - 1)), tomb = SIZE_MAX;
    for (;;) {
        const uint64_t k = H->k[i];
        if (k == key) { *found = 1; return i; }
        if (k == 0) { *found = 0; return tomb != SIZE_MAX ? tomb : i; }
        if (k == 1 && tomb == SIZE_MAX) tomb = i;
        i = (i + 1) & (H->cap - 1);
    }
}
static void pi_hput(pi_hash *H, uint64_t key, float r);
static void pi_hgrow(pi_hash *H) {
    pi_hash N = {calloc(H->cap * 2, 8), malloc(H->cap * 2 * 4), malloc(H->cap * 2 * 4), H->cap * 2, 0};
    if (!N.k || !N.lo || !N.hi) { fprintf(stderr, "patch-id: no memory for the height-field table\n"); exit(1); }
    for (size_t i = 0; i < H->cap; i++) if (H->k[i] > 1) { int f; const size_t j = pi_hslot(&N, H->k[i], &f); N.k[j] = H->k[i]; N.lo[j] = H->lo[i]; N.hi[j] = H->hi[i]; N.used++; }
    free(H->k); free(H->lo); free(H->hi); *H = N;
}
static void pi_hput(pi_hash *H, uint64_t key, float r) {
    if (2 * (H->used + 1) > H->cap) pi_hgrow(H);
    int f; const size_t i = pi_hslot(H, key, &f);
    if (f) { if (r < H->lo[i]) H->lo[i] = r; if (r > H->hi[i]) H->hi[i] = r; return; }
    if (H->k[i] == 0) H->used++;
    H->k[i] = key; H->lo[i] = H->hi[i] = r;
}
static void pi_hdel(pi_hash *H, uint64_t key) { int f; const size_t i = pi_hslot(H, key, &f); if (f) H->k[i] = 1; }
static int pi_hget(const pi_hash *H, uint64_t key, float *lo, float *hi) { int f; const size_t i = pi_hslot(H, key, &f); if (f) { *lo = H->lo[i]; *hi = H->hi[i]; } return f; }

typedef struct { uint32_t *par; int16_t *off; uint32_t *sz; int32_t *zb, *ab; float *r; uint32_t *nxt, *tail; size_t n, cap; long long merges, refused, cycles; pi_hash H; int NA; float tol; } pi_uf;
static void pi_uf_grow(pi_uf *U, size_t n) {
    if (n <= U->cap) return;
    size_t c = U->cap ? U->cap : 1 << 20; while (c < n) c *= 2;
    U->par = realloc(U->par, c * 4); U->off = realloc(U->off, c * 2); U->sz = realloc(U->sz, c * 4);
    U->zb = realloc(U->zb, c * 4); U->ab = realloc(U->ab, c * 4); U->r = realloc(U->r, c * 4); U->nxt = realloc(U->nxt, c * 4); U->tail = realloc(U->tail, c * 4);
    if (!U->par || !U->off || !U->sz || !U->zb || !U->ab || !U->r || !U->nxt || !U->tail) { fprintf(stderr, "patch-id: no memory for %zu atoms\n", c); exit(1); }
    U->cap = c;
    if (!U->H.cap) { U->H.cap = 1 << 22; U->H.k = calloc(U->H.cap, 8); U->H.lo = malloc(U->H.cap * 4); U->H.hi = malloc(U->H.cap * 4); }
}
static uint32_t pi_find(pi_uf *U, uint32_t c, int *T) {
    uint32_t r = c; int t = 0;
    while (U->par[r] != r) { t += U->off[r]; r = U->par[r]; }
    /* compress: every node on the path points at r with its total offset */
    uint32_t x = c; int tx = t;
    while (U->par[x] != x) { const uint32_t p = U->par[x]; const int o = U->off[x]; U->par[x] = r; U->off[x] = (int16_t)tx; tx -= o; x = p; }
    *T = t; return r;
}
/* a new atom (its own set; zb / ab / r set) */
static void pi_atom(pi_uf *U, uint32_t a) { U->par[a] = a; U->off[a] = 0; U->sz[a] = 1; U->nxt[a] = PI_NONE; U->tail[a] = a; if (U->r[a] >= 0) pi_hput(&U->H, pi_hkey(a, U->zb[a], U->ab[a]), U->r[a]); }
/* join a and b with theta_a = theta_b + 2 pi t */
static void pi_join(pi_uf *U, uint32_t a, uint32_t b, int t) {
    int Ta, Tb; uint32_t ra = pi_find(U, a, &Ta), rb = pi_find(U, b, &Tb);
    if (ra == rb) { if (Tb - t != Ta) U->cycles++; return; }
    /* theta_ra = theta_a + 2pi Ta = theta_b + 2pi (t + Ta); theta_rb = theta_b + 2pi Tb -> theta_ra = theta_rb + 2pi (t + Ta - Tb) */
    int o = t + Ta - Tb;
    if (U->sz[ra] < U->sz[rb]) { const uint32_t x = ra; ra = rb; rb = x; o = -o; }   /* ra large: theta_ra = theta_rb + 2 pi o */
    if (U->tol > 0) {
        for (uint32_t x = rb; x != PI_NONE; x = U->nxt[x]) {
            if (U->r[x] < 0) continue;
            int Tx; pi_find(U, x, &Tx); float lo, hi;
            if (pi_hget(&U->H, pi_hkey(ra, U->zb[x], U->ab[x] + (Tx + o) * U->NA), &lo, &hi) && (U->r[x] < lo - U->tol || U->r[x] > hi + U->tol)) { U->refused++; return; }
        }
        for (uint32_t x = rb; x != PI_NONE; x = U->nxt[x]) {
            if (U->r[x] < 0) continue;
            int Tx; pi_find(U, x, &Tx);
            pi_hdel(&U->H, pi_hkey(rb, U->zb[x], U->ab[x] + Tx * U->NA));
            pi_hput(&U->H, pi_hkey(ra, U->zb[x], U->ab[x] + (Tx + o) * U->NA), U->r[x]);
        }
    }
    U->par[rb] = ra; U->off[rb] = (int16_t)o; U->sz[ra] += U->sz[rb];
    U->nxt[U->tail[ra]] = rb; U->tail[ra] = U->tail[rb];
    U->merges++;
}

typedef struct { uint32_t a, b; int t; uint32_t n; } pi_link;
static int pi_cmp_link(const void *x, const void *y) { const pi_link *a = x, *b = y; if (a->n != b->n) return a->n > b->n ? -1 : 1; if (a->a != b->a) return a->a < b->a ? -1 : 1; return (a->b > b->b) - (a->b < b->b); }
typedef struct { pi_link *l; size_t n, cap; } pi_links;
static void pi_lpush(pi_links *L, pi_link x) { if (L->n == L->cap) { L->cap = L->cap ? 2 * L->cap : 4096; L->l = realloc(L->l, L->cap * sizeof *L->l); if (!L->l) abort(); } L->l[L->n++] = x; }

typedef struct {
    pi_wrap *W; int nw; axis *ax; double s0, f, dir;
    int B, BAND, nby, nbx, nbz; int64_t o[3];   /* band origin (z, y, x) in grid voxels; band extent BAND x nby B x nbx B */
    int64_t NY, NX; uint8_t thr; int min_comp, range, gt_range, min_link, bad_gap; double ratio; _Atomic long long nbad;
    uint8_t *sheet, *gt; uint16_t *lab; uint32_t *base; uint16_t *ncomp; float *phi;   /* per band block: atom count, centre angle (NAN: skipped) */
    uint32_t *prev; float *prev_phi; int have_prev;   /* the previous band's last z plane: global atom ids (PI_NONE) and its block angles */
    int CR, NS, *nrb; uint32_t *cbase;   /* cells: per wrap row blocks, cell index base */
    _Atomic long next; long nitems; _Atomic int fail, slot; int nt;   /* slot: the pass's worker index (reset per pass) */
    pi_uf *U; pi_recs *trec; pi_links *tlink; pi_smps *tsmp; uint32_t smp_T; double mm;   /* per thread; sample threshold; grid voxel in mm */
    _Atomic long long nvert, nhit, namb, ngt; double *gre, *gim;   /* per thread circular sums of e */
} pi_job;

/* ---- wraps (as surf-labels) ---- */
static int pi_cmp_id(const void *a, const void *b) { const pi_wrap *x = a, *y = b; return (x->id > y->id) - (x->id < y->id); }
static int pi_open(pi_wrap *w) {
    if (sfc_open_file(w->path, &w->r)) { fprintf(stderr, "patch-id: cannot open %s\n", w->path); return -1; }
    int c = sfc_find_channel(w->r, "x"); if (c < 0) c = 0; w->ch = (uint32_t)c;
    const sfc_channel *ch = sfc_channel_info(w->r, w->ch); w->w = (int)ch->width; w->h = (int)ch->height;
    const uint64_t ml = sfc_metadata_size(w->r); char *meta = calloc(ml + 1, 1);
    if (ml && sfc_read_metadata(w->r, meta, ml)) { free(meta); return -1; }
    const char *q = strstr(meta, "\"voxel_size_um\":"); w->um = q ? atof(q + 16) : 0; free(meta);
    if (!(w->um > 0)) { fprintf(stderr, "patch-id: %s: no voxel_size_um\n", w->path); return -1; }
    const size_t budget = (size_t)((w->w + 63) / 64 + 1) * 3 * 64 * 64 * 12 * 2;
    if (sfc_cache_create(w->r, budget, &w->ca)) return -1;
    w->rz0 = malloc((size_t)w->h * 4); w->rz1 = malloc((size_t)w->h * 4);
    return w->rz0 && w->rz1 ? 0 : -1;
}
typedef struct { pi_job *J; _Atomic int next, fail; double lo[3], hi[3]; pthread_mutex_t mu; } pi_scan;
static void *pi_scan_run(void *arg) {
    pi_scan *S = arg; pi_job *J = S->J;
    for (int k; (k = atomic_fetch_add(&S->next, 1)) < J->nw;) {
        pi_wrap *w = &J->W[k]; const int R = 64; float *xyz = malloc((size_t)R * w->w * 12); uint8_t *v = malloc((size_t)R * w->w);
        double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
        for (int r0 = 0; r0 < w->h && xyz && v; r0 += R) {
            const int n = r0 + R <= w->h ? R : w->h - r0; memset(v, 0, (size_t)n * w->w);
            if (sfc_cache_read_xyz_region(w->ca, w->ch, 0, (uint64_t)r0, (uint32_t)w->w, (uint32_t)n, xyz, 12, (size_t)w->w * 12, v)) { atomic_store(&S->fail, 1); break; }
            for (int r = 0; r < n; r++) {
                float a = 1e30f, b = -1e30f;
                for (int c = 0; c < w->w; c++) { const size_t i = (size_t)r * w->w + c; if (!v[i] || !(xyz[3 * i + 2] >= 0)) continue;
                    const double g[3] = {(xyz[3 * i + 2] + 0.5) * J->f - 0.5, (xyz[3 * i + 1] + 0.5) * J->f - 0.5, (xyz[3 * i] + 0.5) * J->f - 0.5};
                    if (g[0] < a) a = (float)g[0]; if (g[0] > b) b = (float)g[0]; w->nv++;
                    for (int d = 0; d < 3; d++) { if (g[d] < lo[d]) lo[d] = g[d]; if (g[d] > hi[d]) hi[d] = g[d]; } }
                w->rz0[r0 + r] = a; w->rz1[r0 + r] = b;
            }
        }
        sfc_cache_clear(w->ca); free(xyz); free(v);
        pthread_mutex_lock(&S->mu); for (int d = 0; d < 3; d++) { if (lo[d] < S->lo[d]) S->lo[d] = lo[d]; if (hi[d] > S->hi[d]) S->hi[d] = hi[d]; } pthread_mutex_unlock(&S->mu);
    }
    return nullptr;
}
static int pi_load_band(pi_job *J, pi_wrap *w, double z0, double z1) {
    free(w->g); free(w->v); w->g = nullptr; w->v = nullptr; w->bn = 0;
    int a = -1, b = -1;
    for (int r = 0; r < w->h; r++) if (w->rz0[r] <= w->rz1[r] && w->rz1[r] >= z0 && w->rz0[r] <= z1) { if (a < 0) a = r; b = r; }
    if (a < 0) return 0;
    const int n = b - a + 1; const size_t nv = (size_t)n * w->w;
    float *xyz = malloc(nv * 12); w->g = malloc(nv * 16); w->v = calloc(nv, 1);
    if (!xyz || !w->g || !w->v) { free(xyz); return -1; }
    if (sfc_cache_read_xyz_region(w->ca, w->ch, 0, (uint64_t)a, (uint32_t)w->w, (uint32_t)n, xyz, 12, (size_t)w->w * 12, w->v)) { free(xyz); return -1; }
    for (size_t i = 0; i < nv; i++) {
        if (!w->v[i] || !(xyz[3 * i + 2] >= 0)) { w->v[i] = 0; continue; }
        const double X = xyz[3 * i], Y = xyz[3 * i + 1], Z = xyz[3 * i + 2];
        double cy, cx; axis_at(J->ax, Z * J->s0, &cy, &cx); cy /= J->s0; cx /= J->s0;
        float *g = w->g + 4 * i;
        g[0] = (float)((Z + 0.5) * J->f - 0.5); g[1] = (float)((Y + 0.5) * J->f - 0.5); g[2] = (float)((X + 0.5) * J->f - 0.5);
        g[3] = (float)atan2(Y - cy, X - cx);
    }
    free(xyz); sfc_cache_clear(w->ca);
    w->b0 = a; w->bn = n;
    return 0;
}
typedef struct { pi_job *J; double z0, z1; _Atomic int next, fail; } pi_loadjob;
static void *pi_load_run(void *arg) {
    pi_loadjob *L = arg;
    for (int k; (k = atomic_fetch_add(&L->next, 1)) < L->J->nw;) if (pi_load_band(L->J, &L->J->W[k], L->z0, L->z1)) atomic_store(&L->fail, 1);
    return nullptr;
}

/* the axis (grid voxels) at grid z */
static void pi_axis(const pi_job *J, double gz, double *cy, double *cx) {
    const double s = J->s0 / J->f;   /* axis voxels per grid voxel (grid um / axis um) */
    axis_at(J->ax, gz * s, cy, cx); *cy /= s; *cx /= s;
}

/* ---- per block: 6-connected components ---- */
static uint32_t pi_ufl_find(uint32_t *p, uint32_t x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
static void pi_block_cc(pi_job *J, int bz, int by, int bx, uint32_t *p, uint32_t *cnt, uint16_t *map) {
    const int B = J->B; const size_t bi = ((size_t)bz * J->nby + by) * J->nbx + bx;
    const int64_t Z0 = (int64_t)bz * B, Y0 = (int64_t)by * B, X0 = (int64_t)bx * B;
    double cy, cx; pi_axis(J, J->o[0] + Z0 + B / 2.0, &cy, &cx);
    const double yc = J->o[1] + Y0 + B / 2.0, xc = J->o[2] + X0 + B / 2.0, rr = hypot(yc - cy, xc - cx);
    J->ncomp[bi] = 0; J->phi[bi] = NAN;
    #define PI_V(z, y, x) (((size_t)(Z0 + (z)) * J->NY + (size_t)(Y0 + (y))) * J->NX + (size_t)(X0 + (x)))
    if (rr < 2.0 * B) { for (int z = 0; z < B; z++) for (int y = 0; y < B; y++) memset(&J->lab[PI_V(z, y, 0)], 0, (size_t)B * 2); return; }
    J->phi[bi] = (float)atan2(yc - cy, xc - cx);
    const uint32_t NONE = UINT32_MAX;
    for (int z = 0; z < B; z++) for (int y = 0; y < B; y++) for (int x = 0; x < B; x++) {
        const uint32_t i = ((uint32_t)z * B + y) * B + x;
        if (J->sheet[PI_V(z, y, x)] < J->thr) { p[i] = NONE; continue; }
        p[i] = i;
        if (x && p[i - 1] != NONE) { const uint32_t a = pi_ufl_find(p, i - 1), b = pi_ufl_find(p, i); if (a != b) p[a > b ? a : b] = a < b ? a : b; }
        if (y && p[i - B] != NONE) { const uint32_t a = pi_ufl_find(p, i - B), b = pi_ufl_find(p, i); if (a != b) p[a > b ? a : b] = a < b ? a : b; }
        if (z && p[i - B * B] != NONE) { const uint32_t a = pi_ufl_find(p, i - B * B), b = pi_ufl_find(p, i); if (a != b) p[a > b ? a : b] = a < b ? a : b; }
    }
    const uint32_t N = (uint32_t)B * B * B;
    memset(cnt, 0, N * 4);
    for (uint32_t i = 0; i < N; i++) if (p[i] != NONE) cnt[pi_ufl_find(p, i)]++;
    uint16_t nc = 0;
    for (uint32_t i = 0; i < N; i++) { map[i] = 0; if (p[i] == i && cnt[i] >= (uint32_t)J->min_comp && nc < 65534) map[i] = ++nc; }
    for (int z = 0; z < B; z++) for (int y = 0; y < B; y++) for (int x = 0; x < B; x++) {
        const uint32_t i = ((uint32_t)z * B + y) * B + x;
        J->lab[PI_V(z, y, x)] = p[i] == NONE ? 0 : map[pi_ufl_find(p, i)];
    }
    J->ncomp[bi] = nc;
    /* atoms met twice by one line along the block's radial axis (two runs >= 2 voxels apart): two sheets joined by a bridge, or a
       fold -> marked PI_BAD (no links, no votes; a ray that meets one stops there) */
    if (nc && J->bad_gap > 0) {
        const int ax_y = fabs(yc - cy) > fabs(xc - cx);   /* the more radial of y / x */
        int16_t *last = (int16_t *)cnt; uint8_t *bad = (uint8_t *)(cnt + 32768);   /* (cnt is free: N >= 512 words) */
        for (int a = 0; a <= nc; a++) { last[a] = -1000; bad[a] = 0; }
        uint16_t touched[64]; 
        for (int z = 0; z < B; z++) for (int u = 0; u < B; u++) {
            int nt = 0;
            for (int t = 0; t < B; t++) {
                const uint16_t l = J->lab[ax_y ? PI_V(z, t, u) : PI_V(z, u, t)]; if (!l) continue;
                if (last[l] >= 0 && t - last[l] - 1 >= J->bad_gap) bad[l] = 1;
                if (last[l] < 0 && nt < 64) touched[nt++] = l;
                last[l] = (int16_t)t;
            }
            for (int k = 0; k < nt; k++) last[touched[k]] = -1000;
        }
        int nb = 0; for (int a = 1; a <= nc; a++) nb += bad[a];
        if (nb) for (int z = 0; z < B; z++) for (int y = 0; y < B; y++) for (int x = 0; x < B; x++) { uint16_t *l = &J->lab[PI_V(z, y, x)]; if (*l && bad[*l]) *l = PI_BAD; }
        atomic_fetch_add(&J->nbad, nb);
    }
    #undef PI_V
}
typedef struct { pi_job *J; _Atomic long next; } pi_ccjob;
static void *pi_cc_run(void *arg) {
    pi_ccjob *C = arg; pi_job *J = C->J; const uint32_t N = (uint32_t)J->B * J->B * J->B;
    uint32_t *p = malloc(N * 4), *cnt = malloc((N > 65536 ? N : 65536) * 4); uint16_t *map = malloc(N * 2);
    const long nb = (long)J->nbz * J->nby * J->nbx;
    for (long b; (b = atomic_fetch_add(&C->next, 1)) < nb;) pi_block_cc(J, (int)(b / ((long)J->nby * J->nbx)), (int)(b / J->nbx % J->nby), (int)(b % J->nbx), p, cnt, map);
    free(p); free(cnt); free(map);
    return nullptr;
}

/* ---- links across one face: mutual best partners ---- */
typedef struct { uint32_t a, b, n; } pi_pair;
static int pi_cmp_pair(const void *x, const void *y) { const pi_pair *a = x, *b = y; if (a->a != b->a) return a->a < b->a ? -1 : 1; return (a->b > b->b) - (a->b < b->b); }
static void pi_face(pi_job *J, uint32_t *ka, uint32_t *kb, int n, int t, pi_links *L, pi_pair *P) {
    /* ka[i], kb[i]: global atoms on either side of face pixel i (PI_NONE none) */
    int np = 0;
    for (int i = 0; i < n; i++) if (ka[i] != PI_NONE && kb[i] != PI_NONE) P[np++] = (pi_pair){ka[i], kb[i], 1};
    if (!np) return;
    qsort(P, np, sizeof *P, pi_cmp_pair);
    int m = 0; for (int i = 1; i < np; i++) { if (P[i].a == P[m].a && P[i].b == P[m].b) P[m].n++; else P[++m] = P[i]; } np = m + 1;
    for (int i = 0; i < np; i++) {
        const pi_pair *x = &P[i]; if ((int)x->n < J->min_link) continue;
        uint32_t s2a = 0, s2b = 0; int best = 1;
        for (int j = 0; j < np && best; j++) { if (j == i) continue; const pi_pair *y = &P[j];
            if (y->a == x->a) { if (y->n > x->n || (y->n == x->n && y->b < x->b)) best = 0; if (y->n > s2a) s2a = y->n; }
            if (y->b == x->b) { if (y->n > x->n || (y->n == x->n && y->a < x->a)) best = 0; if (y->n > s2b) s2b = y->n; } }
        if (!best || s2a >= J->ratio * x->n || s2b >= J->ratio * x->n) continue;
        pi_lpush(L, (pi_link){x->a, x->b, t, x->n});
    }
}
static inline uint32_t pi_gid(const pi_job *J, size_t bi, uint16_t l) { return l && l != PI_BAD ? J->base[bi] + l - 1 : PI_NONE; }
static inline int pi_turn(float pa, float pb) { return (int)lround((pa - pb) / (2 * M_PI)); }   /* theta_a = theta_b + 2 pi t across neighbouring blocks */
typedef struct { pi_job *J; _Atomic long next; } pi_lkjob;
static void *pi_link_run(void *arg) {
    pi_lkjob *K = arg; pi_job *J = K->J; const int B = J->B; const long nb = (long)J->nbz * J->nby * J->nbx;
    uint32_t *ka = malloc((size_t)B * B * 4), *kb = malloc((size_t)B * B * 4); pi_pair *P = malloc((size_t)B * B * sizeof *P);
    const int tid = atomic_fetch_add(&J->slot, 1);
    for (long b; (b = atomic_fetch_add(&K->next, 1)) < nb;) {
        const int bz = (int)(b / ((long)J->nby * J->nbx)), by = (int)(b / J->nbx % J->nby), bx = (int)(b % J->nbx);
        const size_t bi = (size_t)b; if (isnan(J->phi[bi])) continue;
        const int64_t Z0 = (int64_t)bz * B, Y0 = (int64_t)by * B, X0 = (int64_t)bx * B;
        #define PI_L(z, y, x) J->lab[((size_t)(z) * J->NY + (size_t)(y)) * J->NX + (size_t)(x)]
        pi_links loc = {0};
        if (bx + 1 < J->nbx && !isnan(J->phi[bi + 1])) {   /* +x face */
            for (int z = 0; z < B; z++) for (int y = 0; y < B; y++) { ka[z * B + y] = pi_gid(J, bi, PI_L(Z0 + z, Y0 + y, X0 + B - 1)); kb[z * B + y] = pi_gid(J, bi + 1, PI_L(Z0 + z, Y0 + y, X0 + B)); }
            pi_face(J, ka, kb, B * B, pi_turn(J->phi[bi], J->phi[bi + 1]), &loc, P);
        }
        if (by + 1 < J->nby && !isnan(J->phi[bi + J->nbx])) {   /* +y face */
            for (int z = 0; z < B; z++) for (int x = 0; x < B; x++) { ka[z * B + x] = pi_gid(J, bi, PI_L(Z0 + z, Y0 + B - 1, X0 + x)); kb[z * B + x] = pi_gid(J, bi + J->nbx, PI_L(Z0 + z, Y0 + B, X0 + x)); }
            pi_face(J, ka, kb, B * B, pi_turn(J->phi[bi], J->phi[bi + J->nbx]), &loc, P);
        }
        const size_t bu = bi + (size_t)J->nby * J->nbx;
        if (bz + 1 < J->nbz && !isnan(J->phi[bu])) {   /* +z face inside the band */
            for (int y = 0; y < B; y++) for (int x = 0; x < B; x++) { ka[y * B + x] = pi_gid(J, bi, PI_L(Z0 + B - 1, Y0 + y, X0 + x)); kb[y * B + x] = pi_gid(J, bu, PI_L(Z0 + B, Y0 + y, X0 + x)); }
            pi_face(J, ka, kb, B * B, pi_turn(J->phi[bi], J->phi[bu]), &loc, P);
        }
        if (bz == 0 && J->have_prev) {   /* -z face: the previous band's last plane */
            const size_t pb = (size_t)by * J->nbx + bx;
            if (!isnan(J->prev_phi[pb])) {
                for (int y = 0; y < B; y++) for (int x = 0; x < B; x++) { ka[y * B + x] = pi_gid(J, bi, PI_L(0, Y0 + y, X0 + x)); kb[y * B + x] = J->prev[(size_t)(Y0 + y) * J->NX + (X0 + x)]; }
                pi_face(J, ka, kb, B * B, pi_turn(J->phi[bi], J->prev_phi[pb]), &loc, P);
            }
        }
        { /* the block's atoms: centroid z block, angle bin (atom frame), radius */
            const int nc = J->ncomp[bi];
            if (nc) {
                double *acc = calloc((size_t)nc * 4, sizeof(double));
                for (int z = 0; z < B; z++) for (int y = 0; y < B; y++) for (int x = 0; x < B; x++) {
                    const uint16_t l = PI_L(Z0 + z, Y0 + y, X0 + x); if (!l || l == PI_BAD) continue;
                    double *a = acc + 4 * (l - 1); a[0]++; a[1] += z; a[2] += y; a[3] += x; }
                for (int l = 0; l < nc; l++) {
                    const double *a = acc + 4 * l; const uint32_t g = J->base[bi] + l;
                    if (!(a[0] > 0)) { J->U->r[g] = -1; J->U->zb[g] = J->U->ab[g] = 0; continue; }
                    const double gz = J->o[0] + Z0 + a[1] / a[0], gy = J->o[1] + Y0 + a[2] / a[0], gx = J->o[2] + X0 + a[3] / a[0];
                    double cy, cx; pi_axis(J, gz, &cy, &cx);
                    const double th = atan2(gy - cy, gx - cx), ph = J->phi[bi], tc = ph + pi_ang(th - ph);
                    J->U->r[g] = (float)hypot(gy - cy, gx - cx); J->U->zb[g] = (int32_t)floor(gz / B);
                    J->U->ab[g] = (int32_t)floor((tc + M_PI) / (2 * M_PI) * J->U->NA);
                }
                free(acc);
            }
        }
        #undef PI_L
        for (size_t i = 0; i < loc.n; i++) pi_lpush(&J->tlink[tid], loc.l[i]);
        free(loc.l);
    }
    free(ka); free(kb); free(P);
    return nullptr;
}

/* ---- vertices: the atom along the ray, the GT code ---- */
typedef struct { pi_job *J; _Atomic long next; long *rs; long nrows; int64_t z0, z1; } pi_vjob;
static void *pi_vert_run(void *arg) {
    pi_vjob *V = arg; pi_job *J = V->J; const int tid = atomic_fetch_add(&J->slot, 1);
    pi_recs *R = &J->trec[tid]; long long nv = 0, nh = 0, na = 0, ng = 0; double gre = 0, gim = 0;
    for (long g; (g = atomic_fetch_add(&V->next, 1)) < V->nrows;) {
        int lo = 0, hi = J->nw - 1; while (lo < hi) { const int mid = (lo + hi + 1) / 2; if (V->rs[mid] <= g) lo = mid; else hi = mid - 1; }
        pi_wrap *w = &J->W[lo]; const int r = (int)(g - V->rs[lo]), row = w->b0 + r;
        const uint32_t cb = J->cbase[lo] + (uint32_t)(row / J->CR) * J->NS;
        for (int c = 0; c < w->w; c++) {
            const size_t i = (size_t)r * w->w + c; if (!w->v[i]) continue;
            const float *p = w->g + 4 * i; const int64_t zi = (int64_t)floor(p[0] + 0.5);
            if (zi < V->z0 || zi >= V->z1) continue;
            nv++;
            double cy, cx; pi_axis(J, p[0], &cy, &cx);
            const double th = p[3], dy = p[1] - cy, dx = p[2] - cx, rr = hypot(dy, dx); if (!(rr > 1)) continue;
            const double uy = dy / rr, ux = dx / rr, q = w->id + J->dir * th / (2 * M_PI);
            int sec = (int)floor((th / (2 * M_PI) + 0.5) * J->NS); if (sec < 0) sec = 0; if (sec >= J->NS) sec = J->NS - 1;
            const int64_t lz = zi - J->o[0];
            uint32_t comp = PI_NONE; int16_t jv = 0; int amb = 0;
            for (int s = 0; s <= J->range && comp == PI_NONE && !amb; s++) {
                uint32_t found = PI_NONE; size_t fb = 0;
                for (int sg = -1; sg <= 1; sg += 2) {
                    if (s == 0 && sg > 0) break;
                    const int64_t y = (int64_t)floor(p[1] + sg * s * uy + 0.5) - J->o[1], x = (int64_t)floor(p[2] + sg * s * ux + 0.5) - J->o[2];
                    if (y < 0 || x < 0 || y >= J->NY || x >= J->NX) continue;
                    const uint16_t l = J->lab[((size_t)lz * J->NY + y) * J->NX + x]; if (!l) continue;
                    if (l == PI_BAD) { amb = 1; continue; }
                    const size_t bi = ((size_t)(lz / J->B) * J->nby + y / J->B) * J->nbx + x / J->B; const uint32_t gid = pi_gid(J, bi, l);
                    if (found == PI_NONE) { found = gid; fb = bi; } else if (found != gid) amb = 1;
                }
                if (found != PI_NONE && !amb) {
                    const double ph = J->phi[fb], tp = ph + pi_ang(th - ph); const int m = (int)lround((tp - th) / (2 * M_PI));
                    comp = found; jv = (int16_t)(w->id - (int)J->dir * m);
                }
            }
            if (comp != PI_NONE) nh++; if (amb) na++;
            uint8_t eb = 255, gcode = 0;
            if (J->gt) for (int s = 0; s <= J->gt_range && eb == 255; s++) for (int sg = -1; sg <= 1 && eb == 255; sg += 2) {
                if (s == 0 && sg > 0) break;
                const int64_t y = (int64_t)floor(p[1] + sg * s * uy + 0.5) - J->o[1], x = (int64_t)floor(p[2] + sg * s * ux + 0.5) - J->o[2];
                if (y < 0 || x < 0 || y >= J->NY || x >= J->NX) continue;
                const uint8_t gc = J->gt[((size_t)lz * J->NY + y) * J->NX + x]; if (gc < 1 || gc > 252) continue;
                double e = (gc - 1) / 18.0 - q; e = fmod(e, 14.0); if (e < 0) e += 14;
                gre += cos(2 * M_PI * e); gim += sin(2 * M_PI * e); ng++;
                eb = (uint8_t)(((long)lround(e)) % 14); gcode = gc;
            }
            pi_push(R, (pi_rec){cb + (uint32_t)sec, comp, jv, eb, 0, 1});
            if (eb != 255 && (uint32_t)(pi_mix(((uint64_t)w->id << 42) ^ ((uint64_t)row << 21) ^ (uint64_t)c) >> 32) < J->smp_T) {
                pi_smps *Sm = &J->tsmp[tid];
                if (Sm->n == Sm->cap) { Sm->cap = Sm->cap ? 2 * Sm->cap : 1 << 14; Sm->s = realloc(Sm->s, Sm->cap * sizeof *Sm->s); if (!Sm->s) abort(); }
                Sm->s[Sm->n++] = (pi_smp){{(float)(p[0] * J->mm), (float)(p[1] * J->mm), (float)(p[2] * J->mm)}, cb + (uint32_t)sec, comp, jv, eb, gcode};
            }
        }
        if (R->n > (size_t)1 << 24) pi_reduce(R);
    }
    atomic_fetch_add(&J->nvert, nv); atomic_fetch_add(&J->nhit, nh); atomic_fetch_add(&J->namb, na); atomic_fetch_add(&J->ngt, ng);
    J->gre[tid] += gre; J->gim[tid] += gim;
    return nullptr;
}

typedef struct { uint32_t root; int J, G; uint32_t n; } pi_vote;   /* G: GT sheet in the root frame mod 14 (99 none) */
static int pi_cmp_vote(const void *a, const void *b) { const pi_vote *x = a, *y = b; if (x->root != y->root) return x->root < y->root ? -1 : 1; if (x->J != y->J) return x->J < y->J ? -1 : 1; return (x->G > y->G) - (x->G < y->G); }
typedef struct { uint32_t root; int Jstar, ok; uint32_t votes; double share, purity; int Gmode; } pi_patch;
static int pi_cmp_patch(const void *a, const void *b) { const pi_patch *x = a, *y = b; return (x->root > y->root) - (x->root < y->root); }

static int pi_cmpu(const void *a, const void *b) { const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return (x > y) - (x < y); }

int cmd_patch_id(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: ufsm patch-id <wraps-dir> <out-dir> --axis F --axis-um U0 --dir +-1 --sheet ROOT KEY --grid-um U\n"
                        "       [--thr 1] [--zr Z0,Z1] [--block 32] [--band 64] [--min-comp 8] [--range 3] [--min-link 16] [--ratio 0.3]\n"
                        "       [--min-votes 500] [--min-share 0.5] [--cells 32,32] [--gt ROOT KEY [--gt-range 4]] [--threads N]\n");
        return 2;
    }
    const char *in = argv[2], *out = argv[3], *axp = pi_opt(argc, argv, "--axis", nullptr);
    const double axum = atof(pi_opt(argc, argv, "--axis-um", "0")), gum = atof(pi_opt(argc, argv, "--grid-um", "0")), dir = atof(pi_opt(argc, argv, "--dir", "0"));
    pi_job J; memset(&J, 0, sizeof J);
    J.nt = ufsm_threads(atoi(pi_opt(argc, argv, "--threads", "0")), 16); if (J.nt > 64) J.nt = 64;
    J.B = atoi(pi_opt(argc, argv, "--block", "32")); J.BAND = atoi(pi_opt(argc, argv, "--band", "64"));
    J.thr = (uint8_t)atoi(pi_opt(argc, argv, "--thr", "1")); J.min_comp = atoi(pi_opt(argc, argv, "--min-comp", "8"));
    J.range = atoi(pi_opt(argc, argv, "--range", "3")); J.gt_range = atoi(pi_opt(argc, argv, "--gt-range", "4"));
    J.min_link = atoi(pi_opt(argc, argv, "--min-link", "16")); J.bad_gap = atoi(pi_opt(argc, argv, "--bad-gap", "2")); J.ratio = atof(pi_opt(argc, argv, "--ratio", "0.3"));
    const long long min_votes = atoll(pi_opt(argc, argv, "--min-votes", "500")); const double min_share = atof(pi_opt(argc, argv, "--min-share", "0.5"));
    if (sscanf(pi_opt(argc, argv, "--cells", "32,32"), "%d,%d", &J.CR, &J.NS) != 2 || J.CR < 1 || J.NS < 1) { fprintf(stderr, "patch-id: --cells R,S\n"); return 2; }
    J.dir = dir;
    const int si = pi_idx(argc, argv, "--sheet");
    if (!axp || !(axum > 0) || !(gum > 0) || (dir != 1 && dir != -1) || si < 0 || si + 2 >= argc || J.B < 8 || J.B > 64 || J.BAND < J.B || J.BAND % J.B || J.thr < 1) {
        fprintf(stderr, "patch-id: need --axis F --axis-um U0 --grid-um U --dir 1|-1 --sheet ROOT KEY; --block 8..64 dividing --band\n"); return 2;
    }
    axis ax = {0}; if (axis_load(&ax, axp)) { fprintf(stderr, "patch-id: cannot load the axis %s\n", axp); return 1; }
    J.ax = &ax;
    z3 *zs = z3_open(store_open(argv[si + 1]), argv[si + 2], nullptr); if (!zs) { fprintf(stderr, "patch-id: --sheet: %s\n", z3_error()); return 1; }
    const int64_t *shape = z3_meta_of(zs)->shape;
    z3 *zg = nullptr;
    { const int gi = pi_idx(argc, argv, "--gt"); if (gi > 0) { if (gi + 2 >= argc) return 2; zg = z3_open(store_open(argv[gi + 1]), argv[gi + 2], nullptr);
        if (!zg || memcmp(z3_meta_of(zg)->shape, shape, 3 * sizeof *shape)) { fprintf(stderr, "patch-id: --gt %s %s: cannot open or not on the sheet grid\n", argv[gi + 1], argv[gi + 2]); return 1; } } }
    /* wraps */
    DIR *dd = opendir(in); if (!dd) { fprintf(stderr, "patch-id: cannot open %s\n", in); return 1; }
    int nw = 0, cap = 64; pi_wrap *W = calloc(cap, sizeof *W); struct dirent *de;
    while ((de = readdir(dd))) {
        int id; char tail[8] = ""; if (sscanf(de->d_name, "w%d.%7s", &id, tail) != 2 || strcmp(tail, "sfc")) continue;
        if (nw == cap) { cap *= 2; W = realloc(W, cap * sizeof *W); }
        memset(&W[nw], 0, sizeof W[nw]); W[nw].id = id; snprintf(W[nw].path, sizeof W[nw].path, "%s/%s", in, de->d_name); nw++;
    }
    closedir(dd);
    if (!nw) { fprintf(stderr, "patch-id: no wNNN.sfc in %s\n", in); return 1; }
    qsort(W, nw, sizeof *W, pi_cmp_id);
    for (int k = 0; k < nw; k++) { if (pi_open(&W[k])) return 1; if (fabs(W[k].um - W[0].um) > 1e-9) { fprintf(stderr, "patch-id: wraps at different voxel sizes\n"); return 1; } }
    J.W = W; J.nw = nw; J.s0 = W[0].um / axum; J.f = W[0].um / gum;
    const double t0 = pi_now();
    pi_scan S = {&J, 0, 0, {1e30, 1e30, 1e30}, {-1e30, -1e30, -1e30}, PTHREAD_MUTEX_INITIALIZER};
    { pthread_t th[64]; for (int t = 0; t < J.nt; t++) pthread_create(&th[t], nullptr, pi_scan_run, &S); for (int t = 0; t < J.nt; t++) pthread_join(th[t], nullptr); }
    if (atomic_load(&S.fail)) { fprintf(stderr, "patch-id: cannot read the wraps\n"); return 1; }
    /* cells */
    J.nrb = malloc(nw * sizeof *J.nrb); J.cbase = malloc((nw + 1) * sizeof *J.cbase); J.cbase[0] = 0;
    for (int k = 0; k < nw; k++) { J.nrb[k] = (W[k].h + J.CR - 1) / J.CR; J.cbase[k + 1] = J.cbase[k] + (uint32_t)J.nrb[k] * J.NS; }
    /* the extent: wraps' box + margin, block-aligned, inside the grid */
    int64_t zr0 = 0, zr1 = shape[0];
    { long long a, b; if (pi_opt(argc, argv, "--zr", nullptr) && sscanf(pi_opt(argc, argv, "--zr", ""), "%lld,%lld", &a, &b) == 2) { zr0 = a; zr1 = b; } }
    { const int64_t a = (int64_t)floor(S.lo[0]) - 1, b = (int64_t)ceil(S.hi[0]) + 2; if (a > zr0) zr0 = a; if (b < zr1) zr1 = b; if (zr0 < 0) zr0 = 0; if (zr1 > shape[0]) zr1 = shape[0]; }
    int64_t oy = (int64_t)floor(S.lo[1]) - J.range - 2, ox = (int64_t)floor(S.lo[2]) - J.range - 2, ey = (int64_t)ceil(S.hi[1]) + J.range + 3, ex = (int64_t)ceil(S.hi[2]) + J.range + 3;
    if (oy < 0) oy = 0; if (ox < 0) ox = 0; if (ey > shape[1]) ey = shape[1]; if (ex > shape[2]) ex = shape[2];
    J.nby = (int)((ey - oy + J.B - 1) / J.B); J.nbx = (int)((ex - ox + J.B - 1) / J.B); J.NY = (int64_t)J.nby * J.B; J.NX = (int64_t)J.nbx * J.B;
    J.nbz = J.BAND / J.B;
    const size_t plane = (size_t)J.NY * J.NX, bvox = (size_t)J.BAND * plane, nbb = (size_t)J.nbz * J.nby * J.nbx;
    long long nvt = 0; for (int k = 0; k < nw; k++) nvt += W[k].nv;
    fprintf(stderr, "patch-id: %d wraps, %lld vertices, grid z %lld..%lld, y %lld+%lld x %lld+%lld, bands of %d, blocks of %d, %.1f GB band buffers\n", nw, nvt,
            (long long)zr0, (long long)zr1, (long long)oy, (long long)J.NY, (long long)ox, (long long)J.NX, J.BAND, J.B, (double)bvox * (zg ? 4 : 3) / 1e9 + plane * 4 / 1e9);
    J.sheet = malloc(bvox); J.lab = malloc(bvox * 2); J.gt = zg ? malloc(bvox) : nullptr;
    J.base = malloc(nbb * 4); J.ncomp = malloc(nbb * 2); J.phi = malloc(nbb * 4);
    J.prev = malloc(plane * 4); J.prev_phi = malloc((size_t)J.nby * J.nbx * 4);
    J.trec = calloc(J.nt, sizeof *J.trec); J.tsmp = calloc(J.nt, sizeof *J.tsmp); J.mm = gum * 1e-3;
    J.smp_T = (uint32_t)(4294967295.0 * atof(pi_opt(argc, argv, "--pair-rate", "0.02"))); J.tlink = calloc(J.nt, sizeof *J.tlink); J.gre = calloc(J.nt, sizeof(double)); J.gim = calloc(J.nt, sizeof(double));
    if (!J.sheet || !J.lab || (zg && !J.gt) || !J.base || !J.prev) { fprintf(stderr, "patch-id: no memory for the band buffers\n"); return 1; }
    pi_uf U = {0}; size_t natoms = 0; J.U = &U;
    U.NA = atoi(pi_opt(argc, argv, "--hf-bins", "2048")); U.tol = (float)atof(pi_opt(argc, argv, "--hf-tol", "3")); pi_recs ALL = {0};
    double tread = 0, tcc = 0, tlink = 0, tvert = 0;
    for (int64_t z0 = zr0; z0 < zr1; z0 += J.BAND) {
        const int64_t z1 = z0 + J.BAND < zr1 ? z0 + J.BAND : zr1, nz = z1 - z0;
        J.o[0] = z0; J.o[1] = oy; J.o[2] = ox;
        double ta = pi_now();
        memset(J.sheet, 0, bvox); if (J.gt) memset(J.gt, 0, bvox);
        { const int64_t o[3] = {z0, oy, ox}, n[3] = {nz, ey - oy, ex - ox};
          uint8_t *tmp = malloc((size_t)n[0] * n[1] * n[2]); if (!tmp) return 1;
          for (int pass = 0; pass < (zg ? 2 : 1); pass++) {
              if (z3_read(pass ? zg : zs, o, n, tmp, J.nt)) { fprintf(stderr, "patch-id: %s\n", z3_error()); return 1; }
              uint8_t *dst = pass ? J.gt : J.sheet;
              for (int64_t a = 0; a < n[0]; a++) for (int64_t b = 0; b < n[1]; b++) memcpy(dst + ((size_t)a * J.NY + b) * J.NX, tmp + ((size_t)a * n[1] + b) * n[2], (size_t)n[2]);
          }
          free(tmp); }
        { pi_loadjob L = {&J, (double)z0 - 1, (double)z1 + 1, 0, 0}; pthread_t th[64]; for (int t = 0; t < J.nt; t++) pthread_create(&th[t], nullptr, pi_load_run, &L); for (int t = 0; t < J.nt; t++) pthread_join(th[t], nullptr);
          if (atomic_load(&L.fail)) { fprintf(stderr, "patch-id: cannot read the wraps' band\n"); return 1; } }
        double tb = pi_now(); tread += tb - ta;
        { pi_ccjob C = {&J, 0}; pthread_t th[64]; for (int t = 0; t < J.nt; t++) pthread_create(&th[t], nullptr, pi_cc_run, &C); for (int t = 0; t < J.nt; t++) pthread_join(th[t], nullptr); }
        for (size_t b = 0; b < nbb; b++) { J.base[b] = (uint32_t)natoms; natoms += J.ncomp[b]; }
        if (natoms >= PI_NONE) { fprintf(stderr, "patch-id: too many atoms\n"); return 1; }
        pi_uf_grow(&U, natoms);
        for (size_t a = U.n; a < natoms; a++) { U.r[a] = -1; U.par[a] = (uint32_t)a; }
        double tc = pi_now(); tcc += tc - tb;
        { for (int t = 0; t < J.nt; t++) J.tlink[t].n = 0;
          pi_lkjob K = {&J, 0}; pthread_t th[64]; atomic_store(&J.slot, 0);
          for (int t = 0; t < J.nt; t++) pthread_create(&th[t], nullptr, pi_link_run, &K); for (int t = 0; t < J.nt; t++) pthread_join(th[t], nullptr); }
        long long nl = 0;
        for (size_t a = U.n; a < natoms; a++) pi_atom(&U, (uint32_t)a);
        U.n = natoms;
        { pi_links AL = {0}; for (int t = 0; t < J.nt; t++) for (size_t i = 0; i < J.tlink[t].n; i++) pi_lpush(&AL, J.tlink[t].l[i]);
          qsort(AL.l, AL.n, sizeof *AL.l, pi_cmp_link);   /* strongest contacts first */
          for (size_t i = 0; i < AL.n; i++) pi_join(&U, AL.l[i].a, AL.l[i].b, AL.l[i].t);
          nl += (long long)AL.n; free(AL.l); }
        double td = pi_now(); tlink += td - tc;
        /* vertices */
        { long *rs = malloc((nw + 1) * sizeof *rs); rs[0] = 0; for (int k = 0; k < nw; k++) rs[k + 1] = rs[k] + W[k].bn;
          pi_vjob V = {&J, 0, rs, rs[nw], z0, z1}; pthread_t th[64]; atomic_store(&J.slot, 0); for (int t = 0; t < J.nt; t++) pthread_create(&th[t], nullptr, pi_vert_run, &V); for (int t = 0; t < J.nt; t++) pthread_join(th[t], nullptr);
          free(rs); }
        for (int t = 0; t < J.nt; t++) { pi_reduce(&J.trec[t]); for (size_t i = 0; i < J.trec[t].n; i++) pi_push(&ALL, J.trec[t].r[i]); J.trec[t].n = 0; }
        if (ALL.n > (size_t)1 << 26) pi_reduce(&ALL);
        tvert += pi_now() - td;
        /* the last plane for the next band */
        { const int64_t lz = nz - 1;
          for (int64_t y = 0; y < J.NY; y++) for (int64_t x = 0; x < J.NX; x++) {
              const size_t bi = ((size_t)(lz / J.B) * J.nby + y / J.B) * J.nbx + x / J.B;
              J.prev[(size_t)y * J.NX + x] = pi_gid(&J, bi, J.lab[((size_t)lz * J.NY + y) * J.NX + x]);
          }
          for (int b = 0; b < J.nby * J.nbx; b++) J.prev_phi[b] = J.phi[(size_t)(lz / J.B) * J.nby * J.nbx + b];
          J.have_prev = 1; }
        fprintf(stderr, "\rpatch-id: z %lld / %lld: %zu atoms, %lld merges (%lld loop / %lld height-field refusals), links %lld, %lld vertices on an atom of %lld  %.0fs   ",
                (long long)z1, (long long)zr1, natoms, U.merges, U.cycles, U.refused, nl, (long long)atomic_load(&J.nhit), (long long)atomic_load(&J.nvert), pi_now() - t0);
    }
    fprintf(stderr, "\npatch-id: read %.0fs, components %.0fs, links %.0fs, vertices %.0fs\n", tread, tcc, tlink, tvert);
    pi_reduce(&ALL);
    for (int k = 0; k < nw; k++) { free(W[k].g); free(W[k].v); sfc_cache_destroy(W[k].ca); sfc_close(W[k].r); }
    free(J.sheet); free(J.lab); free(J.gt); free(J.prev);
    /* GT gauge: the circular mean fraction, then the modal whole turn */
    double gre = 0, gim = 0; for (int t = 0; t < J.nt; t++) { gre += J.gre[t]; gim += J.gim[t]; }
    const double gfr = atan2(gim, gre) / (2 * M_PI);
    /* votes per root */
    size_t nvo = 0; pi_vote *VO = malloc((ALL.n + 1) * sizeof *VO); int *Tof = malloc((ALL.n + 1) * sizeof *Tof); uint32_t *Rof = malloc((ALL.n + 1) * sizeof *Rof);
    for (size_t i = 0; i < ALL.n; i++) {
        const pi_rec *r = &ALL.r[i]; Rof[i] = PI_NONE; Tof[i] = 0; if (r->comp == PI_NONE) continue;
        int T; const uint32_t root = pi_find(&U, r->comp, &T); Rof[i] = root; Tof[i] = T;
        const int Jv = r->jv - (int)dir * T;
        VO[nvo++] = (pi_vote){root, Jv, r->eb == 255 ? 99 : ((r->eb + Jv) % 14 + 14) % 14, r->n};
    }
    qsort(VO, nvo, sizeof *VO, pi_cmp_vote);
    size_t npa = 0, cpa = 1024; pi_patch *PA = malloc(cpa * sizeof *PA);
    for (size_t i = 0; i < nvo;) {
        size_t e = i; long long tot = 0, best = 0; int bj = 0; long long gh[14] = {0}, gtot = 0;
        while (e < nvo && VO[e].root == VO[i].root) e++;
        for (size_t j = i; j < e;) { size_t f = j; long long s = 0; while (f < e && VO[f].J == VO[j].J) { s += VO[f].n; if (VO[f].G < 14) { gh[VO[f].G] += VO[f].n; gtot += VO[f].n; } f++; } tot += s; if (s > best) { best = s; bj = VO[j].J; } j = f; }
        long long gm = 0; int gmb = -1; for (int b = 0; b < 14; b++) if (gh[b] > gm) { gm = gh[b]; gmb = b; }
        if (npa == cpa) { cpa *= 2; PA = realloc(PA, cpa * sizeof *PA); }
        PA[npa++] = (pi_patch){VO[i].root, bj, tot >= min_votes && best >= min_share * tot, (uint32_t)(tot > UINT32_MAX ? UINT32_MAX : tot), tot ? (double)best / tot : 0, gtot ? (double)gm / gtot : NAN, gmb};
        i = e;
    }
    free(VO);
    /* patch stats */
    long long vtot = 0, vok = 0; double pur_w = 0, pur_n = 0, sh_w = 0; size_t nok = 0;
    for (size_t p = 0; p < npa; p++) { vtot += PA[p].votes; if (PA[p].ok) { vok += PA[p].votes; nok++; sh_w += PA[p].share * PA[p].votes; if (isfinite(PA[p].purity)) { pur_w += PA[p].purity * PA[p].votes; pur_n += PA[p].votes; } } }
    /* GT: the global modal turn */
    long long eh[14] = {0}; for (size_t i = 0; i < ALL.n; i++) if (ALL.r[i].eb < 14) eh[ALL.r[i].eb] += ALL.r[i].n;   /* (eb: rounded e; the gauge fraction should be near 0) */
    int gm = 0; for (int b = 1; b < 14; b++) if (eh[b] > eh[gm]) gm = b;
    /* --oracle 1 (validation only): each kept patch takes its GT-modal sheet instead of the wraps' majority (the ceiling of patch
       relabelling with perfect anchors) */
    const int oracle = atoi(pi_opt(argc, argv, "--oracle", "0"));
    #define PI_DK(P, Jv) (oracle && (P)->Gmode >= 0 ? ((((P)->Gmode - gm - (Jv)) % 14 + 21) % 14 - 7) : (P)->Jstar - (Jv))
    /* per record: the correction */
    int8_t *dk = malloc(ALL.n + 1); uint8_t *dok = calloc(ALL.n + 1, 1); uint32_t *pid = malloc((ALL.n + 1) * 4);
    for (size_t i = 0; i < ALL.n; i++) {
        dk[i] = 0; pid[i] = PI_NONE; if (Rof[i] == PI_NONE) continue;
        pi_patch key = {Rof[i], 0, 0, 0, 0, 0, 0}; const pi_patch *P = bsearch(&key, PA, npa, sizeof *PA, pi_cmp_patch);
        if (!P || !P->ok) continue;
        const int Jv = ALL.r[i].jv - (int)dir * Tof[i], d = PI_DK(P, Jv);
        if (d < -PI_DKR || d > PI_DKR) continue;
        dk[i] = (int8_t)d; dok[i] = 1; pid[i] = Rof[i];
    }
    /* cells: records are sorted by cell */
    mkdir(out, 0775);
    char pth[1600]; snprintf(pth, sizeof pth, "%s/cells.tsv", out); FILE *fc = fopen(pth, "w"); if (!fc) { fprintf(stderr, "patch-id: cannot write %s\n", pth); return 1; }
    fprintf(fc, "wrap\trow0\tsector\tpatch\tdk\tconf\tbrk\tn\tnpatch\n");
    typedef struct { long long n, g, rb, rr, rc, np, pv; } wst; wst *WS = calloc(nw, sizeof *WS);
    long long ncell = 0, ncellp = 0, ncellfix = 0;
    const uint32_t ncells = J.cbase[nw]; int8_t *cell_dk = calloc(ncells + 1, 1);
    for (size_t i = 0; i < ALL.n;) {
        size_t e = i; while (e < ALL.n && ALL.r[e].cell == ALL.r[i].cell) e++;
        const uint32_t cell = ALL.r[i].cell; int wi = 0; while (wi + 1 < nw && J.cbase[wi + 1] <= cell) wi++;
        const int rb = (int)((cell - J.cbase[wi]) / J.NS), sec = (int)((cell - J.cbase[wi]) % J.NS);
        long long n = 0, np = 0, h[2 * PI_DKR + 1] = {0};
        typedef struct { uint32_t p; long long n; } pc; pc pcs[8]; int npc = 0;
        for (size_t j = i; j < e; j++) {
            n += ALL.r[j].n; if (!dok[j]) continue;
            np += ALL.r[j].n; h[dk[j] + PI_DKR] += ALL.r[j].n;
            int f = -1; for (int q = 0; q < npc; q++) if (pcs[q].p == pid[j]) f = q;
            if (f < 0 && npc < 8) { pcs[npc] = (pc){pid[j], 0}; f = npc++; } if (f >= 0) pcs[f].n += ALL.r[j].n;
        }
        int md = 0; for (int b = 0; b < 2 * PI_DKR + 1; b++) if (h[b] > h[md]) md = b;
        const int cdk = md - PI_DKR; const double conf = np ? (double)h[md] / np : 0;
        uint32_t mp = PI_NONE; long long mpn = 0, sec2 = 0; for (int q = 0; q < npc; q++) { if (pcs[q].n > mpn) { sec2 = mpn; mpn = pcs[q].n; mp = pcs[q].p; } else if (pcs[q].n > sec2) sec2 = pcs[q].n; }
        const int brk = np && (conf < 0.8 || sec2 > 0.2 * np);
        const int use = np >= 0.3 * n && conf >= 0.5;
        ncell++; ncellp += np > 0; ncellfix += use && cdk != 0; if (use) cell_dk[cell] = (int8_t)cdk;
        if (np) fprintf(fc, "%d\t%d\t%d\t%lld\t%d\t%.3f\t%d\t%lld\t%lld\n", W[wi].id, rb * J.CR, sec, mp == PI_NONE ? -1LL : (long long)mp, cdk, conf, brk, n, np);
        wst *s = &WS[wi]; s->n += n; s->np += np;
        for (size_t j = i; j < e; j++) {
            if (ALL.r[j].eb >= 14) continue;
            const long long c = ALL.r[j].n; const int e0 = ALL.r[j].eb;
            s->g += c; s->rb += c * (e0 == gm);
            s->rr += c * ((((e0 - (dok[j] ? dk[j] : 0)) % 14 + 14) % 14) == gm);
            s->rc += c * ((((e0 - (use ? cdk : 0)) % 14 + 14) % 14) == gm);
        }
        i = e;
    }
    fclose(fc);
    snprintf(pth, sizeof pth, "%s/wraps.tsv", out); FILE *fw = fopen(pth, "w");
    if (fw) fprintf(fw, "wrap\tvertices\tpatched\tgt\tright_before\tright_vote\tright_cell\n");
    wst A = {0}, Pk = {0};
    for (int k = 0; k < nw; k++) {
        const wst *s = &WS[k];
        if (fw) fprintf(fw, "w%03d\t%lld\t%.3f\t%lld\t%.4f\t%.4f\t%.4f\n", W[k].id, s->n, s->n ? (double)s->np / s->n : 0, s->g, s->g ? (double)s->rb / s->g : NAN, s->g ? (double)s->rr / s->g : NAN, s->g ? (double)s->rc / s->g : NAN);
        A.n += s->n; A.np += s->np; A.g += s->g; A.rb += s->rb; A.rr += s->rr; A.rc += s->rc;
        if (W[k].id >= 89 && W[k].id <= 170) { Pk.n += s->n; Pk.np += s->np; Pk.g += s->g; Pk.rb += s->rb; Pk.rr += s->rr; Pk.rc += s->rc; }
    }
    if (fw) fclose(fw);
    printf("patch-id: %zu atoms, %lld merges, %lld loop / %lld height-field refusals; %zu patches with votes, %zu kept (>= %lld votes, majority >= %.2f): %.3f of the voted vertices; "
           "majority share %.3f (vote-weighted)\n", natoms, U.merges, U.cycles, U.refused, npa, nok, min_votes, min_share, vtot ? (double)vok / vtot : 0, vok ? sh_w / vok : 0);
    printf("patch-id: vertices %lld, on an atom %.3f (ambiguous %.3f), in a kept patch %.3f; cells %lld, with patch votes %lld, corrected (dk != 0) %lld\n",
           (long long)J.nvert, J.nvert ? (double)J.nhit / J.nvert : 0, J.nvert ? (double)J.namb / J.nvert : 0, A.n ? (double)A.np / A.n : 0, ncell, ncellp, ncellfix);
    if (zg) {
        printf("patch-id: GT gauge %.3f + %d turns; GT-coded vertices %lld (%.3f); kept patches' GT purity %.3f (vote-weighted: the ceiling of patch relabelling)\n",
               gfr, gm, A.g, A.n ? (double)A.g / A.n : 0, pur_n ? pur_w / pur_n : NAN);
        printf("patch-id: right sheet: all wraps before %.4f, per-vote fix %.4f, per-cell fix %.4f | w089-w170 before %.4f, vote %.4f, cell %.4f\n",
               A.g ? (double)A.rb / A.g : NAN, A.g ? (double)A.rr / A.g : NAN, A.g ? (double)A.rc / A.g : NAN, Pk.g ? (double)Pk.rb / Pk.g : NAN, Pk.g ? (double)Pk.rr / Pk.g : NAN, Pk.g ? (double)Pk.rc / Pk.g : NAN);
    }
    /* pair winding accuracy on the sampled GT-coded vertices (src/pairacc.h): before, per-vote fix, per-cell fix */
    { size_t ns = 0; for (int t = 0; t < J.nt; t++) ns += J.tsmp[t].n;
      if (zg && ns > 100) {
          pa_buf pb[3]; for (int v = 0; v < 3; v++) pa_init(&pb[v], ns + 16);
          for (int t = 0; t < J.nt; t++) for (size_t i = 0; i < J.tsmp[t].n; i++) {
              const pi_smp *q = &J.tsmp[t].s[i]; int d1 = 0;
              if (q->comp != PI_NONE) { int T; const uint32_t root = pi_find(&U, q->comp, &T); pi_patch key = {root, 0, 0, 0, 0, 0, 0};
                  const pi_patch *P = bsearch(&key, PA, npa, sizeof *PA, pi_cmp_patch);
                  if (P && P->ok) { const int d = PI_DK(P, q->jv - (int)dir * T); if (d >= -PI_DKR && d <= PI_DKR) d1 = d; } }
              const int e = q->eb, dd[3] = {0, d1, cell_dk[q->cell]};
              const uint64_t key = ((uint64_t)t << 40) ^ i;
              for (int v = 0; v < 3; v++) pa_add(&pb[v], q->p[0], q->p[1], q->p[2], (float)-(e - dd[v]), q->gc, key);
          }
          const double dist[4] = {0.5, 1, 2, 4}; const char *nm[3] = {"before", "vote fix", "cell fix"};
          for (int v = 0; v < 3; v++) { double acc[4], accx[4]; size_t n[4], nx[4]; pa_score(&pb[v], dist, 4, 200000, 7, acc, accx, n, nx);
              printf("patch-id: pair winding right (%zu samples) %-8s: 0.5 mm %.4f | 1 mm %.4f | 2 mm %.4f | 4 mm %.4f\n", ns, nm[v], acc[0], acc[1], acc[2], acc[3]);
              pa_free(&pb[v]); }
      } }
    { /* patch size / purity over every patch with enough votes (also those without a majority) */
      long long nbig = 0, vbig = 0; double pw = 0, pn = 0; for (size_t p = 0; p < npa; p++) if (PA[p].votes >= min_votes) { nbig++; vbig += PA[p].votes; if (isfinite(PA[p].purity)) { pw += PA[p].purity * PA[p].votes; pn += PA[p].votes; } }
      printf("patch-id: %lld patches with >= %lld votes hold %.3f of the votes; their GT purity %.3f; %lld atoms rejected as bridged / folded\n", nbig, min_votes, vtot ? (double)vbig / vtot : 0, pn ? pw / pn : NAN, (long long)J.nbad);
      uint32_t *vs = malloc((npa + 1) * 4); for (size_t p = 0; p < npa; p++) vs[p] = PA[p].votes;
      qsort(vs, npa, 4, pi_cmpu);
      long long acc = 0, half = vtot / 2; uint32_t med = 0; for (size_t p = 0; p < npa; p++) { acc += vs[p]; if (acc >= half) { med = vs[p]; break; } }
      printf("patch-id: patch votes: largest %u, vote-weighted median %u\n", npa ? vs[npa - 1] : 0, med);
      free(vs); }
    free(cell_dk);
    free(ALL.r); free(dk); free(dok); free(pid); free(Rof); free(Tof); free(PA); free(WS);
    z3_close(zs); if (zg) z3_close(zg);
    return 0;
}
