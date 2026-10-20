/* Least-cost integer potentials by graph-cut moves (src/l1pot.h). */
#include "l1pot.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int vote_cmp(const void *p, const void *q) {
    const l1_vote *x = p, *y = q;
    if (x->a != y->a) return x->a < y->a ? -1 : 1;
    if (x->b != y->b) return x->b < y->b ? -1 : 1;
    return x->d != y->d ? (x->d < y->d ? -1 : 1) : 0;
}
size_t l1_votes_merge(l1_vote *v, size_t nv) {
    size_t m = 0;
    for (size_t i = 0; i < nv; i++) {
        if (v[i].a == v[i].b || v[i].w <= 0) continue;
        if (v[i].a > v[i].b) { const uint32_t t = v[i].a; v[i].a = v[i].b; v[i].b = t; v[i].d = -v[i].d; }
        v[m++] = v[i];
    }
    qsort(v, m, sizeof *v, vote_cmp);
    size_t o = 0;
    for (size_t i = 0; i < m; i++) {
        if (o && v[o - 1].a == v[i].a && v[o - 1].b == v[i].b && v[o - 1].d == v[i].d) v[o - 1].w += v[i].w;
        else v[o++] = v[i];
    }
    return o;
}
static inline int64_t iabs64(int64_t x) { return x < 0 ? -x : x; }
int64_t l1_energy(const l1_vote *v, size_t nv, const int32_t *s) {
    int64_t e = 0;
    for (size_t i = 0; i < nv; i++) e += v[i].w * iabs64((int64_t)s[v[i].b] - s[v[i].a] - v[i].d);
    return e;
}

/* Dinic max flow on a fixed arc structure (capacities set per move) */
typedef struct {
    int N; size_t M;                 /* nodes (incl. source N - 2, sink N - 1), arcs (pairs: arc k and its reverse k ^ 1) */
    uint32_t *to; int64_t *cap;      /* per arc */
    size_t *first, *adj;             /* CSR: arcs leaving node u: adj[first[u] .. first[u + 1]) */
    int *lev; size_t *it; uint32_t *q;
    size_t *stk;                     /* the DFS path (arcs) */
} mf_graph;
static int64_t mf_dfs(mf_graph *g, int S, int T) {   /* one blocking flow (iterative DFS over the level graph) */
    int64_t total = 0;
    for (;;) {
        size_t depth = 0; uint32_t u = (uint32_t)S;
        for (;;) {
            if ((int)u == T) {   /* augment by the path's bottleneck */
                int64_t f = INT64_MAX; for (size_t k = 0; k < depth; k++) if (g->cap[g->stk[k]] < f) f = g->cap[g->stk[k]];
                for (size_t k = 0; k < depth; k++) { g->cap[g->stk[k]] -= f; g->cap[g->stk[k] ^ 1] += f; }
                total += f; depth = 0; u = (uint32_t)S; continue;
            }
            int adv = 0;
            for (; g->it[u] < g->first[u + 1]; g->it[u]++) {
                const size_t a = g->adj[g->it[u]]; const uint32_t w = g->to[a];
                if (g->cap[a] > 0 && g->lev[w] == g->lev[u] + 1) { g->stk[depth++] = a; u = w; adv = 1; break; }
            }
            if (adv) continue;
            if (!depth) return total;   /* the source is blocked */
            g->lev[u] = -1;   /* dead end: retreat */
            const size_t a = g->stk[--depth]; u = g->to[a ^ 1]; g->it[u]++;
        }
    }
}
static int64_t mf_run(mf_graph *g) {
    const int S = g->N - 2, T = g->N - 1; int64_t flow = 0;
    for (;;) {
        for (int i = 0; i < g->N; i++) g->lev[i] = -1;
        size_t qh = 0, qt = 0; g->q[qt++] = (uint32_t)S; g->lev[S] = 0;
        while (qh < qt) {
            const uint32_t u = g->q[qh++];
            for (size_t k = g->first[u]; k < g->first[u + 1]; k++) { const size_t a = g->adj[k]; const uint32_t w = g->to[a]; if (g->cap[a] > 0 && g->lev[w] < 0) { g->lev[w] = g->lev[u] + 1; g->q[qt++] = w; } }
        }
        if (g->lev[T] < 0) return flow;
        for (int i = 0; i < g->N; i++) g->it[i] = g->first[i];
        flow += mf_dfs(g, S, T);
    }
}

/* Boykov-Kolmogorov max flow ("An experimental comparison of min-cut/max-flow algorithms", PAMI 2004): two search trees grown from the
   terminals, paths augmented where they meet, orphans re-adopted (with the time / distance heuristic). Node-only graph: arc 2e is
   ea[e] -> eb[e], 2e + 1 its reverse; tr[u] > 0 a residual from the source, < 0 to the sink. On these piece graphs (most nodes carry
   a terminal arc) it is ~100x faster than Dinic (the hold box at 4^3 tiles: 20 s -> 0.1 s per move). */
enum { BK_FREE = 0, BK_S = 1, BK_T = 2 };
#define BK_TERM ((int64_t)-1)
#define BK_NONE ((int64_t)-2)
typedef struct {
    uint32_t n; const uint32_t *ea, *eb; int64_t *rc, *tr; const size_t *first, *adj;
    uint8_t *tree, *inq; int64_t *par; uint32_t *ts, *dist, *q; size_t qh, qn; uint32_t *orph; size_t no, cap_o; uint32_t now;
} bk_graph;
static inline uint32_t bk_head(const bk_graph *g, size_t a) { return a & 1 ? g->ea[a >> 1] : g->eb[a >> 1]; }
static inline void bk_activate(bk_graph *g, uint32_t u) { if (g->inq[u]) return; g->inq[u] = 1; g->q[(g->qh + g->qn++) % (g->n + 1)] = u; }
static inline void bk_orphan(bk_graph *g, uint32_t u) { g->par[u] = BK_NONE; g->orph[g->no++] = u; }
static int64_t bk_run(bk_graph *g) {
    const uint32_t n = g->n; int64_t flow = 0; g->qh = g->qn = 0; g->no = 0; g->now = 0;
    for (uint32_t u = 0; u < n; u++) {
        g->inq[u] = 0; g->ts[u] = 0; g->dist[u] = 0;
        if (g->tr[u] > 0) { g->tree[u] = BK_S; g->par[u] = BK_TERM; g->dist[u] = 1; bk_activate(g, u); }
        else if (g->tr[u] < 0) { g->tree[u] = BK_T; g->par[u] = BK_TERM; g->dist[u] = 1; bk_activate(g, u); }
        else { g->tree[u] = BK_FREE; g->par[u] = BK_NONE; }
    }
    uint32_t cur = UINT32_MAX;   /* the active node being grown (kept while it still finds paths) */
    for (;;) {
        if (cur == UINT32_MAX || !g->tree[cur]) {
            cur = UINT32_MAX;
            while (g->qn) { const uint32_t u = g->q[g->qh]; g->qh = (g->qh + 1) % (n + 1); g->qn--; g->inq[u] = 0; if (g->tree[u]) { cur = u; break; } }
            if (cur == UINT32_MAX) break;
        }
        /* growth */
        const uint32_t u = cur; size_t mid = SIZE_MAX;
        for (size_t k = g->first[u]; k < g->first[u + 1] && mid == SIZE_MAX; k++) {
            const size_t a = g->adj[k]; const uint32_t v = bk_head(g, a);
            if (g->tree[u] == BK_S) {
                if (g->rc[a] <= 0) continue;
                if (!g->tree[v]) { g->tree[v] = BK_S; g->par[v] = (int64_t)(a ^ 1); g->ts[v] = g->ts[u]; g->dist[v] = g->dist[u] + 1; bk_activate(g, v); }
                else if (g->tree[v] == BK_T) mid = a;
                else if (g->ts[v] <= g->ts[u] && g->dist[v] > g->dist[u]) { g->par[v] = (int64_t)(a ^ 1); g->ts[v] = g->ts[u]; g->dist[v] = g->dist[u] + 1; }
            } else {
                if (g->rc[a ^ 1] <= 0) continue;
                if (!g->tree[v]) { g->tree[v] = BK_T; g->par[v] = (int64_t)(a ^ 1); g->ts[v] = g->ts[u]; g->dist[v] = g->dist[u] + 1; bk_activate(g, v); }
                else if (g->tree[v] == BK_S) mid = a ^ 1;
                else if (g->ts[v] <= g->ts[u] && g->dist[v] > g->dist[u]) { g->par[v] = (int64_t)(a ^ 1); g->ts[v] = g->ts[u]; g->dist[v] = g->dist[u] + 1; }
            }
        }
        if (mid == SIZE_MAX) { cur = UINT32_MAX; continue; }   /* u is exhausted */
        g->now++;
        /* augment along source ... x -mid-> y ... sink (S tree: par[v] = arc v -> parent, its flow arc the reverse; T tree: par[v] itself) */
        const uint32_t x = bk_head(g, mid ^ 1), y = bk_head(g, mid); int64_t f = g->rc[mid];
        uint32_t v = x; while (g->par[v] != BK_TERM) { const size_t a = (size_t)g->par[v]; if (g->rc[a ^ 1] < f) f = g->rc[a ^ 1]; v = bk_head(g, a); }
        if (g->tr[v] < f) f = g->tr[v];
        v = y; while (g->par[v] != BK_TERM) { const size_t a = (size_t)g->par[v]; if (g->rc[a] < f) f = g->rc[a]; v = bk_head(g, a); }
        if (-g->tr[v] < f) f = -g->tr[v];
        g->rc[mid] -= f; g->rc[mid ^ 1] += f;
        v = x; for (;;) {
            if (g->par[v] == BK_TERM) { g->tr[v] -= f; if (!g->tr[v]) bk_orphan(g, v); break; }
            const size_t a = (size_t)g->par[v]; const uint32_t p = bk_head(g, a);
            g->rc[a ^ 1] -= f; g->rc[a] += f; if (!g->rc[a ^ 1]) bk_orphan(g, v); v = p;
        }
        v = y; for (;;) {
            if (g->par[v] == BK_TERM) { g->tr[v] += f; if (!g->tr[v]) bk_orphan(g, v); break; }
            const size_t a = (size_t)g->par[v]; const uint32_t p = bk_head(g, a);
            g->rc[a] -= f; g->rc[a ^ 1] += f; if (!g->rc[a]) bk_orphan(g, v); v = p;
        }
        flow += f;
        /* adoption */
        while (g->no) {
            const uint32_t o = g->orph[--g->no]; const uint8_t X = g->tree[o];
            int64_t best = BK_NONE; uint32_t dmin = UINT32_MAX;
            for (size_t k = g->first[o]; k < g->first[o + 1]; k++) {
                const size_t a = g->adj[k]; const uint32_t qn = bk_head(g, a);
                if (g->tree[qn] != X || (X == BK_S ? g->rc[a ^ 1] : g->rc[a]) <= 0) continue;
                uint32_t d = 0, j = qn;   /* qn's origin: a terminal (valid) or an orphan */
                for (;;) {
                    if (g->ts[j] == g->now) { d += g->dist[j]; break; }
                    const int64_t pa = g->par[j]; d++;
                    if (pa == BK_TERM) { g->ts[j] = g->now; g->dist[j] = 1; break; }
                    if (pa == BK_NONE) { d = UINT32_MAX; break; }
                    j = bk_head(g, (size_t)pa);
                }
                if (d == UINT32_MAX) continue;
                if (d < dmin) { dmin = d; best = (int64_t)a; }
                for (j = qn; g->ts[j] != g->now; j = bk_head(g, (size_t)g->par[j])) { g->ts[j] = g->now; g->dist[j] = d--; }
            }
            if (best != BK_NONE) { g->par[o] = best; g->ts[o] = g->now; g->dist[o] = dmin + 1; continue; }
            for (size_t k = g->first[o]; k < g->first[o + 1]; k++) {   /* no parent: o becomes free, its children orphans */
                const size_t a = g->adj[k]; const uint32_t qn = bk_head(g, a);
                if (g->tree[qn] != X) continue;
                if ((X == BK_S ? g->rc[a ^ 1] : g->rc[a]) > 0) bk_activate(g, qn);
                if (g->par[qn] >= 0 && bk_head(g, (size_t)g->par[qn]) == o) { if (g->no == g->cap_o) abort(); bk_orphan(g, qn); }
            }
            g->tree[o] = BK_FREE;
        }
        if (!g->tree[cur]) cur = UINT32_MAX;
    }
    return flow;
}

int l1_solve(uint32_t n, const l1_vote *v, size_t nv, int32_t *s, int max_iter, l1_stats *st) {
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    l1_stats z = {0}; if (st) *st = z;
    if (!n) return 0;
    /* edges: runs of equal (a, b) */
    size_t ne = 0; for (size_t i = 0; i < nv; i++) if (!i || v[i].a != v[i - 1].a || v[i].b != v[i - 1].b) ne++;
    size_t *eo = malloc((ne + 1) * sizeof *eo);
    /* arcs: per edge a -> b and its reverse; per node source -> u and u -> sink (each with its reverse) */
    mf_graph g = {.N = (int)n + 2, .M = 2 * ne + 4 * (size_t)n};
    g.to = malloc(g.M * sizeof *g.to); g.cap = malloc(g.M * sizeof *g.cap); g.first = calloc((size_t)g.N + 1, sizeof *g.first); g.adj = malloc(g.M * sizeof *g.adj);
    g.lev = malloc((size_t)g.N * sizeof *g.lev); g.it = malloc((size_t)g.N * sizeof *g.it); g.q = malloc((size_t)g.N * sizeof *g.q); g.stk = malloc(((size_t)g.N + 1) * sizeof *g.stk);
    int64_t *un = malloc((size_t)n * sizeof *un); int32_t *s2 = malloc((size_t)n * sizeof *s2); uint8_t *mv = malloc(n);
    if (!eo || !g.to || !g.cap || !g.first || !g.adj || !g.lev || !g.it || !g.q || !g.stk || !un || !s2 || !mv) {
        free(eo); free(g.to); free(g.cap); free(g.first); free(g.adj); free(g.lev); free(g.it); free(g.q); free(g.stk); free(un); free(s2); free(mv); return -1;
    }
    ne = 0; for (size_t i = 0; i < nv; i++) if (!i || v[i].a != v[i - 1].a || v[i].b != v[i - 1].b) eo[ne++] = i;
    eo[ne] = nv;
    const uint32_t S = n, T = n + 1;
    for (size_t e = 0; e < ne; e++) { g.to[2 * e] = v[eo[e]].b; g.to[2 * e + 1] = v[eo[e]].a; }
    for (uint32_t u = 0; u < n; u++) { const size_t b = 2 * ne + 4 * (size_t)u; g.to[b] = u; g.to[b + 1] = S; g.to[b + 2] = T; g.to[b + 3] = u; }
    /* CSR by the arcs' tail (the head of the reverse arc) */
    for (size_t a = 0; a < g.M; a++) g.first[g.to[a ^ 1] + 1]++;
    for (int u = 0; u < g.N; u++) g.first[u + 1] += g.first[u];
    {
        size_t *fill = malloc((size_t)g.N * sizeof *fill); if (!fill) abort();
        memcpy(fill, g.first, (size_t)g.N * sizeof *fill);
        for (size_t a = 0; a < g.M; a++) g.adj[fill[g.to[a ^ 1]]++] = a;
        free(fill);
    }
    /* the BK graph: the same edges without the terminal arcs (node capacities instead); CSR of node arcs */
    const char *dn = getenv("UFSM_L1_DINIC"); const int dinic = dn && *dn == '1';   /* the reference solver (tests) */
    bk_graph bk = {.n = n};
    uint32_t *bea = malloc((ne ? ne : 1) * sizeof *bea), *beb = malloc((ne ? ne : 1) * sizeof *beb); size_t *bfirst = calloc((size_t)n + 1, sizeof *bfirst), *badj = malloc((2 * ne + 1) * sizeof *badj);
    bk.rc = malloc((2 * ne + 1) * sizeof *bk.rc); bk.tr = malloc((size_t)n * sizeof *bk.tr); bk.tree = malloc(n); bk.inq = malloc(n); bk.par = malloc((size_t)n * sizeof *bk.par);
    bk.ts = malloc((size_t)n * sizeof *bk.ts); bk.dist = malloc((size_t)n * sizeof *bk.dist); bk.q = malloc(((size_t)n + 1) * sizeof *bk.q); bk.orph = malloc((size_t)n * sizeof *bk.orph); bk.cap_o = n;
    if (!bea || !beb || !bfirst || !badj || !bk.rc || !bk.tr || !bk.tree || !bk.inq || !bk.par || !bk.ts || !bk.dist || !bk.q || !bk.orph) abort();
    for (size_t e = 0; e < ne; e++) { bea[e] = v[eo[e]].a; beb[e] = v[eo[e]].b; bfirst[bea[e] + 1]++; bfirst[beb[e] + 1]++; }
    for (uint32_t u = 0; u < n; u++) bfirst[u + 1] += bfirst[u];
    { size_t *fill = malloc(((size_t)n + 1) * sizeof *fill); if (!fill) abort(); memcpy(fill, bfirst, ((size_t)n + 1) * sizeof *fill);
      for (size_t e = 0; e < ne; e++) { badj[fill[bea[e]]++] = 2 * e; badj[fill[beb[e]]++] = 2 * e + 1; } free(fill); }
    bk.ea = bea; bk.eb = beb; bk.first = bfirst; bk.adj = badj;
    const int64_t e0 = l1_energy(v, nv, s); int64_t ecur = e0; int iters = 0, moves = 0; size_t nmoved = 0;
    /* +1 and -1 moves alternate; stop when neither lowers E */
    int fails = 0;
    for (int sgn = 1; iters < max_iter && fails < 2; sgn = -sgn, iters++) {
        memset(un, 0, (size_t)n * sizeof *un);
        for (size_t e = 0; e < ne; e++) {
            const uint32_t a = v[eo[e]].a, b = v[eo[e]].b; const int64_t t = (int64_t)s[b] - s[a];
            int64_t A = 0, B = 0, C = 0;   /* f(t), f(t + sgn) (b moved), f(t - sgn) (a moved) */
            for (size_t k = eo[e]; k < eo[e + 1]; k++) { A += v[k].w * iabs64(t - v[k].d); B += v[k].w * iabs64(t + sgn - v[k].d); C += v[k].w * iabs64(t - sgn - v[k].d); }
            /* E = A + (B - A) [x_a = 0, x_b = 1] + (C - A) [x_a = 1, x_b = 0] (B + C >= 2 A: f is convex): two arcs without terminal ones
               while B, C >= A (the votes agree); else [1, 0] = x_a - x_b + [0, 1] (C < A) or [0, 1] = x_b - x_a + [1, 0] (B < A) moves
               the negative part onto the nodes. (Unaries on every node made BK push one tiny path per edge: 3 s per move -> 0.1 s.) */
            if (C < A) { un[a] += C - A; un[b] += A - C; g.cap[2 * e] = B + C - 2 * A; g.cap[2 * e + 1] = 0; }
            else if (B < A) { un[b] += B - A; un[a] += A - B; g.cap[2 * e] = 0; g.cap[2 * e + 1] = B + C - 2 * A; }
            else { g.cap[2 * e] = B - A; g.cap[2 * e + 1] = C - A; }
        }
        int64_t base = 0;
        for (uint32_t u = 0; u < n; u++) {
            const size_t b = 2 * ne + 4 * (size_t)u;
            g.cap[b] = un[u] > 0 ? un[u] : 0; g.cap[b + 1] = 0;          /* source -> u: paid when u moves */
            g.cap[b + 2] = un[u] < 0 ? -un[u] : 0; g.cap[b + 3] = 0;     /* u -> sink: paid when u stays */
            if (un[u] < 0) base += un[u];
        }
        int64_t flow;
        if (dinic) {
            flow = mf_run(&g);
            if (base + flow >= 0) { fails++; continue; }
            /* the moved set: nodes the source cannot reach in the residual graph */
            for (int i = 0; i < g.N; i++) g.lev[i] = -1;
            size_t qh = 0, qt = 0; g.q[qt++] = S; g.lev[S] = 0;
            while (qh < qt) { const uint32_t u = g.q[qh++]; for (size_t k = g.first[u]; k < g.first[u + 1]; k++) { const size_t a = g.adj[k]; const uint32_t w = g.to[a]; if (g.cap[a] > 0 && g.lev[w] < 0) { g.lev[w] = 0; g.q[qt++] = w; } } }
            for (uint32_t u = 0; u < n; u++) mv[u] = g.lev[u] < 0;
        } else {
            for (size_t e = 0; e < ne; e++) { bk.rc[2 * e] = g.cap[2 * e]; bk.rc[2 * e + 1] = g.cap[2 * e + 1]; }
            for (uint32_t u = 0; u < n; u++) bk.tr[u] = un[u] > 0 ? un[u] : un[u] < 0 ? un[u] : 0;
            flow = bk_run(&bk);
            if (base + flow >= 0) { fails++; continue; }
            for (uint32_t u = 0; u < n; u++) mv[u] = bk.tree[u] != BK_S;   /* the source side: the S tree */
        }
        size_t nm = 0; for (uint32_t u = 0; u < n; u++) { s2[u] = s[u] + (mv[u] ? sgn : 0); nm += mv[u]; }
        const int64_t enew = l1_energy(v, nv, s2);
        if (enew >= ecur || !nm) { fails++; continue; }   /* (guards rounding of the construction) */
        memcpy(s, s2, (size_t)n * sizeof *s); ecur = enew; moves++; nmoved += nm; fails = 0;
    }
    struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
    if (st) { st->iters = iters; st->moves = moves; st->e0 = e0; st->e1 = ecur; st->nedge = ne; st->nmoved = nmoved; st->secs = (double)(t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9; }
    free(eo); free(g.to); free(g.cap); free(g.first); free(g.adj); free(g.lev); free(g.it); free(g.q); free(g.stk); free(un); free(s2); free(mv);
    free(bea); free(beb); free(bfirst); free(badj); free(bk.rc); free(bk.tr); free(bk.tree); free(bk.inq); free(bk.par); free(bk.ts); free(bk.dist); free(bk.q); free(bk.orph);
    return 0;
}
