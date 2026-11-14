#include "split.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int kind; const void *p; shape5 s; int esz, h; double *b; int n; int slot; void *q; size_t ns, nr; void *st; } coll_t;   /* kind 1 halo, 2 reduce, 3 k halo planes (n = k),
                                                    4 xchg (p send of ns bytes, q receive of nr), 5 typed sum (q values, n of them, slot = type), 6 split_top gather (n = the whole depth),
                                                    7 yield (nothing exchanged: the turn changes hands) */

struct split_ctx {
    int dev[2];
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int turn, done[2], narrived;
    unsigned seq;                 /* collectives executed */
    coll_t slot[2];
    void (*job)(int, void *); void *arg;
    void *sb[2][2], *rb[2][2]; size_t cap;   /* halo send / receive buffers per slot (0 asynchronous, 1 synchronous) and device */
    double *rd[2]; int rcap;             /* reduce receive buffers per device */
    void *rt[2]; int rtcap;              /* typed-sum scratch per device (8-byte values) */
    nn_tls tls;                          /* the op state (layer ids, formats, SR counter: thread-local in nn) handed on with the turn, so both sides
                                            and the caller see one sequence of it as when the globals were shared */
    /* the two side threads persist across split_run calls (2026-11-13: a pair of new threads per step leaked the nn modules' thread-local
       device buffers every step -- the ink MIL scratch, GroupNorm partials --, ~0.3 GB per step per GPU at P 1024 until a cudaMalloc failed) */
    pthread_t th[2]; int nth, quit; unsigned gen;
};

static _Thread_local split_ctx *t_ctx;
static _Thread_local int t_side = -1;

split_ctx *split_create(int dev0, int dev1) {
    split_ctx *c = calloc(1, sizeof *c);
    c->dev[0] = dev0; c->dev[1] = dev1;
    pthread_mutex_init(&c->mu, nullptr); pthread_cond_init(&c->cv, nullptr);
    nn_split_set_reduce(split_reduce);
    return c;
}
void split_free(split_ctx *c) {
    if (!c) return;
    if (c->nth) { pthread_mutex_lock(&c->mu); c->quit = 1; pthread_cond_broadcast(&c->cv); pthread_mutex_unlock(&c->mu); for (int i = 0; i < 2; i++) pthread_join(c->th[i], nullptr); c->nth = 0; }
    for (int i = 0; i < 2; i++) { nn_init(c->dev[i]); for (int k = 0; k < 2; k++) { nn_free(c->sb[k][i]); nn_free(c->rb[k][i]); } nn_free(c->rd[i]); nn_free(c->rt[i]); }
    pthread_mutex_destroy(&c->mu); pthread_cond_destroy(&c->cv);
    free(c);
}

/* with mu held: hand the turn to the other side and wait for it to come back */
static void pass_and_wait(split_ctx *c, int me) {
    nn_tls_capture(&c->tls);
    c->turn = 1 - me;
    pthread_cond_broadcast(&c->cv);
    while (c->turn != me) pthread_cond_wait(&c->cv, &c->mu);
    nn_tls_install(&c->tls);
}

static void grow(split_ctx *c, size_t bytes, int nd) {
    for (int i = 0; i < 2; i++) {
        nn_init(c->dev[i]);
        if (bytes > c->cap) for (int k = 0; k < 2; k++) { nn_free(c->sb[k][i]); nn_free(c->rb[k][i]); c->sb[k][i] = nn_malloc(bytes); c->rb[k][i] = nn_malloc(bytes); if (!c->sb[k][i] || !c->rb[k][i]) { fprintf(stderr, "split: out of device memory for the %.1f MB halo buffers on GPU %d\n", bytes / 1e6, c->dev[i]); abort(); } }
        if (nd > c->rcap) { nn_free(c->rd[i]); c->rd[i] = nn_malloc((size_t)nd * sizeof(double)); if (!c->rd[i]) { fprintf(stderr, "split: out of device memory on GPU %d\n", c->dev[i]); abort(); } }
    }
    if (bytes > c->cap) c->cap = bytes;
    if (nd > c->rcap) c->rcap = nd;
    nn_init(c->dev[t_side]);
}

static void execute(split_ctx *c) {
    coll_t *a = &c->slot[0], *b = &c->slot[1];
    if (a->kind != b->kind || ((a->kind == 1 || a->kind == 3 || a->kind == 6) && (memcmp(&a->s, &b->s, sizeof a->s) || a->esz != b->esz || a->h != b->h || a->slot != b->slot)) || ((a->kind == 2 || a->kind == 5 || a->kind == 6) && (a->n != b->n || a->slot != b->slot))) {
        fprintf(stderr, "split: the two halves diverged (collective %u: kind %d / %d)\n", c->seq, a->kind, b->kind);
        abort();
    }
    if (a->kind == 7) {   /* split_yield */
    } else if (a->kind == 1) {
        size_t nb = nn_split_halo_bytes(a->s, a->esz);
        if (nb > c->cap) grow(c, nb, 0);
        void *t[2] = {(void *)a->p, (void *)b->p};
        nn_split_halo_begin(t, c->dev, a->s, a->esz, a->h, c->sb[a->slot], c->rb[a->slot], a->slot);
    } else if (a->kind == 4) {   /* a pairwise exchange: each side's receive takes the other's send */
        if (a->ns != b->nr || b->ns != a->nr) { fprintf(stderr, "split: xchg sizes differ (collective %u: %zu -> %zu, %zu -> %zu)\n", c->seq, a->ns, b->nr, b->ns, a->nr); abort(); }
        const void *snd[2] = {a->p, b->p}; void *rcv[2] = {a->q, b->q}; const size_t nr[2] = {a->nr, b->nr}; void *st[2] = {a->st, b->st};
        nn_split_xchg(c->dev, snd, rcv, nr, st);
    } else if (a->kind == 5) {
        if (a->n > c->rtcap) { for (int i = 0; i < 2; i++) { nn_init(c->dev[i]); nn_free(c->rt[i]); c->rt[i] = nn_malloc((size_t)a->n * 8); if (!c->rt[i]) { fprintf(stderr, "split: out of device memory on GPU %d\n", c->dev[i]); abort(); } }
                               c->rtcap = a->n; nn_init(c->dev[t_side]); }
        void *bb[2] = {a->q, b->q}, *st[2] = {a->st, b->st};
        nn_split_sum_t(bb, c->dev, a->n, a->slot, c->rt, st);
    } else if (a->kind == 6) {   /* split_top gather (coll_t.n = the whole depth), the synchronous buffers */
        size_t nb = (size_t)(a->n / 2) * nn_split_halo_bytes(a->s, a->esz);
        if (nb > c->cap) grow(c, nb, 0);
        void *t[2] = {(void *)a->p, (void *)b->p};
        nn_split_gather(t, c->dev, a->s, a->esz, a->h, a->n, c->sb[1], c->rb[1]);
    } else if (a->kind == 3) {   /* k halo planes (coll_t.n = k), the synchronous buffers */
        size_t nb = (size_t)a->n * nn_split_halo_bytes(a->s, a->esz);
        if (nb > c->cap) grow(c, nb, 0);
        void *t[2] = {(void *)a->p, (void *)b->p};
        nn_split_halo_k(t, c->dev, a->s, a->esz, a->h, a->n, c->sb[1], c->rb[1]);
    } else {
        if (a->n > c->rcap) grow(c, 0, a->n);
        double *bb[2] = {a->b, b->b};
        nn_split_allreduce(bb, c->dev, a->n, c->rd);
    }
    c->seq++;
}

static void collective(const coll_t *x) {
    split_ctx *c = t_ctx;
    if (!c) { fprintf(stderr, "split: collective outside split_run\n"); abort(); }
    const int me = t_side;
    pthread_mutex_lock(&c->mu);
    c->slot[me] = *x;
    if (++c->narrived == 1) {
        if (c->done[1 - me]) { fprintf(stderr, "split: side %d finished while side %d waits at a collective\n", 1 - me, me); abort(); }
        const unsigned s0 = c->seq;
        pass_and_wait(c, me);
        if (c->seq != s0 + 1) { fprintf(stderr, "split: the two halves diverged (side %d resumed without its collective)\n", me); abort(); }
        pthread_mutex_unlock(&c->mu);
        return;
    }
    c->narrived = 0;
    pthread_mutex_unlock(&c->mu);
    execute(c);   /* this thread holds the turn; the other side waits */
}

static void halo_begin(const void *p, shape5 s, int esz, int h, int slot) { coll_t x = {1, p, s, esz, h, nullptr, 0, slot}; collective(&x); }
void split_halo_begin(const void *p, shape5 s, int esz, int h) { halo_begin(p, s, esz, h, 0); }
void split_halo_end(const void *p, shape5 s, int esz, int h) { nn_split_halo_end((void *)p, t_side, s, esz, h, t_ctx->rb[0][t_side], 0); }
void split_halo(const void *p, shape5 s, int esz, int h) { halo_begin(p, s, esz, h, 1); nn_split_halo_end((void *)p, t_side, s, esz, h, t_ctx->rb[1][t_side], 1); }
void split_halo_k(const void *p, shape5 s, int esz, int h, int k) { coll_t x = {3, p, s, esz, h, nullptr, k, 1}; collective(&x); }
void split_xchg(const void *send_d, size_t ns, void *recv_d, size_t nr, void *stream) { coll_t x = {4, send_d, {0}, 0, 0, nullptr, 0, 0, recv_d, ns, nr, stream}; collective(&x); }
void split_sum(void *d, int n, int type, void *stream) { coll_t x = {5, nullptr, {0}, 0, 0, nullptr, n, type, d, 0, 0, stream}; collective(&x); }
void split_gather(const void *p, shape5 s, int esz, int h, int D) { coll_t x = {6, p, s, esz, h, nullptr, D, 1}; collective(&x); }
void split_yield(void) { coll_t x = {7}; collective(&x); }
int split_side(void) { return t_side; }
void split_reduce(double *b, int n) { coll_t x = {2, nullptr, {0}, 0, 0, b, n, 0}; collective(&x); }

typedef struct { split_ctx *c; int side; } warg_t;
static warg_t g_wa[64][2]; static int g_nwa; static pthread_mutex_t g_wamu = PTHREAD_MUTEX_INITIALIZER;   /* (the workers' arguments: live as long as they run) */
static void *worker(void *p) {
    warg_t *w = p;
    split_ctx *c = w->c;
    const int me = w->side;
    t_ctx = c; t_side = me;
    unsigned seen = 0;
    pthread_mutex_lock(&c->mu);
    for (;;) {
        while (c->gen == seen && !c->quit) pthread_cond_wait(&c->cv, &c->mu);
        if (c->quit) break;
        seen = c->gen;
        while (c->turn != me) pthread_cond_wait(&c->cv, &c->mu);
        nn_init(c->dev[me]);
        if (c->dev[0] == c->dev[1]) nn_split_thread_slot(me);
        nn_tls_install(&c->tls);
        pthread_mutex_unlock(&c->mu);
        c->job(me, c->arg);
        pthread_mutex_lock(&c->mu);
        nn_tls_capture(&c->tls);
        c->done[me] = 1;
        if (c->narrived) { fprintf(stderr, "split: side %d finished while side %d waits at a collective\n", me, 1 - me); abort(); }
        if (!c->done[1 - me]) c->turn = 1 - me;
        pthread_cond_broadcast(&c->cv);   /* (the other side's turn, or split_run: both done) */
    }
    pthread_mutex_unlock(&c->mu);
    t_ctx = nullptr; t_side = -1;
    nn_split_thread_slot(-1);
    return nullptr;
}

void split_run(split_ctx *c, void (*job)(int, void *), void *arg) {
    pthread_mutex_lock(&c->mu);
    c->job = job; c->arg = arg; c->turn = 0; c->done[0] = c->done[1] = 0; c->narrived = 0;
    nn_tls_capture(&c->tls);
    if (!c->nth) {
        pthread_mutex_lock(&g_wamu); const int k = g_nwa++ % 64; pthread_mutex_unlock(&g_wamu);
        for (int i = 0; i < 2; i++) { g_wa[k][i] = (warg_t){c, i}; pthread_create(&c->th[i], nullptr, worker, &g_wa[k][i]); }
        c->nth = 2;
    }
    c->gen++;
    pthread_cond_broadcast(&c->cv);
    while (!(c->done[0] && c->done[1])) pthread_cond_wait(&c->cv, &c->mu);
    pthread_mutex_unlock(&c->mu);
    nn_tls_install(&c->tls);
}
void dp_grad_avg(float *const *g, const int *dev, int ng, size_t n, float *tmp, size_t chunk, int avg) {
    nn_init(dev[0]);
    for (int k = 1; k < ng; k++) for (size_t o = 0; o < n; o += chunk) {
        const size_t c = n - o < chunk ? n - o : chunk;
        nn_peer_copy(tmp, dev[0], g[k] + o, dev[k], c * 4); nn_axpy(g[0] + o, 1.f, tmp, c);
    }
    if (avg) nn_scale(g[0], 1.f / ng, n);
    for (int k = 1; k < ng; k++) nn_peer_copy(g[k], dev[k], g[0], dev[0], n * 4);
}
