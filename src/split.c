#include "split.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int kind; const void *p; shape5 s; int esz, h; double *b; int n; int slot; } coll_t;   /* kind 1 halo, 2 reduce */

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
    for (int i = 0; i < 2; i++) { nn_init(c->dev[i]); for (int k = 0; k < 2; k++) { nn_free(c->sb[k][i]); nn_free(c->rb[k][i]); } nn_free(c->rd[i]); }
    pthread_mutex_destroy(&c->mu); pthread_cond_destroy(&c->cv);
    free(c);
}

/* with mu held: hand the turn to the other side and wait for it to come back */
static void pass_and_wait(split_ctx *c, int me) {
    c->turn = 1 - me;
    pthread_cond_broadcast(&c->cv);
    while (c->turn != me) pthread_cond_wait(&c->cv, &c->mu);
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
    if (a->kind != b->kind || (a->kind == 1 && (memcmp(&a->s, &b->s, sizeof a->s) || a->esz != b->esz || a->h != b->h || a->slot != b->slot)) || (a->kind == 2 && a->n != b->n)) {
        fprintf(stderr, "split: the two halves diverged (collective %u: kind %d / %d)\n", c->seq, a->kind, b->kind);
        abort();
    }
    if (a->kind == 1) {
        size_t nb = nn_split_halo_bytes(a->s, a->esz);
        if (nb > c->cap) grow(c, nb, 0);
        void *t[2] = {(void *)a->p, (void *)b->p};
        nn_split_halo_begin(t, c->dev, a->s, a->esz, a->h, c->sb[a->slot], c->rb[a->slot], a->slot);
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
void split_reduce(double *b, int n) { coll_t x = {2, nullptr, {0}, 0, 0, b, n, 0}; collective(&x); }

typedef struct { split_ctx *c; int side; } warg_t;
static void *worker(void *p) {
    warg_t *w = p;
    split_ctx *c = w->c;
    const int me = w->side;
    t_ctx = c; t_side = me;
    nn_init(c->dev[me]);
    if (c->dev[0] == c->dev[1]) nn_split_thread_slot(me);
    pthread_mutex_lock(&c->mu);
    while (c->turn != me) pthread_cond_wait(&c->cv, &c->mu);
    pthread_mutex_unlock(&c->mu);
    c->job(me, c->arg);
    pthread_mutex_lock(&c->mu);
    c->done[me] = 1;
    if (c->narrived) { fprintf(stderr, "split: side %d finished while side %d waits at a collective\n", me, 1 - me); abort(); }
    if (!c->done[1 - me]) { c->turn = 1 - me; pthread_cond_broadcast(&c->cv); }
    pthread_mutex_unlock(&c->mu);
    t_ctx = nullptr; t_side = -1;
    nn_split_thread_slot(-1);
    return nullptr;
}

void split_run(split_ctx *c, void (*job)(int, void *), void *arg) {
    c->job = job; c->arg = arg; c->turn = 0; c->done[0] = c->done[1] = 0; c->narrived = 0;
    pthread_t th[2]; warg_t wa[2];
    for (int i = 0; i < 2; i++) { wa[i] = (warg_t){c, i}; pthread_create(&th[i], nullptr, worker, &wa[i]); }
    for (int i = 0; i < 2; i++) pthread_join(th[i], nullptr);
}
