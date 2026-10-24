#include "store.h"
#include <curl/curl.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <stdatomic.h>

struct store {
    char *root;      /* without trailing '/' */
    int local;
    char *bearer;    /* nullptr or "Authorization: Bearer ..." */
    pthread_mutex_t mu;
    /* request pacing: a token bucket at min(rps, acap) requests/s holding up to burst; 0 = no limit */
    double rps;           /* fixed cap: store_set_rate, HuggingFace 14/s, UFSM_HTTP_RPS */
    double acap;          /* adaptive cap: halved from the observed rate on a 429 / 503, +10%/s while none come, dropped past 4x */
    double burst, tokens, t_fill, t_inc, t_cut;
    double rate_obs, t_rate; int n_start;   /* observed request starts per second */
    double pause_until;   /* 429 / 503 back-off shared by every thread of this store */
    int strikes; double t_strike;
};

static double mono(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* ---- accounting ---- */
static atomic_uint_fast64_t st_requests, st_bytes, st_429, st_503, st_5xx, st_transport, st_retries, st_rl, st_stalls, st_us, st_sleep_us, st_max_us, st_connects;
static pthread_mutex_t st_mu = PTHREAD_MUTEX_INITIALIZER;
static char st_last_rl[192];
static int http_log = -1;
static int log_on(void) { if (http_log < 0) { const char *v = getenv("UFSM_HTTP_LOG"); http_log = v && *v && strcmp(v, "0"); } return http_log; }
void store_stats_get(store_stats_t *o) {
    *o = (store_stats_t){atomic_load(&st_requests), atomic_load(&st_bytes), atomic_load(&st_429), atomic_load(&st_503), atomic_load(&st_5xx),
                         atomic_load(&st_transport), atomic_load(&st_retries), atomic_load(&st_rl), atomic_load(&st_stalls), atomic_load(&st_connects),
                         atomic_load(&st_us) * 1e-6, atomic_load(&st_sleep_us) * 1e-6, atomic_load(&st_max_us) * 1e-6, {0}};
    pthread_mutex_lock(&st_mu); memcpy(o->last_ratelimit, st_last_rl, sizeof st_last_rl); pthread_mutex_unlock(&st_mu);
}
void store_stats_reset(void) {
    atomic_uint_fast64_t *all[] = {&st_requests, &st_bytes, &st_429, &st_503, &st_5xx, &st_transport, &st_retries, &st_rl, &st_stalls, &st_us, &st_sleep_us, &st_max_us, &st_connects};
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) atomic_store(all[i], 0);
    pthread_mutex_lock(&st_mu); st_last_rl[0] = 0; pthread_mutex_unlock(&st_mu);
}
static double t_origin;
static double since_start(void) { if (!t_origin) t_origin = mono(); return mono() - t_origin; }
/* one finished transfer: bytes, time, stalls, status classes */
static void account(CURL *c, CURLcode rc, long code, int rl_seen) {
    curl_off_t dl = 0, us = 0;
    curl_easy_getinfo(c, CURLINFO_SIZE_DOWNLOAD_T, &dl);
    curl_easy_getinfo(c, CURLINFO_TOTAL_TIME_T, &us);
    long nc = 0; curl_easy_getinfo(c, CURLINFO_NUM_CONNECTS, &nc); atomic_fetch_add(&st_connects, (uint64_t)(nc > 0 ? nc : 0));
    atomic_fetch_add(&st_requests, 1); atomic_fetch_add(&st_bytes, (uint64_t)(dl > 0 ? dl : 0)); atomic_fetch_add(&st_us, (uint64_t)(us > 0 ? us : 0));
    uint64_t prev = atomic_load(&st_max_us);
    while ((uint64_t)us > prev && !atomic_compare_exchange_weak(&st_max_us, &prev, (uint64_t)us)) {}
    if (us > 2000000 && dl * 1000000.0 / us < 65536) atomic_fetch_add(&st_stalls, 1);
    if (rl_seen) atomic_fetch_add(&st_rl, 1);
    if (code == 429) atomic_fetch_add(&st_429, 1);
    else if (code == 503) atomic_fetch_add(&st_503, 1);
    else if (code >= 500) atomic_fetch_add(&st_5xx, 1);
    if (rc != CURLE_OK && code < 400) atomic_fetch_add(&st_transport, 1);
}

/* token bucket step (s->mu held): returns seconds to wait (0 = go). reserve: take the token even when it must be waited for. */
static double bucket(store *s, double now, int reserve) {
    if (++s->n_start, now - s->t_rate >= 1) { s->rate_obs = s->n_start / (now - s->t_rate + 1e-9); s->n_start = 0; s->t_rate = now; }
    if (s->acap > 0 && now - s->t_inc >= 1 && now - s->t_strike >= 2) {   /* additive recovery */
        s->acap *= 1.1; s->t_inc = now;
        if (s->acap > 4 * (s->rate_obs > 1 ? s->rate_obs : 1) && s->acap > 400) s->acap = 0;
    }
    double r = s->rps > 0 && (s->acap <= 0 || s->rps < s->acap) ? s->rps : s->acap;
    double wait = s->pause_until > now ? s->pause_until - now : 0;
    if (r <= 0) { if (wait > 0 && !reserve) s->n_start--; return wait; }
    s->tokens += (now - s->t_fill) * r; s->t_fill = now;
    if (s->tokens > s->burst) s->tokens = s->burst;
    if (!reserve && (wait > 0 || s->tokens < 1)) { s->n_start--; return wait > 0 ? wait : (1 - s->tokens) / r; }
    s->tokens -= 1;
    if (s->tokens < 0 && -s->tokens / r > wait) wait = -s->tokens / r;
    return wait;
}
/* wait for this store's pacing slot and any shared back-off */
static void pace(store *s) {
    pthread_mutex_lock(&s->mu);
    double wait = bucket(s, mono(), 1);
    pthread_mutex_unlock(&s->mu);
    if (wait > 0) { atomic_fetch_add(&st_sleep_us, (uint64_t)(wait * 1e6)); usleep((useconds_t)(wait * 1e6)); }
}
/* non-blocking pace for the multi loop: 0 = start now (token taken), else seconds to wait */
static double pace_try(store *s) {
    pthread_mutex_lock(&s->mu);
    double wait = bucket(s, mono(), 0);
    pthread_mutex_unlock(&s->mu);
    return wait;
}

void store_set_rate(store *s, double rps) { pthread_mutex_lock(&s->mu); s->rps = rps; s->burst = 1; s->tokens = 1; pthread_mutex_unlock(&s->mu); }

/* Rate-limit headers: "ratelimit: ...;t=NN" (IETF draft / HuggingFace), "retry-after: NN", "x-ratelimit-reset: NN";
   any header whose name contains "ratelimit" counts as seen. reset = seconds until the window resets (0 = unknown). */
static void parse_rl(const char *buf, size_t n, double *reset, int *seen) {
    const char *colon = memchr(buf, ':', n);
    if (!colon) return;
    size_t nl = (size_t)(colon - buf);
    char name[64]; if (nl >= sizeof name) return;
    for (size_t i = 0; i < nl; i++) name[i] = (char)(buf[i] >= 'A' && buf[i] <= 'Z' ? buf[i] + 32 : buf[i]);
    name[nl] = 0;
    if (!strstr(name, "ratelimit") && strcmp(name, "retry-after")) return;
    char value[160]; size_t vn = n - nl - 1; while (vn && (colon[vn] == '\r' || colon[vn] == '\n')) vn--;
    snprintf(value, sizeof value, "%.*s", (int)vn, colon + 1);
    *seen = 1;
    if (!strcmp(name, "ratelimit")) { char *t = strstr(value, ";t="); if (t) *reset = atof(t + 3); }
    else if (!strcmp(name, "retry-after") || !strcmp(name, "x-ratelimit-reset") || !strcmp(name, "ratelimit-reset")) {
        double v = atof(value); if (v > 0 && v < 3600) *reset = v;
    }
    pthread_mutex_lock(&st_mu); snprintf(st_last_rl, sizeof st_last_rl, "%s:%s", name, value); pthread_mutex_unlock(&st_mu);
    if (log_on()) fprintf(stderr, "store: t=%.1fs header %s:%s\n", since_start(), name, value);
}
static _Thread_local double tl_reset;
static _Thread_local int tl_rl;
static size_t on_ratelimit_hdr(char *buf, size_t sz, size_t nm, void *ud) {
    size_t n = sz * nm;
    parse_rl(buf, n, &tl_reset, &tl_rl);
    return n;
}

/* 429 / 503: every thread of the store waits (pace) until the window resets (header), else a short jittered exponential back-off
   (0.5 s doubling to 8 s over strikes within 5 s of each other); the adaptive cap halves the observed request rate (once a second). */
static void throttle(store *s, long code, int attempt, double reset) {
    pthread_mutex_lock(&s->mu);
    double now = mono();
    if (now < s->pause_until && !(reset > 0)) {   /* in flight before the current pause: the same event */
        pthread_mutex_unlock(&s->mu); atomic_fetch_add(&st_retries, 1); return;
    }
    s->strikes = now - s->t_strike < 5 ? s->strikes + 1 : 1; s->t_strike = now;
    double t = reset > 0 && reset < 600 ? reset + 1 : 0.5 * (1 << (s->strikes - 1 < 4 ? s->strikes - 1 : 4)) * (0.75 + 0.5 * (rand() / (double)RAND_MAX));
    double until = now + t;
    int fresh = until > s->pause_until + 0.5;
    if (until > s->pause_until) s->pause_until = until;
    if (now - s->t_cut >= 1) {
        double base = s->rate_obs > 1 ? s->rate_obs : 20;
        if (s->acap > 0 && s->acap < base) base = s->acap;
        s->acap = base * 0.5 > 2 ? base * 0.5 : 2; s->t_cut = now; s->t_inc = now;
        if (s->burst < 1) s->burst = 1;
    }
    double cap = s->acap;
    pthread_mutex_unlock(&s->mu);
    atomic_fetch_add(&st_retries, 1);
    if (fresh || log_on()) fprintf(stderr, "store: t=%.1fs HTTP %ld (%s), backing off %.1fs, rate cap %.0f/s\n", since_start(), code, code == 429 ? "rate limited" : "unavailable", t, cap);
}

static _Thread_local long tl_code;
long store_last_status(void) { return tl_code; }

typedef struct { CURL *curl; struct curl_slist *hdr; } thread_http;
static _Thread_local thread_http *tl_http;
static pthread_key_t http_key;
static pthread_once_t http_key_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t http_global_mu = PTHREAD_MUTEX_INITIALIZER;
/* Each handle has one owner; worker exit returns it to a bounded idle pool.
   Reset request options/headers, retaining only libcurl's connection/DNS/TLS caches. */
static int http_initialized, http_key_error, http_reuse;
enum { HTTP_IDLE_MAX = 32 };
static thread_http *http_idle[HTTP_IDLE_MAX];
static int http_idle_n;
static void http_destroy(thread_http *h) {
    if (!h) return;
    curl_easy_cleanup(h->curl);
    curl_slist_free_all(h->hdr);
    free(h);
}
static void http_thread_free(void *ptr) {
    thread_http *h = ptr;
    if (!h) return;
    curl_easy_reset(h->curl);
    curl_slist_free_all(h->hdr); h->hdr = nullptr;
    pthread_mutex_lock(&http_global_mu);
    if (http_initialized && http_reuse && http_idle_n < HTTP_IDLE_MAX) {
        http_idle[http_idle_n++] = h;
        h = nullptr;
    }
    pthread_mutex_unlock(&http_global_mu);
    http_destroy(h);
}
/* Per-thread multi handles (store_read_batch): persistent connections, HTTP/2 multiplexed. */
enum { MULTI_MAX = 512 };
typedef struct { CURLM *m; CURL *e[MULTI_MAX]; int ne; struct curl_slist *hdr; } thread_multi;
static _Thread_local thread_multi *tl_multi;
static pthread_key_t multi_key;
static void multi_destroy(void *p) {
    thread_multi *t = p;
    if (!t) return;
    for (int i = 0; i < t->ne; i++) { curl_multi_remove_handle(t->m, t->e[i]); curl_easy_cleanup(t->e[i]); }
    curl_multi_cleanup(t->m);
    curl_slist_free_all(t->hdr);
    free(t);
}
/* curl_share: DNS and TLS sessions shared by every handle (UFSM_HTTP_SHARE=1, default); =2 also shares the easy handles'
   connection cache (measured 2-3x slower at 8 readers: connection churn); 0 = off */
static CURLSH *g_share;
static int g_share_mode;
static pthread_mutex_t share_mu[CURL_LOCK_DATA_LAST];
static void share_lock(CURL *h, curl_lock_data d, curl_lock_access a, void *u) { pthread_mutex_lock(&share_mu[d]); }
static void share_unlock(CURL *h, curl_lock_data d, void *u) { pthread_mutex_unlock(&share_mu[d]); }
static void http_key_init(void) {
    http_key_error = pthread_key_create(&http_key, http_thread_free) || pthread_key_create(&multi_key, multi_destroy);
    for (int i = 0; i < CURL_LOCK_DATA_LAST; i++) pthread_mutex_init(&share_mu[i], nullptr);
}
static long env_long(const char *name, long def) { const char *v = getenv(name); return v && *v ? atol(v) : def; }
/* Process-wide cap on HTTP requests in flight (UFSM_HTTP_INFLIGHT, default 48; 0 = none). The volcomp mirror answers 429 once
   one IP has more than ~64-96 requests in flight (measured 2026-10-23: 64 concurrent 1 MB ranges pass, 96 draw 70% 429s;
   request rate itself is not limited), so readers with many threads, or several readers, queue here instead. */
static pthread_mutex_t inflight_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t inflight_cv = PTHREAD_COND_INITIALIZER;
static long inflight_n, inflight_max = -1;
static long inflight_cap(void) { if (inflight_max < 0) inflight_max = env_long("UFSM_HTTP_INFLIGHT", 48); return inflight_max; }
static void inflight_acquire(void) {
    if (inflight_cap() <= 0) return;
    pthread_mutex_lock(&inflight_mu);
    while (inflight_n >= inflight_max) pthread_cond_wait(&inflight_cv, &inflight_mu);
    inflight_n++;
    pthread_mutex_unlock(&inflight_mu);
}
static int inflight_try(void) {
    if (inflight_cap() <= 0) return 1;
    pthread_mutex_lock(&inflight_mu);
    int ok = inflight_n < inflight_max; if (ok) inflight_n++;
    pthread_mutex_unlock(&inflight_mu);
    return ok;
}
static void inflight_release(void) {
    if (inflight_cap() <= 0) return;
    pthread_mutex_lock(&inflight_mu); inflight_n--; pthread_cond_signal(&inflight_cv); pthread_mutex_unlock(&inflight_mu);
}
void store_global_init(void) {
    pthread_once(&http_key_once, http_key_init);
    pthread_mutex_lock(&http_global_mu);
    if (!http_initialized) {
        http_initialized = curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK;
        const char *reuse = getenv("UFSM_HTTP_REUSE");
        http_reuse = !reuse || atoi(reuse) != 0;
        g_share_mode = (int)env_long("UFSM_HTTP_SHARE", 1);
        if (http_initialized && g_share_mode > 0 && (g_share = curl_share_init())) {
            curl_share_setopt(g_share, CURLSHOPT_LOCKFUNC, share_lock);
            curl_share_setopt(g_share, CURLSHOPT_UNLOCKFUNC, share_unlock);
            curl_share_setopt(g_share, CURLSHOPT_SHARE, CURL_LOCK_DATA_DNS);
            curl_share_setopt(g_share, CURLSHOPT_SHARE, CURL_LOCK_DATA_SSL_SESSION);
            if (g_share_mode >= 2) curl_share_setopt(g_share, CURLSHOPT_SHARE, CURL_LOCK_DATA_CONNECT);
        }
    }
    pthread_mutex_unlock(&http_global_mu);
}
void store_global_cleanup(void) {
    /* Call after other store callers join; destroy every idle connection before global cleanup. */
    if (tl_http) {
        pthread_setspecific(http_key, nullptr);
        http_thread_free(tl_http);
        tl_http = nullptr;
    }
    if (tl_multi) { pthread_setspecific(multi_key, nullptr); multi_destroy(tl_multi); tl_multi = nullptr; }
    pthread_mutex_lock(&http_global_mu);
    while (http_idle_n) http_destroy(http_idle[--http_idle_n]);
    if (g_share && curl_share_cleanup(g_share) == CURLSHE_OK) g_share = nullptr;
    if (http_initialized && !g_share) { curl_global_cleanup(); http_initialized = 0; }
    pthread_mutex_unlock(&http_global_mu);
}

store *store_open(const char *root) {
    store *s = calloc(1, sizeof *s);
    if (!s) return nullptr;
    size_t n = strlen(root);
    while (n > 1 && root[n - 1] == '/') n--;
    s->root = strndup(root, n);
    if (!s->root) { free(s); return nullptr; }
    s->local = strncmp(root, "http://", 7) && strncmp(root, "https://", 8);
    if (!s->local) store_global_init();
    pthread_mutex_init(&s->mu, nullptr);
    s->rps = (double)env_long("UFSM_HTTP_RPS", 0);
    s->burst = s->tokens = (double)env_long("UFSM_HTTP_BURST", 32);
    if (s->burst < 1) s->burst = s->tokens = 1;
    s->t_fill = s->t_rate = mono();
    if (strstr(root, "huggingface.co")) { s->rps = 14; s->burst = s->tokens = 1; }
    return s;
}

void store_close(store *s) {
    if (!s) return;
    free(s->root);
    free(s->bearer);
    pthread_mutex_destroy(&s->mu);
    free(s);
}

int store_is_local(const store *s) { return s->local; }
void store_set_bearer(store *s, const char *token) {
    free(s->bearer);
    s->bearer = nullptr;
    if (token && *token) { size_t n = strlen(token) + 32; s->bearer = malloc(n); snprintf(s->bearer, n, "Authorization: Bearer %s", token); }
}
const char *store_root(const store *s) { return s->root; }

static char *join(const store *s, const char *key) {
    size_t a = strlen(s->root), b = strlen(key);
    char *p = malloc(a + b + 2);
    if (!p) return nullptr;
    memcpy(p, s->root, a);
    p[a] = '/';
    memcpy(p + a + 1, key, b + 1);
    return p;
}

/* ---- local ---- */

static int64_t local_size(const store *s, const char *key) {
    char *p = join(s, key);
    struct stat st;
    int r = p ? stat(p, &st) : -1;
    int e = errno;
    free(p);
    return r ? (e == ENOENT || e == ENOTDIR ? -2 : -1) : (int64_t)st.st_size;
}

static int64_t local_read(const store *s, const char *key, int64_t off, int64_t len, uint8_t *out) {
    char *p = join(s, key);
    int fd = p ? open(p, O_RDONLY) : -1;
    int e = errno;
    free(p);
    if (fd < 0) return e == ENOENT || e == ENOTDIR ? -2 : -1;
    int64_t got = 0;
    while (got < len) {
        ssize_t r = pread(fd, out + got, (size_t)(len - got), off + got);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
        got += r;
    }
    static int nocache = -1;   /* UFSM_STORE_NOCACHE=1: drop the pages read (a bulk pass over local data, e.g. ufsm scanprep, must not evict the page cache of jobs beside it) */
    if (nocache < 0) { const char *v = getenv("UFSM_STORE_NOCACHE"); nocache = v && *v == '1'; }
    if (nocache && got > 0) posix_fadvise(fd, off, got, POSIX_FADV_DONTNEED);
    close(fd);
    return got == len ? got : -1;
}

/* ---- http ---- */

typedef struct {
    uint8_t *buf; size_t n, cap; int fixed, overflow, has_range;
    uint64_t range_start, range_end, range_total; int range_total_known;
    int rl; double reset;   /* rate-limit header seen in this response, seconds until its window resets */
} sink;

static size_t on_data(char *ptr, size_t sz, size_t nm, void *ud) {
    sink *k = ud;
    size_t n = sz * nm;
    if (k->fixed) {
        if (n > k->cap - k->n) { k->overflow = 1; return 0; }
    } else if (k->n + n + 1 > k->cap) {
        size_t cap = k->cap ? k->cap * 2 : 65536;
        while (cap < k->n + n + 1) cap *= 2;
        uint8_t *b = realloc(k->buf, cap);
        if (!b) return 0;
        k->buf = b;
        k->cap = cap;
    }
    memcpy(k->buf + k->n, ptr, n);
    k->n += n;
    return sz * nm;
}

static size_t on_read_hdr(char *buf, size_t sz, size_t nm, void *ud) {
    sink *k = ud; size_t n = sz * nm;
    if (n >= 5 && !strncmp(buf, "HTTP/", 5)) { k->n = 0; k->has_range = 0; k->overflow = 0; }
    if (n > 14 && !strncasecmp(buf, "content-range:", 14)) {
        char value[128], tail; unsigned long long a, b, total;
        snprintf(value, sizeof value, "%.*s", (int)(n - 14), buf + 14);
        int fields = sscanf(value, " bytes %llu-%llu/%llu %c", &a, &b, &total, &tail);
        if (fields == 3) {
            k->has_range = 1; k->range_start = a; k->range_end = b; k->range_total = total; k->range_total_known = 1;
        } else if (sscanf(value, " bytes %llu-%llu/* %c", &a, &b, &tail) == 2) {
            k->has_range = 1; k->range_start = a; k->range_end = b; k->range_total_known = 0;
        }
    }
    parse_rl(buf, n, &k->reset, &k->rl);
    return n;
}

/* options every request gets (after curl_easy_reset) */
static void base_opts(CURL *c, struct curl_slist *hdr) {
    if (hdr) curl_easy_setopt(c, CURLOPT_HTTPHEADER, hdr);
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(c, CURLOPT_LOW_SPEED_TIME, 60L);
    curl_easy_setopt(c, CURLOPT_LOW_SPEED_LIMIT, 1024L);
    curl_easy_setopt(c, CURLOPT_USERAGENT, "ufsm/0.1");
    curl_easy_setopt(c, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(c, CURLOPT_FAILONERROR, 1L);
    static long bufsize = -1;
    if (bufsize < 0) bufsize = env_long("UFSM_HTTP_BUF", 0) << 10;   /* curl receive buffer, KB (0 = curl's 16 KB default) */
    if (bufsize > 0) curl_easy_setopt(c, CURLOPT_BUFFERSIZE, bufsize);
    static long ver = -1;
    if (ver < 0) ver = env_long("UFSM_HTTP_VERSION", 0);   /* 1: HTTP/1.1 only, 2: HTTP/2 over TLS (curl's default), 0: default */
    if (ver == 1) curl_easy_setopt(c, CURLOPT_HTTP_VERSION, (long)CURL_HTTP_VERSION_1_1);
    else if (ver == 2) curl_easy_setopt(c, CURLOPT_HTTP_VERSION, (long)CURL_HTTP_VERSION_2TLS);
}

static CURL *fresh_handle(const store *s) {
    if (!tl_http) {
        pthread_mutex_lock(&http_global_mu);
        int ready = http_initialized && !http_key_error;
        thread_http *h = ready && http_idle_n ? http_idle[--http_idle_n] : nullptr;
        pthread_mutex_unlock(&http_global_mu);
        if (!ready) return nullptr;
        if (!h) {
            h = calloc(1, sizeof *h);
            if (!h) return nullptr;
            h->curl = curl_easy_init();
        }
        if (!h->curl || pthread_setspecific(http_key, h)) { http_destroy(h); return nullptr; }
        tl_http = h;
    }
    CURL *c = tl_http->curl;
    curl_easy_reset(c);
    curl_slist_free_all(tl_http->hdr); tl_http->hdr = nullptr;
    if (s->bearer) tl_http->hdr = curl_slist_append(nullptr, s->bearer);
    base_opts(c, tl_http->hdr);
    curl_easy_setopt(c, CURLOPT_MAXCONNECTS, 2L);
    if (g_share) curl_easy_setopt(c, CURLOPT_SHARE, g_share);
    return c;
}

static double backoff_s(int attempt) { return 0.2 * (1 << (attempt < 5 ? attempt : 5)); }

/* GET (len < 0: whole object; tail: the last len bytes) with retries. Returns HTTP status, or -1 on transport failure. */
static long http_get(store *s, const char *key, int64_t off, int64_t len, int tail, sink *k) {
    char *url = join(s, key);
    if (!url) return -1;
    long code = -1;
    for (int attempt = 0; attempt < 12; attempt++) {
        pace(s);
        CURL *c = fresh_handle(s);
        if (!c) break;
        curl_easy_setopt(c, CURLOPT_URL, url);
        curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, on_data);
        curl_easy_setopt(c, CURLOPT_WRITEDATA, k);
        curl_easy_setopt(c, CURLOPT_HEADERFUNCTION, on_read_hdr);
        curl_easy_setopt(c, CURLOPT_HEADERDATA, k);
        char range[64];
        if (tail) {
            snprintf(range, sizeof range, "-%lld", (long long)len);
            curl_easy_setopt(c, CURLOPT_RANGE, range);
        } else if (len >= 0) {
            snprintf(range, sizeof range, "%lld-%lld", (long long)off, (long long)(off + len - 1));
            curl_easy_setopt(c, CURLOPT_RANGE, range);
        }
        k->n = 0; k->overflow = 0; k->has_range = 0; k->rl = 0; k->reset = 0;
        inflight_acquire();
        CURLcode rc = curl_easy_perform(c);
        inflight_release();
        code = 0;
        curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
        account(c, rc, code, k->rl);
        if (rc == CURLE_OK) break;
        if (k->overflow || rc == CURLE_WRITE_ERROR) { code = -1; break; }
        if (code == 429 || code == 503) { throttle(s, code, attempt, k->reset); continue; }
        if (code >= 400 && code < 500) break;
        code = -1;
        atomic_fetch_add(&st_retries, 1);
        usleep((useconds_t)(backoff_s(attempt) * 1e6));
    }
    free(url);
    tl_code = code;
    return code;
}

static int64_t http_size(store *s, const char *key) {
    char *url = join(s, key);
    if (!url) return -1;
    int64_t out = -1;
    for (int attempt = 0; attempt < 12; attempt++) {
        pace(s);
        CURL *c = fresh_handle(s);
        if (!c) break;
        curl_easy_setopt(c, CURLOPT_URL, url);
        curl_easy_setopt(c, CURLOPT_NOBODY, 1L);
        curl_easy_setopt(c, CURLOPT_HEADERFUNCTION, on_ratelimit_hdr);
        curl_easy_setopt(c, CURLOPT_HEADERDATA, nullptr);
        tl_reset = 0; tl_rl = 0;
        inflight_acquire();
        CURLcode rc = curl_easy_perform(c);
        inflight_release();
        long code = 0;
        curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
        account(c, rc, code, tl_rl);
        tl_code = code;
        if (rc == CURLE_OK && code >= 200 && code < 300) {
            curl_off_t cl = -1;
            curl_easy_getinfo(c, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &cl);
            out = cl;
            break;
        }
        if (code == 404) { out = -2; break; }
        if (code == 429 || code == 503) { throttle(s, code, attempt, tl_reset); continue; }
        if (code >= 400 && code < 500) break;
        atomic_fetch_add(&st_retries, 1);
        usleep((useconds_t)(backoff_s(attempt) * 1e6));
    }
    free(url);
    return out;
}

/* absolute-URL GET with Link header capture */
typedef struct { char *next; } hdrs;
static size_t on_hdr(char *buf, size_t sz, size_t nm, void *ud) {
    hdrs *h = ud; size_t n = sz * nm;
    on_ratelimit_hdr(buf, sz, nm, nullptr);
    if (n > 6 && !strncasecmp(buf, "link:", 5)) {
        char *value = strndup(buf + 5, n - 5);
        if (!value) return 0;
        char *p = value; while (*p == ' ') p++;
        for (char *q = p; (q = strchr(q, '<')); q++) {
            char *e = strchr(q, '>'); if (!e) break;
            char *rel = strstr(e, "rel=\"next\"");
            char *semi = strchr(e, ',');
            if (rel && (!semi || rel < semi)) { free(h->next); h->next = strndup(q + 1, (size_t)(e - q - 1)); break; }
            q = e;
        }
        free(value);
    }
    return n;
}

uint8_t *store_get_url(store *s, const char *url, size_t *len, char **link_next) {
    sink k = {0};
    hdrs h = {0};
    long code = -1;
    for (int attempt = 0; attempt < 12; attempt++) {
        pace(s);
        CURL *c = fresh_handle(s);
        if (!c) break;
        curl_easy_setopt(c, CURLOPT_URL, url);
        curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, on_data);
        curl_easy_setopt(c, CURLOPT_WRITEDATA, &k);
        curl_easy_setopt(c, CURLOPT_HEADERFUNCTION, on_hdr);
        curl_easy_setopt(c, CURLOPT_HEADERDATA, &h);
        k.n = 0; free(h.next); h.next = nullptr; tl_reset = 0; tl_rl = 0;
        inflight_acquire();
        CURLcode rc = curl_easy_perform(c);
        inflight_release();
        code = 0;
        curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
        account(c, rc, code, tl_rl);
        if (rc == CURLE_OK) break;
        if (code == 404 || code == 403) break;
        if (code == 429 || code == 503) { throttle(s, code, attempt, tl_reset); continue; }
        code = -1;
        atomic_fetch_add(&st_retries, 1);
        usleep((useconds_t)(backoff_s(attempt) * 1e6));
    }
    if (code != 200) { free(k.buf); free(h.next); return nullptr; }
    if (!k.buf) k.buf = calloc(1, 1); else k.buf[k.n] = 0;
    if (len) *len = k.n;
    if (link_next) *link_next = h.next; else free(h.next);
    return k.buf;
}

/* ---- public ---- */

int64_t store_size(store *s, const char *key) { return s->local ? local_size(s, key) : http_size(s, key); }

/* a finished ranged GET into a fixed sink: len, -2 missing, -1 invalid */
static int64_t range_result(long code, const sink *k, int64_t off, int64_t len) {
    if (code == 404) return -2;
    if (code != 206 && code != 200) return -1;
    if (k->overflow || (int64_t)k->n != len) return -1;
    if (code == 200) return off == 0 ? len : -1;
    return k->has_range && k->range_start == (uint64_t)off && k->range_end == (uint64_t)(off + len - 1) &&
           (!k->range_total_known || k->range_total > k->range_end) ? len : -1;
}

int64_t store_read(store *s, const char *key, int64_t off, int64_t len, uint8_t *out) {
    if (off < 0 || len < 0 || off > INT64_MAX - len) return -1;
    if (!len) return 0;
    if (s->local) return local_read(s, key, off, len, out);
    sink k = {.buf = out, .cap = (size_t)len, .fixed = 1};
    return range_result(http_get(s, key, off, len, 0, &k), &k, off, len);
}

int64_t store_read_tail(store *s, const char *key, int64_t len, uint8_t *out, int64_t *total) {
    *total = -1;
    if (len <= 0) return -1;
    static int tail_on = -1;
    if (tail_on < 0) tail_on = env_long("UFSM_HTTP_TAIL", 1) != 0;
    if (s->local || !tail_on) {
        int64_t size = store_size(s, key);
        if (size < 0) return size;
        *total = size;
        if (size < len) return -1;
        return store_read(s, key, size - len, len, out);
    }
    sink k = {.buf = out, .cap = (size_t)len, .fixed = 1};
    long code = http_get(s, key, 0, len, 1, &k);
    if (code == 404) return -2;
    /* the suffix must come back as a 206 naming the object's last len bytes and its total size */
    if (code == 206 && k.has_range && k.range_total_known) *total = (int64_t)k.range_total;
    if (code != 206 || k.overflow || (int64_t)k.n != len || !k.has_range || !k.range_total_known ||
        k.range_total < (uint64_t)len || k.range_end != k.range_total - 1 || k.range_start != k.range_total - (uint64_t)len) return -1;
    return len;
}

uint8_t *store_read_all(store *s, const char *key, size_t *len) {
    if (s->local) {
        int64_t n = local_size(s, key);
        if (n < 0) return nullptr;
        uint8_t *b = malloc((size_t)n + 1);
        if (!b) return nullptr;
        if (n && local_read(s, key, 0, n, b) != n) { free(b); return nullptr; }
        b[n] = 0;
        if (len) *len = (size_t)n;
        return b;
    }
    sink k = {0};
    long code = http_get(s, key, 0, -1, 0, &k);
    if (code != 200) { free(k.buf); return nullptr; }
    if (!k.buf) k.buf = calloc(1, 1);
    else k.buf[k.n] = 0;
    if (len) *len = k.n;
    return k.buf;
}

/* ---- batch (curl_multi) ---- */

int store_batch_enabled(const store *s) {
    static int on = -1;
    if (on < 0) on = env_long("UFSM_HTTP_MULTI", 0) != 0;   /* opt-in: measured no faster than per-thread handles (2026-10-23), 5x fewer connections */
    return !s->local && on;
}

static thread_multi *get_multi(void) {
    if (tl_multi) return tl_multi;
    pthread_mutex_lock(&http_global_mu);
    int ready = http_initialized && !http_key_error;
    pthread_mutex_unlock(&http_global_mu);
    if (!ready) return nullptr;
    thread_multi *t = calloc(1, sizeof *t);
    if (!t || !(t->m = curl_multi_init())) { free(t); return nullptr; }
    curl_multi_setopt(t->m, CURLMOPT_PIPELINING, CURLPIPE_MULTIPLEX);
    if (pthread_setspecific(multi_key, t)) { multi_destroy(t); return nullptr; }
    return tl_multi = t;
}

typedef struct { int req, attempt; double not_before; sink k; char range[64]; CURL *e; } batch_slot;

int store_read_batch(store *s, store_req *reqs, int n, int conc, void (*done)(void *ud, int i), void *ud) {
    int failed = 0;
    if (s->local || !store_batch_enabled(s)) {
        for (int i = 0; i < n; i++) {
            reqs[i].result = store_read(s, reqs[i].key, reqs[i].off, reqs[i].len, reqs[i].out);
            failed |= reqs[i].result == -1;
            if (done) done(ud, i);
        }
        return failed ? -1 : 0;
    }
    if (n <= 0) return 0;
    if (conc <= 0) conc = (int)env_long("UFSM_HTTP_CONC", 64);
    if (conc > MULTI_MAX) conc = MULTI_MAX;
    if (conc > n) conc = n;
    long streams = env_long("UFSM_HTTP_STREAMS", 16); if (streams < 1) streams = 1;
    static long pipewait = -1;
    if (pipewait < 0) pipewait = env_long("UFSM_HTTP_PIPEWAIT", 1) != 0;
    long conns = (conc + streams - 1) / streams; if (conns < 1) conns = 1;
    thread_multi *t = get_multi();
    if (!t) return -1;
    curl_multi_setopt(t->m, CURLMOPT_MAX_HOST_CONNECTIONS, conns);
    curl_multi_setopt(t->m, CURLMOPT_MAX_CONCURRENT_STREAMS, streams);
    curl_multi_setopt(t->m, CURLMOPT_MAXCONNECTS, conns + 4);   /* keep them between batches */
    curl_slist_free_all(t->hdr); t->hdr = s->bearer ? curl_slist_append(nullptr, s->bearer) : nullptr;
    while (t->ne < conc) { CURL *e = curl_easy_init(); if (!e) break; t->e[t->ne++] = e; }
    if (!t->ne) return -1;
    if (conc > t->ne) conc = t->ne;
    /* queue: request ids waiting to start (retries go to the back with a not-before time) */
    int *queue = malloc((size_t)n * sizeof *queue), *attempts = calloc((size_t)n, sizeof *attempts);
    double *not_before = calloc((size_t)n, sizeof *not_before);
    batch_slot *slot = calloc((size_t)conc, sizeof *slot);
    char **urls = calloc((size_t)n, sizeof *urls);
    if (!queue || !attempts || !not_before || !slot || !urls) { free(queue); free(attempts); free(not_before); free(slot); free(urls); return -1; }
    int qh = 0, qn = n, inflight = 0, finished = 0;
    for (int i = 0; i < n; i++) queue[i] = i;
    for (int i = 0; i < conc; i++) { slot[i].req = -1; slot[i].e = t->e[i]; }
    while (finished < n) {
        /* start what may start */
        double now = mono(), wake = 0.1;
        for (int si = 0; si < conc && qn > 0; si++) {
            if (slot[si].req >= 0) continue;
            int r = queue[qh];
            if (not_before[r] > now) { if (not_before[r] - now < wake) wake = not_before[r] - now; break; }
            if (!inflight_try()) { if (wake > 0.005) wake = 0.005; break; }
            double w = pace_try(s);
            if (w > 0) { inflight_release(); if (w < wake) wake = w; break; }
            qh = (qh + 1) % n; qn--;
            store_req *q = &reqs[r];
            if (!urls[r] && !(urls[r] = join(s, q->key))) { inflight_release(); q->result = -1; failed = 1; finished++; if (done) done(ud, r); continue; }
            batch_slot *b = &slot[si];
            b->req = r; b->attempt = attempts[r];
            b->k = (sink){.buf = q->out, .cap = (size_t)q->len, .fixed = 1};
            CURL *e = b->e;
            curl_easy_reset(e);
            base_opts(e, t->hdr);
            if (g_share && g_share_mode < 2) curl_easy_setopt(e, CURLOPT_SHARE, g_share);
            if (pipewait) curl_easy_setopt(e, CURLOPT_PIPEWAIT, 1L);
            curl_easy_setopt(e, CURLOPT_URL, urls[r]);
            curl_easy_setopt(e, CURLOPT_WRITEFUNCTION, on_data);
            curl_easy_setopt(e, CURLOPT_WRITEDATA, &b->k);
            curl_easy_setopt(e, CURLOPT_HEADERFUNCTION, on_read_hdr);
            curl_easy_setopt(e, CURLOPT_HEADERDATA, &b->k);
            curl_easy_setopt(e, CURLOPT_PRIVATE, (void *)b);
            snprintf(b->range, sizeof b->range, "%lld-%lld", (long long)q->off, (long long)(q->off + q->len - 1));
            curl_easy_setopt(e, CURLOPT_RANGE, b->range);
            if (curl_multi_add_handle(t->m, e) != CURLM_OK) { inflight_release(); b->req = -1; q->result = -1; failed = 1; finished++; if (done) done(ud, r); continue; }
            inflight++;
        }
        if (!inflight) {   /* everything left is waiting for a back-off or a pacing slot */
            if (qn > 0) { atomic_fetch_add(&st_sleep_us, (uint64_t)(wake * 1e6)); usleep((useconds_t)(wake * 1e6)); }
            continue;
        }
        int running = 0;
        curl_multi_perform(t->m, &running);
        CURLMsg *msg; int left;
        while ((msg = curl_multi_info_read(t->m, &left))) {
            if (msg->msg != CURLMSG_DONE) continue;
            CURL *e = msg->easy_handle; CURLcode rc = msg->data.result;
            batch_slot *b = nullptr; curl_easy_getinfo(e, CURLINFO_PRIVATE, (char **)&b);
            long code = 0; curl_easy_getinfo(e, CURLINFO_RESPONSE_CODE, &code);
            account(e, rc, code, b->k.rl);
            curl_multi_remove_handle(t->m, e);
            inflight--; inflight_release();
            int r = b->req; b->req = -1;
            store_req *q = &reqs[r];
            int retry = 0;
            if (rc == CURLE_OK || code == 404) q->result = range_result(rc == CURLE_OK ? code : 404, &b->k, q->off, q->len);
            else if (b->k.overflow || rc == CURLE_WRITE_ERROR) q->result = -1;
            else if (code == 429 || code == 503) { throttle(s, code, b->attempt, b->k.reset); retry = 1; }
            else if (code >= 400 && code < 500) q->result = -1;
            else { retry = 1; atomic_fetch_add(&st_retries, 1); not_before[r] = mono() + backoff_s(b->attempt); }
            if (retry && b->attempt + 1 < 12) {
                attempts[r] = b->attempt + 1;
                queue[(qh + qn) % n] = r; qn++;
                continue;
            }
            if (retry) q->result = -1;
            if (q->result == -1) failed = 1;
            finished++;
            if (done) done(ud, r);
        }
        if (finished < n) {
            int ms = (int)(wake * 1000); if (ms < 1) ms = 1;
            curl_multi_poll(t->m, nullptr, 0, ms, nullptr);
        }
    }
    for (int i = 0; i < n; i++) free(urls[i]);
    free(queue); free(attempts); free(not_before); free(slot); free(urls);
    tl_code = failed ? -1 : 206;
    return failed ? -1 : 0;
}
