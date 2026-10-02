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

struct store {
    char *root;      /* without trailing '/' */
    int local;
    char *bearer;    /* nullptr or "Authorization: Bearer ..." */
    double rps;      /* request pacing, 0 = unlimited */
    pthread_mutex_t mu;
    double next_ok;  /* earliest monotonic time the next request may start */
};

static double mono(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void pace(store *s) {
    if (s->rps <= 0) return;
    pthread_mutex_lock(&s->mu);
    double now = mono(), t = s->next_ok > now ? s->next_ok : now;
    s->next_ok = t + 1.0 / s->rps;
    pthread_mutex_unlock(&s->mu);
    if (t > now) usleep((useconds_t)((t - now) * 1e6));
}

void store_set_rate(store *s, double rps) { s->rps = rps; }

/* seconds until the rate window resets, from a "ratelimit: ...;t=NN" header; default 30 */
static _Thread_local double tl_reset;
static size_t on_ratelimit_hdr(char *buf, size_t sz, size_t nm, void *ud) {
    size_t n = sz * nm;
    if (n > 10 && !strncasecmp(buf, "ratelimit:", 10)) {
        char value[256]; snprintf(value, sizeof value, "%.*s", (int)(n - 10), buf + 10);
        char *t = strstr(value, ";t="); if (t) tl_reset = atof(t + 3);
    }
    if (ud) return ((size_t (*)(char *, size_t, size_t, void *))ud)(buf, sz, nm, nullptr);
    return n;
}

static void sleep_429(void) {
    double t = tl_reset > 0 && tl_reset < 600 ? tl_reset + 1 : 30;
    fprintf(stderr, "store: rate limited, sleeping %.0fs\n", t);
    usleep((useconds_t)(t * 1e6));
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
static void http_key_init(void) { http_key_error = pthread_key_create(&http_key, http_thread_free); }
void store_global_init(void) {
    pthread_once(&http_key_once, http_key_init);
    pthread_mutex_lock(&http_global_mu);
    if (!http_initialized) {
        http_initialized = curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK;
        const char *reuse = getenv("UFSM_HTTP_REUSE");
        http_reuse = !reuse || atoi(reuse) != 0;
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
    pthread_mutex_lock(&http_global_mu);
    while (http_idle_n) http_destroy(http_idle[--http_idle_n]);
    if (http_initialized) { curl_global_cleanup(); http_initialized = 0; }
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
    if (strstr(root, "huggingface.co")) s->rps = 14;
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
    close(fd);
    return got == len ? got : -1;
}

/* ---- http ---- */

typedef struct {
    uint8_t *buf; size_t n, cap; int fixed, overflow, has_range;
    uint64_t range_start, range_end, range_total; int range_total_known;
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
    return on_ratelimit_hdr(buf, sz, nm, nullptr);
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
    if (s->bearer) { tl_http->hdr = curl_slist_append(nullptr, s->bearer); curl_easy_setopt(c, CURLOPT_HTTPHEADER, tl_http->hdr); }
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(c, CURLOPT_LOW_SPEED_TIME, 60L);
    curl_easy_setopt(c, CURLOPT_LOW_SPEED_LIMIT, 1024L);
    curl_easy_setopt(c, CURLOPT_USERAGENT, "ufsm/0.1");
    curl_easy_setopt(c, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(c, CURLOPT_MAXCONNECTS, 2L);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(c, CURLOPT_FAILONERROR, 1L);
    return c;
}

/* GET (len < 0: whole object) with retries. Returns HTTP status, or -1 on transport failure. */
static long http_get(store *s, const char *key, int64_t off, int64_t len, sink *k) {
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
        tl_reset = 0;
        char range[64];
        if (len >= 0) {
            snprintf(range, sizeof range, "%lld-%lld", (long long)off, (long long)(off + len - 1));
            curl_easy_setopt(c, CURLOPT_RANGE, range);
        }
        k->n = 0; k->overflow = 0; k->has_range = 0;
        CURLcode rc = curl_easy_perform(c);
        curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
        if (rc == CURLE_OK) break;
        if (k->overflow || rc == CURLE_WRITE_ERROR) { code = -1; break; }
        if (code == 429) { sleep_429(); continue; }
        if (code >= 400 && code < 500) break;
        code = -1;
        usleep((useconds_t)(200000u << (attempt < 5 ? attempt : 5)));
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
        tl_reset = 0;
        CURLcode rc = curl_easy_perform(c);
        long code = 0;
        curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
        tl_code = code;
        if (rc == CURLE_OK && code >= 200 && code < 300) {
            curl_off_t cl = -1;
            curl_easy_getinfo(c, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &cl);
            out = cl;
            break;
        }
        if (code == 404) { out = -2; break; }
        if (code == 429) { sleep_429(); continue; }
        if (code >= 400 && code < 500) break;
        usleep((useconds_t)(200000u << (attempt < 5 ? attempt : 5)));
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
        k.n = 0; free(h.next); h.next = nullptr; tl_reset = 0;
        CURLcode rc = curl_easy_perform(c);
        curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
        if (rc == CURLE_OK) break;
        if (code == 404 || code == 403) break;
        if (code == 429) { sleep_429(); continue; }
        code = -1;
        usleep((useconds_t)(200000u << (attempt < 5 ? attempt : 5)));
    }
    if (code != 200) { free(k.buf); free(h.next); return nullptr; }
    if (!k.buf) k.buf = calloc(1, 1); else k.buf[k.n] = 0;
    if (len) *len = k.n;
    if (link_next) *link_next = h.next; else free(h.next);
    return k.buf;
}

/* ---- public ---- */

int64_t store_size(store *s, const char *key) { return s->local ? local_size(s, key) : http_size(s, key); }

int64_t store_read(store *s, const char *key, int64_t off, int64_t len, uint8_t *out) {
    if (off < 0 || len < 0 || off > INT64_MAX - len) return -1;
    if (!len) return 0;
    if (s->local) return local_read(s, key, off, len, out);
    sink k = {.buf = out, .cap = (size_t)len, .fixed = 1};
    long code = http_get(s, key, off, len, &k);
    if (code == 404) return -2;
    if (code != 206 && code != 200) return -1;
    if (k.overflow || (int64_t)k.n != len) return -1;
    if (code == 200) return off == 0 ? len : -1;
    return k.has_range && k.range_start == (uint64_t)off && k.range_end == (uint64_t)(off + len - 1) &&
           (!k.range_total_known || k.range_total > k.range_end) ? len : -1;
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
    long code = http_get(s, key, 0, -1, &k);
    if (code != 200) { free(k.buf); return nullptr; }
    if (!k.buf) k.buf = calloc(1, 1);
    else k.buf[k.n] = 0;
    if (len) *len = k.n;
    return k.buf;
}
