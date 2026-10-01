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
    if (n > 10 && !strncasecmp(buf, "ratelimit:", 10)) { char *t = strstr(buf, ";t="); if (t) tl_reset = atof(t + 3); }
    if (ud) return ((size_t (*)(char *, size_t, size_t, void *))ud)(buf, sz, nm, nullptr);
    return n;
}

static void sleep_429(void) {
    double t = tl_reset > 0 && tl_reset < 600 ? tl_reset + 1 : 30;
    fprintf(stderr, "store: rate limited, sleeping %.0fs\n", t);
    usleep((useconds_t)(t * 1e6));
}

static _Thread_local CURL *tl_curl;
static _Thread_local long tl_code;
long store_last_status(void) { return tl_code; }

void store_global_init(void) {
    static int done;
    if (!done) { curl_global_init(CURL_GLOBAL_DEFAULT); done = 1; }
}
void store_global_cleanup(void) { curl_global_cleanup(); }

store *store_open(const char *root) {
    store *s = calloc(1, sizeof *s);
    if (!s) return nullptr;
    size_t n = strlen(root);
    while (n > 1 && root[n - 1] == '/') n--;
    s->root = strndup(root, n);
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

typedef struct { uint8_t *buf; size_t n, cap; int fixed; } sink;

static size_t on_data(char *ptr, size_t sz, size_t nm, void *ud) {
    sink *k = ud;
    size_t n = sz * nm;
    if (k->fixed) {
        if (k->n + n > k->cap) n = k->cap - k->n; /* truncate: server ignored the range */
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

static _Thread_local struct curl_slist *tl_hdr;

static CURL *fresh_handle(const store *s) {
    if (!tl_curl) tl_curl = curl_easy_init();
    CURL *c = tl_curl;
    curl_easy_reset(c);
    if (tl_hdr) { curl_slist_free_all(tl_hdr); tl_hdr = nullptr; }
    if (s->bearer) { tl_hdr = curl_slist_append(nullptr, s->bearer); curl_easy_setopt(c, CURLOPT_HTTPHEADER, tl_hdr); }
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(c, CURLOPT_LOW_SPEED_TIME, 60L);
    curl_easy_setopt(c, CURLOPT_LOW_SPEED_LIMIT, 1024L);
    curl_easy_setopt(c, CURLOPT_USERAGENT, "ufsm/0.1");
    curl_easy_setopt(c, CURLOPT_TCP_KEEPALIVE, 1L);
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
        curl_easy_setopt(c, CURLOPT_URL, url);
        curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, on_data);
        curl_easy_setopt(c, CURLOPT_WRITEDATA, k);
        curl_easy_setopt(c, CURLOPT_HEADERFUNCTION, on_ratelimit_hdr);
        curl_easy_setopt(c, CURLOPT_HEADERDATA, nullptr);
        tl_reset = 0;
        char range[64];
        if (len >= 0) {
            snprintf(range, sizeof range, "%lld-%lld", (long long)off, (long long)(off + len - 1));
            curl_easy_setopt(c, CURLOPT_RANGE, range);
        }
        k->n = 0;
        CURLcode rc = curl_easy_perform(c);
        curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
        if (rc == CURLE_OK) break;
        if (code == 404 || code == 403) break;
        if (code == 429) { sleep_429(); continue; }
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
        curl_easy_setopt(c, CURLOPT_URL, url);
        curl_easy_setopt(c, CURLOPT_NOBODY, 1L);
        curl_easy_setopt(c, CURLOPT_HEADERFUNCTION, on_ratelimit_hdr);
        curl_easy_setopt(c, CURLOPT_HEADERDATA, nullptr);
        tl_reset = 0;
        if (curl_easy_perform(c) == CURLE_OK) {
            curl_off_t cl = -1;
            curl_easy_getinfo(c, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &cl);
            out = cl;
            break;
        }
        long code = 0;
        curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
        tl_code = code;
        if (code == 404 || code == 403) { out = -2; break; }
        if (code == 429) { sleep_429(); continue; }
        usleep((useconds_t)(200000u << (attempt < 5 ? attempt : 5)));
    }
    free(url);
    return out;
}

/* absolute-URL GET with Link header capture */
typedef struct { char *next; } hdrs;
static size_t on_hdr(char *buf, size_t sz, size_t nm, void *ud) {
    hdrs *h = ud; size_t n = sz * nm;
    if (n > 10 && !strncasecmp(buf, "ratelimit:", 10)) { char *t = strstr(buf, ";t="); if (t) tl_reset = atof(t + 3); }
    if (n > 6 && !strncasecmp(buf, "link:", 5)) {
        char *p = buf + 5; while (*p == ' ') p++;
        for (char *q = p; (q = strchr(q, '<')); q++) {
            char *e = strchr(q, '>'); if (!e) break;
            char *rel = strstr(e, "rel=\"next\"");
            char *semi = strchr(e, ',');
            if (rel && (!semi || rel < semi)) { free(h->next); h->next = strndup(q + 1, (size_t)(e - q - 1)); break; }
            q = e;
        }
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
    if (s->local) return local_read(s, key, off, len, out);
    sink k = {out, 0, (size_t)len, 1};
    long code = http_get(s, key, off, len, &k);
    if (code == 404 || code == 403) return -2;
    if (code != 206 && code != 200) return -1;
    return (int64_t)k.n == len ? len : -1;
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
