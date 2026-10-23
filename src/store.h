/* Byte store: a local directory or an HTTP(S) tree. Keys are '/'-separated paths under the root.
   Only whole-object and byte-range reads; thread-safe (one libcurl handle per calling thread). */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct store store;

store *store_open(const char *root);          /* "/path" or "https://host/prefix" (trailing '/' optional) */
void store_close(store *s);
int store_is_local(const store *s);
void store_set_bearer(store *s, const char *token);   /* Authorization: Bearer <token> on every request */
const char *store_root(const store *s);

/* Size of an object in bytes; -2 if missing, -1 on error. Permission failures are errors. */
int64_t store_size(store *s, const char *key);
/* Read [off, off+len) into out. Returns len on success, -2 if missing, -1 on error.
   HTTP partial responses must describe the requested range; a whole response is valid only at offset zero. */
int64_t store_read(store *s, const char *key, int64_t off, int64_t len, uint8_t *out);
/* Read a whole object into a malloc'd buffer (nul-terminated for convenience). nullptr if absent. */
uint8_t *store_read_all(store *s, const char *key, size_t *len);

/* GET an absolute URL (bearer of s applied). Returns malloc'd body (nul-terminated) or nullptr; if
   link_next is non-null it receives the malloc'd URL of the Link rel="next" header, or nullptr. */
uint8_t *store_get_url(store *s, const char *url, size_t *len, char **link_next);

long store_last_status(void);
/* Pace requests on this store to at most rps per second (token bucket shared by all threads); 0 = unlimited.
   HuggingFace roots default to 14/s (their limit is 5000 per 5 minutes). 429 responses sleep until the window resets. */
void store_set_rate(store *s, double rps);   /* HTTP status of this thread's last request (-1 = transport failure) */

/* Global init/teardown (libcurl). Safe to call repeatedly. Cleanup requires other callers to have joined.
   Exiting threads return handles to a bounded idle pool; cleanup closes all pooled connections.
   UFSM_HTTP_REUSE=0 disables pooling for controlled comparisons. */
void store_global_init(void);
void store_global_cleanup(void);

/* Read the last len bytes of an object with one suffix-range GET (a shard index): fills out, sets *total to the object size.
   Returns len, -2 if missing, -1 on error. UFSM_HTTP_TAIL=0 falls back to HEAD + range GET. */
int64_t store_read_tail(store *s, const char *key, int64_t len, uint8_t *out, int64_t *total);

/* Many byte ranges at once. HTTP: one curl_multi per calling thread (persistent connections), HTTP/2 multiplexed,
   at most conc requests in flight (conc <= 0: UFSM_HTTP_CONC, default 64) and UFSM_HTTP_STREAMS (default 16) streams per
   connection, so conc / streams connections share the bandwidth. done(ud, i) runs on the calling thread as request i completes
   (reqs[i].result: len, -2 missing, -1 error); validation and retries match store_read. Local stores read serially.
   Returns 0 if every request succeeded or was missing, -1 otherwise. Requests count against UFSM_HTTP_INFLIGHT. */
typedef struct { const char *key; int64_t off, len; uint8_t *out; int64_t result; } store_req;
int store_read_batch(store *s, store_req *reqs, int n, int conc, void (*done)(void *ud, int i), void *ud);
int store_batch_enabled(const store *s);   /* HTTP store and UFSM_HTTP_MULTI=1 (default off) */

/* Pacing: at most UFSM_HTTP_INFLIGHT (default 48) requests in flight per process (the volcomp mirror allows 64 per IP and answers
   429 beyond, without rate-limit headers); a 429 / 503 pauses the store briefly (Retry-After / ratelimit header, else 0.5 s doubling
   to 8 s) and halves an adaptive request-rate cap that recovers by 10% a second. UFSM_HTTP_RPS / UFSM_HTTP_BURST set a fixed cap. */

/* Process-wide HTTP accounting (all stores, all threads). stall = a request slower than 64 KB/s that took over 2 s.
   ratelimit = responses carrying a ratelimit / x-ratelimit-* / retry-after header; last_ratelimit is the latest such header.
   UFSM_HTTP_LOG=1 prints every 429 / 503 / rate-limit header with a timestamp. */
typedef struct {
    uint64_t requests, bytes, n429, n503, n5xx, transport_errors, retries, ratelimit, stalls, connects;   /* connects: new connections */
    double req_seconds, sleep_seconds, max_req_seconds;
    char last_ratelimit[192];
} store_stats_t;
void store_stats_get(store_stats_t *out);
void store_stats_reset(void);
