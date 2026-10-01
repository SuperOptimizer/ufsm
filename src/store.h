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

/* Size of an object in bytes; -1 if absent or on error. */
int64_t store_size(store *s, const char *key);
/* Read [off, off+len) of an object into out. Returns bytes read (== len on success), -1 on error. */
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

/* Global one-time init/teardown (libcurl). Safe to call repeatedly. */
void store_global_init(void);
void store_global_cleanup(void);
