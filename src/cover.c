#include "cover.h"
#include "checkpoint.h"
#include "json.h"
#include <openssl/evp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

static int number(const json *j, uint64_t *out) {
    if (!j || j->type != J_NUM || !isfinite(j->num) || j->num < 0 || j->num > INT_MAX || j->num != floor(j->num)) return -1;
    *out = (uint64_t)j->num; return 0;
}
static int hash_valid(const char *s) {
    return s && strlen(s) == 64 && strspn(s, "0123456789abcdef") == 64;
}
static int tile_cmp(const void *a, const void *b) {
    const int64_t *x = a, *y = b;
    for (int i = 0; i < 4; i++) if (x[i] != y[i]) return x[i] < y[i] ? -1 : 1;
    return 0;
}
void cover_free(cover_plan *p) { if (p) { free(p->tiles); free(p); } }
cover_plan *cover_load(const char *path, const sources *S, int P) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    if (fseek(f, 0, SEEK_END)) { fclose(f); return nullptr; }
    long len = ftell(f); rewind(f);
    if (len <= 0 || len > 32 * 1024 * 1024) { fclose(f); return nullptr; }
    char *buf = malloc((size_t)len + 1);
    if (!buf) { fclose(f); return nullptr; }
    int ok = fread(buf, 1, (size_t)len, f) == (size_t)len; fclose(f); buf[len] = 0;
    json *j = ok ? json_parse(buf, (size_t)len) : nullptr;
    cover_plan *p = calloc(1, sizeof *p);
    unsigned char hash[EVP_MAX_MD_SIZE]; unsigned nh = 0;
    if (!p || !j || !EVP_Digest(buf, (size_t)len, hash, &nh, EVP_sha256(), nullptr) || nh != 32) goto bad;
    for (int i = 0; i < 32; i++) snprintf(p->sha256 + 2 * i, 3, "%02x", hash[i]);
    uint64_t version, edge, level;
    if (number(json_get(j, "version"), &version) || version != 1 || number(json_get(j, "P"), &edge) || edge != (uint64_t)P ||
        number(json_get(j, "level"), &level) || level || number(json_get(j, "count"), &p->count) || !p->count) goto bad;
    const json *names = json_get(j, "source_names"), *tiles = json_get(j, "tiles");
    if (!names || names->type != J_ARR || names->n != (size_t)S->n || !tiles || tiles->type != J_ARR || tiles->n != p->count) goto bad;
    for (int i = 0; i < S->n; i++) if (strcmp(json_str(json_at(names, (size_t)i), ""), S->src[i].name)) goto bad;
    p->P = P; p->tiles = calloc((size_t)p->count, sizeof *p->tiles); if (!p->tiles) goto bad;
    for (uint64_t k = 0; k < p->count; k++) {
        const json *t = json_at(tiles, (size_t)k);
        if (!t || t->type != J_ARR || t->n != 4) goto bad;
        for (int d = 0; d < 4; d++) { uint64_t v; if (number(json_at(t, (size_t)d), &v)) goto bad; p->tiles[k][d] = (int64_t)v; }
        if (p->tiles[k][0] >= S->n) goto bad;
    }
    int64_t (*sorted)[4] = malloc((size_t)p->count * sizeof *sorted); if (!sorted) goto bad;
    memcpy(sorted, p->tiles, (size_t)p->count * sizeof *sorted); qsort(sorted, (size_t)p->count, sizeof *sorted, tile_cmp);
    for (uint64_t k = 1; k < p->count; k++) if (!tile_cmp(sorted[k - 1], sorted[k])) { free(sorted); goto bad; }
    free(sorted); json_free(j); free(buf); return p;
bad:
    fprintf(stderr, "invalid training cover: %s\n", path);
    cover_free(p); json_free(j); free(buf); return nullptr;
}
int cover_checkpoint_read(const char *path, cover_progress *p) {
    memset(p, 0, sizeof *p);
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    char line[UFSM_CHECKPOINT_HEADER]; int ok = fgets(line, sizeof line, f) != nullptr; fclose(f);
    if (!ok || strncmp(line, "UFSM", 4) || !strchr(line, '\n')) return -1;
    json *j = json_parse(line + 4, strlen(line + 4)); if (!j) return -1;
    const json *c = json_path(j, "extra.cover"); int rc = 0;
    if (c) {
        const char *sha = json_str(json_get(c, "sha256"), ""); uint64_t base, step;
        if (!hash_valid(sha) || number(json_get(c, "count"), &p->count) || !p->count ||
            number(json_get(c, "cursor"), &p->cursor) || p->cursor > p->count ||
            number(json_get(c, "base_step"), &base) || base + p->count > INT_MAX ||
            number(json_get(j, "step"), &step) || step != base + p->cursor) rc = -1;
        else { strcpy(p->sha256, sha); p->base_step = (int)base; rc = 1; }
    }
    json_free(j); return rc;
}
int cover_checkpoint_extra(const char *runtime_json, const cover_progress *p, char *out, size_t cap) {
    size_t n = strlen(runtime_json);
    if (!n || runtime_json[n - 1] != '}' || !hash_valid(p->sha256) || p->cursor > p->count) return -1;
    int len = snprintf(out, cap, "%.*s,\"cover\":{\"sha256\":\"%s\",\"count\":%llu,\"cursor\":%llu,\"base_step\":%d}}",
        (int)(n - 1), runtime_json, p->sha256, (unsigned long long)p->count, (unsigned long long)p->cursor, p->base_step);
    return len < 0 || (size_t)len >= cap ? -1 : 0;
}
