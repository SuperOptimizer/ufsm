#include "hf.h"
#include "json.h"
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

char *hf_token(const char *path) {
    char p[512];
    if (!path) { snprintf(p, sizeof p, "%s/huggingfacetoken", getenv("HOME") ? getenv("HOME") : "."); path = p; }
    FILE *f = fopen(path, "r");
    if (!f) return nullptr;
    char buf[256] = {0};
    if (!fgets(buf, sizeof buf, f)) { fclose(f); return nullptr; }
    fclose(f);
    size_t n = strlen(buf);
    while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r' || buf[n - 1] == ' ')) buf[--n] = 0;
    return n ? strdup(buf) : nullptr;
}

/* "https://huggingface.co/buckets/<org>/<name>/resolve" -> ".../tree" */
static char *api_root(const char *root) {
    const char *r = strstr(root, "/resolve"), *b = strstr(root, "/buckets/");
    if (!r || !b || b > r) return nullptr;
    size_t n = (size_t)(b - root), m = (size_t)(r - b);
    char *a = malloc(n + m + 16);
    memcpy(a, root, n);
    strcpy(a + n, "/api");
    memcpy(a + n + 4, b, m);
    strcpy(a + n + 4 + m, "/tree");
    return a;
}

int hf_list(store *s, const char *key, list_cb cb, void *ud) {
    char *api = api_root(store_root(s));
    if (!api) { fprintf(stderr, "hf_list: root is not a buckets/.../resolve URL\n"); return -1; }
    char *url = malloc(strlen(api) + strlen(key) + 64);
    sprintf(url, "%s/%s?recursive=false&limit=1000", api, key);
    size_t klen = strlen(key);
    int rc = 0, total = 0;
    while (url && !rc) {
        size_t n;
        char *next = nullptr;
        uint8_t *body = store_get_url(s, url, &n, &next);
        free(url);
        url = next;
        if (!body) { fprintf(stderr, "hf_list: request failed\n"); rc = -1; break; }
        json *j = json_parse((char *)body, n);
        if (!j) { fprintf(stderr, "hf_list: cannot parse response (%zu bytes): %.200s\n", n, (char *)body); free(body); rc = -1; break; }
        free(body);
        const json *items = j && j->type == J_ARR ? j : json_get(j, "items");
        for (size_t i = 0; items && i < items->n && !rc; i++) {
            const json *it = json_at(items, i);
            const char *path = json_str(json_get(it, "path"), "");
            const char *name = strncmp(path, key, klen) == 0 && path[klen] == '/' ? path + klen + 1 : path;
            int is_dir = !strcmp(json_str(json_get(it, "type"), ""), "directory");
            rc = cb(name, is_dir, (int64_t)json_num(json_get(it, "size"), 0), ud);
            total++;
        }
        json_free(j);
    }
    free(url);
    free(api);
    return rc < 0 ? -1 : total;
}

/* S3 ListObjectsV2 XML: <Contents><Key>..</Key><Size>..</Size></Contents>, <CommonPrefixes><Prefix>..</Prefix>,
   <NextContinuationToken>..</NextContinuationToken> */
static char *xml_tag(const char *p, const char *tag, const char **after) {
    char open[64], close[64];
    snprintf(open, sizeof open, "<%s>", tag);
    snprintf(close, sizeof close, "</%s>", tag);
    const char *a = strstr(p, open);
    if (!a) return nullptr;
    a += strlen(open);
    const char *b = strstr(a, close);
    if (!b) return nullptr;
    if (after) *after = b + strlen(close);
    return strndup(a, (size_t)(b - a));
}

int s3_list(store *s, const char *prefix, list_cb cb, void *ud) {
    char *token = nullptr;
    int rc = 0, total = 0;
    size_t plen = strlen(prefix);
    do {
        char *url = malloc(strlen(store_root(s)) + plen + (token ? strlen(token) : 0) + 128);
        sprintf(url, "%s/?list-type=2&delimiter=/&max-keys=1000&prefix=%s/%s%s", store_root(s), prefix, token ? "&continuation-token=" : "", token ? token : "");
        size_t n;
        uint8_t *body = store_get_url(s, url, &n, nullptr);
        free(url);
        free(token);
        token = nullptr;
        if (!body) { fprintf(stderr, "s3_list: request failed\n"); rc = -1; break; }
        const char *p = (char *)body;
        for (;;) {
            const char *after;
            char *k = xml_tag(p, "Key", &after);
            if (!k) break;
            char *sz = xml_tag(p, "Size", nullptr);
            const char *name = strncmp(k, prefix, plen) == 0 && k[plen] == '/' ? k + plen + 1 : k;
            if (*name && !rc) { rc = cb(name, 0, sz ? atoll(sz) : 0, ud); total++; }
            free(k); free(sz);
            p = after;
        }
        p = (char *)body;
        for (;;) {
            const char *after;
            char *k = xml_tag(p, "Prefix", &after);
            if (!k) break;
            if (strncmp(k, prefix, plen) == 0 && k[plen] == '/' && k[plen + 1]) {
                size_t l = strlen(k);
                if (k[l - 1] == '/') k[l - 1] = 0;
                if (!rc) { rc = cb(k + plen + 1, 1, 0, ud); total++; }
            }
            free(k);
            p = after;
        }
        token = xml_tag((char *)body, "NextContinuationToken", nullptr);
        free(body);
    } while (token && !rc);
    free(token);
    return rc < 0 ? -1 : total;
}

int dir_list(const char *dir, list_cb cb, void *ud) {
    DIR *d = opendir(dir);
    if (!d) return -1;
    struct dirent *e;
    int rc = 0, total = 0;
    while (!rc && (e = readdir(d))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        char p[2048];
        snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
        struct stat st;
        if (stat(p, &st)) continue;
        rc = cb(e->d_name, S_ISDIR(st.st_mode), (int64_t)st.st_size, ud);
        total++;
    }
    closedir(d);
    return rc < 0 ? -1 : total;
}

int store_list(store *s, const char *key, list_cb cb, void *ud) {
    const char *root = store_root(s);
    if (store_is_local(s)) {
        char p[2048];
        snprintf(p, sizeof p, "%s/%s", root, key);
        return dir_list(p, cb, ud);
    }
    if (strstr(root, "huggingface.co")) return hf_list(s, key, cb, ud);
    if (strstr(root, "s3.amazonaws.com")) return s3_list(s, key, cb, ud);
    fprintf(stderr, "store_list: no listing method for %s\n", root);
    return -1;
}
