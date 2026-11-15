#include "json.h"
#include <ctype.h>
#include <stdarg.h>
#include <strings.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct { const char *p, *end; } cur;

static void ws(cur *c) { while (c->p < c->end && isspace((unsigned char)*c->p)) c->p++; }
static json *node(json_type t) { json *j = calloc(1, sizeof *j); if (j) j->type = t; return j; }
static json *parse_value(cur *c);

static int push(json *j, const char *key, json *v) {
    json **it = realloc(j->items, (j->n + 1) * sizeof *it);
    if (!it) return -1;
    j->items = it;
    if (key) {
        if (j->n < 4096) for (size_t i = 0; i < j->n; i++) if (!strcmp(j->keys[i], key)) return -1;   /* a duplicate key: first-wins hid the second (mini/json 5c; small objects: config / metadata, not big indexes) */
        char **ks = realloc(j->keys, (j->n + 1) * sizeof *ks);
        if (!ks) return -1;
        j->keys = ks;
        unsigned char *g = realloc(j->got, j->n + 1);
        if (!g) return -1;
        j->got = g; g[j->n] = 0;
        j->keys[j->n] = (char *)key;
    }
    j->items[j->n++] = v;
    return 0;
}

static char *parse_string(cur *c) {
    if (c->p >= c->end || *c->p != '"') return nullptr;
    c->p++;
    size_t cap = 16, n = 0;
    char *s = malloc(cap);
    if (!s) return nullptr;
    while (c->p < c->end && *c->p != '"') {
        char ch = *c->p++;
        if (ch == '\\' && c->p < c->end) {
            char e = *c->p++;
            switch (e) {
            case 'n': ch = '\n'; break;
            case 't': ch = '\t'; break;
            case 'r': ch = '\r'; break;
            case 'b': ch = '\b'; break;
            case 'f': ch = '\f'; break;
            case 'u': { /* keep ASCII range only; others become '?' */
                if (c->end - c->p < 4) { free(s); return nullptr; }
                unsigned v = (unsigned)strtoul((char[]){c->p[0], c->p[1], c->p[2], c->p[3], 0}, nullptr, 16);
                c->p += 4;
                ch = v < 128 ? (char)v : '?';
                break;
            }
            default: ch = e;
            }
        }
        if (n + 2 > cap) { cap *= 2; char *t = realloc(s, cap); if (!t) { free(s); return nullptr; } s = t; }
        s[n++] = ch;
    }
    if (c->p >= c->end) { free(s); return nullptr; }
    c->p++;
    s[n] = 0;
    return s;
}

static json *parse_value(cur *c) {
    ws(c);
    if (c->p >= c->end) return nullptr;
    char ch = *c->p;
    if (ch == '{') {
        c->p++;
        json *j = node(J_OBJ);
        ws(c);
        if (c->p < c->end && *c->p == '}') { c->p++; return j; }
        for (;;) {
            ws(c);
            char *k = parse_string(c);
            if (!k) { json_free(j); return nullptr; }
            ws(c);
            if (c->p >= c->end || *c->p != ':') { free(k); json_free(j); return nullptr; }
            c->p++;
            json *v = parse_value(c);
            if (!v || push(j, k, v)) { free(k); json_free(v); json_free(j); return nullptr; }
            ws(c);
            if (c->p < c->end && *c->p == ',') { c->p++; continue; }
            if (c->p < c->end && *c->p == '}') { c->p++; return j; }
            json_free(j);
            return nullptr;
        }
    }
    if (ch == '[') {
        c->p++;
        json *j = node(J_ARR);
        ws(c);
        if (c->p < c->end && *c->p == ']') { c->p++; return j; }
        for (;;) {
            json *v = parse_value(c);
            if (!v || push(j, nullptr, v)) { json_free(v); json_free(j); return nullptr; }
            ws(c);
            if (c->p < c->end && *c->p == ',') { c->p++; continue; }
            if (c->p < c->end && *c->p == ']') { c->p++; return j; }
            json_free(j);
            return nullptr;
        }
    }
    if (ch == '"') {
        char *s = parse_string(c);
        if (!s) return nullptr;
        json *j = node(J_STR);
        j->str = s;
        return j;
    }
    if (c->end - c->p >= 4 && !strncmp(c->p, "true", 4)) { c->p += 4; json *j = node(J_BOOL); j->num = 1; return j; }
    if (c->end - c->p >= 5 && !strncmp(c->p, "false", 5)) { c->p += 5; return node(J_BOOL); }
    if (c->end - c->p >= 4 && !strncmp(c->p, "null", 4)) { c->p += 4; return node(J_NULL); }
    /* JSON number syntax only (mini/json 5: strtod took nan / inf / hex and could read past the end of an unterminated buffer):
       -?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)?, converted from a bounded copy */
    const char *q = c->p;
    if (q < c->end && *q == '-') q++;
    if (q >= c->end || !isdigit((unsigned char)*q)) return nullptr;
    if (*q == '0') q++; else while (q < c->end && isdigit((unsigned char)*q)) q++;
    if (q < c->end && *q == '.') { q++; if (q >= c->end || !isdigit((unsigned char)*q)) return nullptr; while (q < c->end && isdigit((unsigned char)*q)) q++; }
    if (q < c->end && (*q == 'e' || *q == 'E')) { q++; if (q < c->end && (*q == '+' || *q == '-')) q++;
        if (q >= c->end || !isdigit((unsigned char)*q)) return nullptr; while (q < c->end && isdigit((unsigned char)*q)) q++; }
    char nb[400]; const size_t nl = (size_t)(q - c->p); if (nl >= sizeof nb) return nullptr;
    memcpy(nb, c->p, nl); nb[nl] = 0;
    double v = strtod(nb, nullptr);
    c->p = q;
    json *j = node(J_NUM);
    j->num = v;
    return j;
}

static _Thread_local size_t g_err_off;
size_t json_error_offset(void) { return g_err_off; }
json *json_parse(const char *text, size_t len) {
    cur c = {text, text + len};
    json *j = parse_value(&c);
    ws(&c);
    if (j && c.p != c.end) { json_free(j); j = nullptr; }
    g_err_off = j ? 0 : (size_t)(c.p - text);
    return j;
}
void json_mark_read(const json *o) { if (o && o->type == J_OBJ && o->got) memset(o->got, 1, o->n); }
int json_unread(const json *j, const char *path, void (*cb)(const char *, void *), void *u) {
    if (!j || (j->type != J_OBJ && j->type != J_ARR)) return 0;
    int n = 0; char p[1024];
    for (size_t i = 0; i < j->n; i++) {
        if (j->type == J_OBJ) {
            const char *k = j->keys[i];
            snprintf(p, sizeof p, "%s%s%s", path, *path ? "." : "", k);
            if (!strcmp(k, "note") || !strcmp(k, "comment") || k[0] == '_') continue;
            if (!j->got || !j->got[i]) { if (cb) cb(p, u); n++; continue; }   /* (an unread key: its subtree is not reported again) */
        } else snprintf(p, sizeof p, "%s.%zu", path, i);
        n += json_unread(j->items[i], p, cb, u);
    }
    return n;
}

void json_free(json *j) {
    if (!j) return;
    for (size_t i = 0; i < j->n; i++) json_free(j->items[i]);
    if (j->keys) for (size_t i = 0; i < j->n; i++) free(j->keys[i]);
    free(j->items);
    free(j->keys);
    free(j->got);
    free(j->str);
    free(j);
}

const json *json_get(const json *o, const char *key) {
    if (!o || o->type != J_OBJ) return nullptr;
    for (size_t i = 0; i < o->n; i++) if (!strcmp(o->keys[i], key)) { if (o->got) o->got[i] = 1; return o->items[i]; }   /* (got: the json is the caller's; marking is its only write) */
    return nullptr;
}

const json *json_at(const json *a, size_t i) {
    if (!a || a->type != J_ARR || i >= a->n) return nullptr;
    return a->items[i];
}

const json *json_path(const json *j, const char *path) {
    char buf[256];
    size_t n = strlen(path);
    if (n >= sizeof buf) return nullptr;
    memcpy(buf, path, n + 1);
    char *save = nullptr;
    for (char *tok = strtok_r(buf, ".", &save); tok && j; tok = strtok_r(nullptr, ".", &save))
        j = j->type == J_ARR ? json_at(j, (size_t)strtoul(tok, nullptr, 10)) : json_get(j, tok);
    return j;
}

double json_num(const json *j, double dflt) { return j && (j->type == J_NUM || j->type == J_BOOL) ? j->num : dflt; }
const char *json_str(const json *j, const char *dflt) { return j && j->type == J_STR ? j->str : dflt; }

int json_fprintf(FILE *f, const char *fmt, ...) {
    va_list ap, aq; va_start(ap, fmt); va_copy(aq, ap);
    char sb[1024], *b = sb; const int n = vsnprintf(sb, sizeof sb, fmt, ap); va_end(ap);
    if (n < 0) { va_end(aq); return -1; }
    if ((size_t)n >= sizeof sb) { b = malloc((size_t)n + 1); if (!b) { va_end(aq); return -1; } vsnprintf(b, (size_t)n + 1, fmt, aq); }
    va_end(aq);
    int out = 0, instr = 0;
    for (int i = 0; i < n;) {
        const char ch = b[i];
        if (instr) { if (ch == '\\' && i + 1 < n) { fputc(ch, f); fputc(b[i + 1], f); i += 2; out += 2; continue; } if (ch == '"') instr = 0; fputc(ch, f); i++; out++; continue; }
        if (ch == '"') { instr = 1; fputc(ch, f); i++; out++; continue; }
        const int neg = ch == '-' && i + 1 < n && isalpha((unsigned char)b[i + 1]), w = i + neg;
        if (isalpha((unsigned char)b[w]) && (i == 0 || !isalnum((unsigned char)b[i - 1]))) {   /* a bare word: true / false / null or a non-finite */
            int e = w; while (e < n && isalnum((unsigned char)b[e])) e++;
            const int len = e - w;
            if ((len == 3 && (!strncasecmp(b + w, "nan", 3) || !strncasecmp(b + w, "inf", 3))) || (len == 8 && !strncasecmp(b + w, "infinity", 8))) {
                fputs("null", f); out += 4; i = e; continue;   /* a leading '-' goes with it */
            }
            fwrite(b + i, 1, (size_t)(e - i), f); out += e - i; i = e; continue;
        }
        fputc(ch, f); i++; out++;
    }
    if (b != sb) free(b);
    return out;
}
