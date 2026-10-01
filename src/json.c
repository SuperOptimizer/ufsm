#include "json.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char *p, *end; } cur;

static void ws(cur *c) { while (c->p < c->end && isspace((unsigned char)*c->p)) c->p++; }
static json *node(json_type t) { json *j = calloc(1, sizeof *j); if (j) j->type = t; return j; }
static json *parse_value(cur *c);

static int push(json *j, const char *key, json *v) {
    json **it = realloc(j->items, (j->n + 1) * sizeof *it);
    if (!it) return -1;
    j->items = it;
    if (key) {
        char **ks = realloc(j->keys, (j->n + 1) * sizeof *ks);
        if (!ks) return -1;
        j->keys = ks;
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
    char *e;
    double v = strtod(c->p, &e);
    if (e == c->p) return nullptr;
    c->p = e;
    json *j = node(J_NUM);
    j->num = v;
    return j;
}

json *json_parse(const char *text, size_t len) {
    cur c = {text, text + len};
    json *j = parse_value(&c);
    ws(&c);
    if (j && c.p != c.end) { json_free(j); return nullptr; }
    return j;
}

void json_free(json *j) {
    if (!j) return;
    for (size_t i = 0; i < j->n; i++) json_free(j->items[i]);
    if (j->keys) for (size_t i = 0; i < j->n; i++) free(j->keys[i]);
    free(j->items);
    free(j->keys);
    free(j->str);
    free(j);
}

const json *json_get(const json *o, const char *key) {
    if (!o || o->type != J_OBJ) return nullptr;
    for (size_t i = 0; i < o->n; i++) if (!strcmp(o->keys[i], key)) return o->items[i];
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
    for (char *tok = strtok(buf, "."); tok && j; tok = strtok(nullptr, "."))
        j = j->type == J_ARR ? json_at(j, (size_t)strtoul(tok, nullptr, 10)) : json_get(j, tok);
    return j;
}

double json_num(const json *j, double dflt) { return j && (j->type == J_NUM || j->type == J_BOOL) ? j->num : dflt; }
const char *json_str(const json *j, const char *dflt) { return j && j->type == J_STR ? j->str : dflt; }
