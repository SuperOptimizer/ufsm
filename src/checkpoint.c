#include "checkpoint.h"
#include "json.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int valid(const checkpoint_runtime *r) {
    return r->version == 1 && r->train_window > 0 && r->prec >= 0 && r->prec <= 4 &&
        (r->f16 == 0 || r->f16 == 1) && (r->act_mx4 == 0 || r->act_mx4 == 1) &&
        (r->act_mx8 == 0 || r->act_mx8 == 1) && (r->grad_mx8 == 0 || r->grad_mx8 == 1);
}
static int integer(const json *j, const char *key, int *v) {
    const json *n = json_get(j, key);
    if (!n || n->type != J_NUM || !isfinite(n->num) || n->num < 0 || n->num > 2147483647 || n->num != floor(n->num)) return -1;
    *v = (int)n->num; return 0;
}
int checkpoint_runtime_read(const char *path, checkpoint_runtime *r) {
    memset(r, 0, sizeof *r);
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    char line[UFSM_CHECKPOINT_HEADER];
    int got = fgets(line, sizeof line, f) != nullptr; fclose(f);
    if (!got || strncmp(line, "UFSM", 4) || !strchr(line, '\n')) return -1;
    json *j = json_parse(line + 4, strlen(line + 4)); if (!j || j->type != J_OBJ) { json_free(j); return -1; }
    const json *m = json_path(j, "extra.runtime");
    int rc = 0;
    if (m) {
        const json *p = json_get(m, "policy"), *o = json_get(m, "optimizer");
        if (m->type != J_OBJ || integer(m, "version", &r->version) || integer(m, "train_window", &r->train_window) ||
            integer(m, "prec", &r->prec) || integer(m, "f16", &r->f16) || integer(m, "act_mx4", &r->act_mx4) ||
            integer(m, "act_mx8", &r->act_mx8) || integer(m, "grad_mx8", &r->grad_mx8) ||
            !p || p->type != J_STR || strlen(p->str) >= sizeof r->policy ||
            !o || o->type != J_STR || strlen(o->str) >= sizeof r->optimizer || !valid(r)) rc = -1;
        else { strcpy(r->policy, p->str); strcpy(r->optimizer, o->str); rc = 1; }
    }
    json_free(j); return rc;
}
static int quoted(const char *s, char *out, size_t cap) {
    size_t n = 0;
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c < 32) return -1;
        if (c == '"' || c == '\\') { if (n + 1 >= cap) return -1; out[n++] = '\\'; }
        if (n + 1 >= cap) return -1;
        out[n++] = (char)c;
    }
    out[n] = 0; return 0;
}
int checkpoint_runtime_json(const checkpoint_runtime *r, char *out, size_t cap) {
    char p[4096], o[32];
    if (!valid(r) || quoted(r->policy, p, sizeof p) || quoted(r->optimizer, o, sizeof o)) return -1;
    int n = snprintf(out, cap, "{\"runtime\":{\"version\":1,\"train_window\":%d,\"prec\":%d,\"f16\":%d,\"act_mx4\":%d,\"act_mx8\":%d,\"grad_mx8\":%d,\"policy\":\"%s\",\"optimizer\":\"%s\"}}",
        r->train_window, r->prec, r->f16, r->act_mx4, r->act_mx8, r->grad_mx8, p, o);
    return n < 0 || (size_t)n >= cap ? -1 : 0;
}
