/* side_votes_merge (votes.h): several surface sets' side / vote arrays added into one vote array, shard by shard. */
#include "votes.h"
#include "z3w.h"
#include "zarr3.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static _Thread_local char errbuf[256];
const char *side_votes_error(void) { return errbuf; }
static int fail(const char *fmt, ...) { va_list ap; va_start(ap, fmt); vsnprintf(errbuf, sizeof errbuf, fmt, ap); va_end(ap); return -1; }

static int any_shard_under(z3 *z, const int64_t o[3], const int64_t n[3]) {   /* an input shard under the box (or an error: read it) */
    const z3_meta *m = z3_meta_of(z);
    for (int64_t a = o[0] / m->shard[0]; a <= (o[0] + n[0] - 1) / m->shard[0]; a++)
        for (int64_t b = o[1] / m->shard[1]; b <= (o[1] + n[1] - 1) / m->shard[1]; b++)
            for (int64_t c = o[2] / m->shard[2]; c <= (o[2] + n[2] - 1) / m->shard[2]; c++) if (z3_shard_present(z, a, b, c) != 0) return 1;
    return 0;
}

int side_votes_merge(const char *out_group, double um, z3 *const *in, const char *const *names, int nin, int S, int nt, int *nshards, size_t stats[2]) {
    if (nin < 1 || S < 128 || S % 128 || !(um > 0)) return fail("side-votes: one input or more, shard a multiple of 128, um > 0");
    const z3_meta *m = z3_meta_of(in[0]);
    for (int i = 0; i < nin; i++) {
        const z3_meta *mi = z3_meta_of(in[i]);
        if (memcmp(mi->shape, m->shape, sizeof m->shape) || mi->fill != 0) return fail("side-votes: %s: every input on the first one's grid (equal shape), fill 0", names ? names[i] : "?");
    }
    static uint8_t add[256][256]; static int add_ready;   /* the sum of two vote codes (built by the first call: one thread) */
    if (!add_ready) { for (int a = 0; a < 256; a++) for (int b = 0; b < 256; b++) add[a][b] = vote_add((uint8_t)a, (uint8_t)b); add_ready = 1; }
    size_t il = 2; for (int i = 0; i < nin; i++) il += strlen(names ? names[i] : "?") + 4;
    char *at = malloc(il + 512), lv[32], od[1400];
    if (!at) return fail("side-votes: out of memory");
    int k = snprintf(at, il + 512, "{\"ufsm\":{\"content\":\"recto / verso side votes of %d surface sets\",\"encoding\":\"side-votes\","
                     "\"code\":\"src/votes.h: (r, v) recto / verso votes, diagonal code; 0 none, 1 = one verso, 2 = one recto\",\"inputs\":[", nin);
    for (int i = 0; i < nin; i++) k += snprintf(at + k, il + 512 - (size_t)k, "%s\"%s\"", i ? "," : "", names ? names[i] : "?");
    snprintf(at + k, il + 512 - (size_t)k, "]}}");
    z3w_level_name(um, lv, sizeof lv); snprintf(od, sizeof od, "%s/%s", out_group, lv);
    z3w *w = z3w_create(od, m->shape, S, 0, 0, at);
    free(at);
    if (!w || z3w_write_group(out_group, &um, 1, "ufsm-side-votes", nullptr)) return fail("side-votes: %s", z3w_error());
    uint8_t *acc = malloc((size_t)S * S * S), *buf = malloc((size_t)S * S * S);
    if (!acc || !buf) { free(acc); free(buf); z3w_close(w); return fail("side-votes: out of memory"); }
    int nsh = 0, rc = 0; stats[0] = stats[1] = 0;
    for (int64_t sz = 0; !rc && sz * S < m->shape[0]; sz++) for (int64_t sy = 0; !rc && sy * S < m->shape[1]; sy++) for (int64_t sx = 0; !rc && sx * S < m->shape[2]; sx++) {
        const int64_t o[3] = {sz * S, sy * S, sx * S};
        const int64_t n[3] = {m->shape[0] - o[0] < S ? m->shape[0] - o[0] : S, m->shape[1] - o[1] < S ? m->shape[1] - o[1] : S, m->shape[2] - o[2] < S ? m->shape[2] - o[2] : S};
        const size_t nv = (size_t)n[0] * n[1] * n[2]; int any = 0;
        for (int i = 0; i < nin && !rc; i++) {   /* an input with no shard under this one adds nothing */
            if (!any_shard_under(in[i], o, n)) continue;
            if (z3_read(in[i], o, n, any ? buf : acc, nt)) { rc = fail("side-votes: %s: %s", names ? names[i] : "?", z3_error()); break; }
            if (any) for (size_t q = 0; q < nv; q++) acc[q] = add[acc[q]][buf[q]];
            any = 1;
        }
        if (rc || !any) continue;
        for (size_t q = 0; q < nv; q++) if (acc[q]) { int r, v; vote_split(acc[q], &r, &v); stats[0]++; stats[1] += r && v; }
        memset(buf, 0, (size_t)S * S * S);   /* the n-sized read into the S^3 shard layout (edge shards) */
        for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) memcpy(buf + ((size_t)z * S + y) * S, acc + ((size_t)z * n[1] + y) * n[2], (size_t)n[2]);
        if (z3w_write_shard(w, sz, sy, sx, buf, nt)) rc = fail("side-votes: %s", z3w_error());
        else nsh++;
    }
    z3w_close(w); free(acc); free(buf);
    if (nshards) *nshards = nsh;
    return rc;
}
