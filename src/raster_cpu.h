/* Private CPU raster helpers, included after mesh/rjob in ingest.c. */

/* Same 3-4-5 transform as the scalar reference. Neighbouring rows are final
   for this pass; only the final within-row sweep carries a dependency. */
static void raster_distance(uint8_t *d, int N, int cap) {
    const size_t NN = (size_t)N * N;
    for (int dir = 1; dir >= -1; dir -= 2)
        for (int zi = 0; zi < N; zi++) {
            int z = dir > 0 ? zi : N - 1 - zi, zp = z - dir;
            for (int yi = 0; yi < N; yi++) {
                int y = dir > 0 ? yi : N - 1 - yi, yp = y - dir;
                uint8_t *row = d + (size_t)z * NN + (size_t)y * N;
                const uint8_t *nb[4] = {nullptr, nullptr, nullptr, nullptr};
                const int wf[4] = {3, 3, 4, 4}, we[4] = {4, 4, 5, 5};
                if (yp >= 0 && yp < N) nb[0] = d + (size_t)z * NN + (size_t)yp * N;
                if (zp >= 0 && zp < N) {
                    nb[1] = d + (size_t)zp * NN + (size_t)y * N;
                    if (y > 0) nb[2] = nb[1] - N;
                    if (y + 1 < N) nb[3] = nb[1] + N;
                }
                for (int q = 0; q < 4; q++) {
                    const uint8_t *r = nb[q]; if (!r) continue;
                    for (int x = 0; x < N; x++) {
                        int best = row[x], v = r[x] + wf[q];
                        if (v < best) best = v;
                        if (x > 0) { v = r[x - 1] + we[q]; if (v < best) best = v; }
                        if (x + 1 < N) { v = r[x + 1] + we[q]; if (v < best) best = v; }
                        row[x] = (uint8_t)(best > cap ? cap : best);
                    }
                }
                for (int xi = 0; xi < N; xi++) {
                    int x = dir > 0 ? xi : N - 1 - xi, xp = x - dir;
                    int best = row[x];
                    if (xp >= 0 && xp < N && row[xp] + 3 < best) best = row[xp] + 3;
                    row[x] = (uint8_t)(best > cap ? cap : best);
                }
            }
        }
}

/* A quad belongs to one 32x32-cell tile. Include the shared bottom/right
   vertex row in its bounds, and assign it to every padded shard it touches. */
static int raster_index(rjob *j) {
    size_t capacity = 0;
    for (int mi = 0; mi < j->nm; mi++) {
        mesh *m = j->meshes[mi];
        for (int r0 = 0; r0 + 1 < m->h; r0 += 32) for (int c0 = 0; c0 + 1 < m->w; c0 += 32) {
            rtile t = {.mi = mi, .r0 = r0, .c0 = c0, .r1 = r0 + 32, .c1 = c0 + 32};
            if (t.r1 >= m->h) t.r1 = m->h - 1;
            if (t.c1 >= m->w) t.c1 = m->w - 1;
            for (int d = 0; d < 3; d++) { t.lo[d] = 1e30; t.hi[d] = -1e30; }
            int any = 0;
            for (int r = r0; r <= t.r1; r++) for (int c = c0; c <= t.c1; c++) {
                size_t k = (size_t)r * m->w + c; if (!m->valid[k]) continue;
                any = 1;
                for (int a = 0; a < 3; a++) {
                    int d = 2 - a; double v = m->xyz[k * 3 + a] * j->scale;
                    if (v < t.lo[d]) t.lo[d] = v;
                    if (v > t.hi[d]) t.hi[d] = v;
                }
            }
            if (!any) continue;
            if (j->ntile == capacity) {
                capacity = capacity ? capacity * 2 : 1024;
                rtile *p = realloc(j->tiles, capacity * sizeof *p);
                if (!p) return -1;
                j->tiles = p;
            }
            j->tiles[j->ntile++] = t;
        }
    }
    if (j->ntile > UINT32_MAX) return -1;
    size_t ns = (size_t)j->ns[0] * j->ns[1] * j->ns[2];
    j->offset = calloc(ns + 1, sizeof *j->offset);
    if (!j->offset) return -1;
    for (int pass = 0; pass < 2; pass++) {
        size_t *cursor = pass ? calloc(ns, sizeof *cursor) : nullptr;
        if (pass && !cursor) return -1;
        for (size_t ti = 0; ti < j->ntile; ti++) {
            const rtile *t = &j->tiles[ti]; int64_t lo[3], hi[3];
            for (int d = 0; d < 3; d++) {
                lo[d] = (int64_t)floor((t->lo[d] - j->margin - 1) / j->shard);
                hi[d] = (int64_t)floor((t->hi[d] + j->margin) / j->shard);
                if (lo[d] < 0) lo[d] = 0;
                if (hi[d] >= j->ns[d]) hi[d] = j->ns[d] - 1;
            }
            for (int64_t z = lo[0]; z <= hi[0]; z++) for (int64_t y = lo[1]; y <= hi[1]; y++) for (int64_t x = lo[2]; x <= hi[2]; x++) {
                size_t s = ((size_t)z * j->ns[1] + y) * j->ns[2] + x;
                if (!pass) j->offset[s + 1]++;
                else j->refs[j->offset[s] + cursor[s]++] = (uint32_t)ti;
            }
        }
        if (!pass) {
            for (size_t s = 0; s < ns; s++) j->offset[s + 1] += j->offset[s];
            j->refs = malloc((j->offset[ns] ? j->offset[ns] : 1) * sizeof *j->refs);
            if (!j->refs) return -1;
        }
        free(cursor);
    }
    size_t active = 0; for (size_t s = 0; s < ns; s++) active += j->offset[s + 1] > j->offset[s];
    fprintf(stderr, "raster index: %zu mesh tiles, %zu references, %zu/%zu active shards\n", j->ntile, j->offset[ns], active, ns);
    return 0;
}
