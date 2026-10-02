/* Independent dilation and threshold-counter references, including ignored labels
   and prediction support across ignored voxels. */
#include "../src/eval.c"
#include <assert.h>
static void reference(const uint8_t *in, uint8_t *out, const int64_t n[3], int tol) {
    memcpy(out, in, (size_t)n[0]*n[1]*n[2]);
    for (int64_t z=0; z<n[0]; z++) for (int64_t y=0; y<n[1]; y++) for (int64_t x=0; x<n[2]; x++) {
        if (!in[(z*n[1]+y)*n[2]+x]) continue;
        for (int dz=-tol; dz<=tol; dz++) for (int dy=-tol; dy<=tol; dy++) for (int dx=-tol; dx<=tol; dx++) {
            if (dz*dz+dy*dy+dx*dx>tol*tol) continue;
            int64_t zz=z+dz, yy=y+dy, xx=x+dx;
            if (zz>=0 && zz<n[0] && yy>=0 && yy<n[1] && xx>=0 && xx<n[2]) out[(zz*n[1]+yy)*n[2]+xx]=1;
        }
    }
}
static void reference_max(const uint8_t *in, uint8_t *out, const int64_t n[3], int tol) {
    memcpy(out, in, (size_t)n[0] * n[1] * n[2]);
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) {
        uint8_t value = in[(z * n[1] + y) * n[2] + x];
        for (int dz = -tol; dz <= tol; dz++) for (int dy = -tol; dy <= tol; dy++) for (int dx = -tol; dx <= tol; dx++) {
            if (dz * dz + dy * dy + dx * dx > tol * tol) continue;
            int64_t zz = z + dz, yy = y + dy, xx = x + dx;
            if (zz >= 0 && zz < n[0] && yy >= 0 && yy < n[1] && xx >= 0 && xx < n[2]) {
                size_t i = (zz * n[1] + yy) * n[2] + xx;
                if (out[i] < value) out[i] = value;
            }
        }
    }
}
static unsigned rng = 1234;
static uint8_t random_byte(void) { rng = rng * 1664525 + 1013904223; return rng >> 24; }
static void check_histogram(void) {
    int filters = 0, scores = 0;
    for (int z = 1; z <= 12; z++) for (int y = 1; y <= 9; y++) for (int x = 1; x <= 15; x++) for (int tol = 0; tol <= 3; tol++) {
        int64_t n[3] = {z, y, x}; size_t nv = (size_t)z * y * x;
        uint8_t p[nv], a[nv], b[nv];
        for (size_t i = 0; i < nv; i++) p[i] = random_byte();
        reference_max(p, a, n, tol); max_dilate(p, b, n, tol);
        assert(!memcmp(a, b, nv)); filters++;
    }
    for (int tol = 4; tol <= 32; tol *= 2) {
        int64_t n[3] = {2, 3, 4}; uint8_t p[24], a[24], b[24];
        for (size_t i = 0; i < sizeof p; i++) p[i] = random_byte();
        reference_max(p, a, n, tol); max_dilate(p, b, n, tol);
        assert(!memcmp(a, b, sizeof p)); filters++;
    }
    for (int test = 0; test < 180; test++) {
        int64_t n[3] = {1 + test % 7, 1 + test % 9, 1 + test % 13};
        size_t nv = (size_t)n[0] * n[1] * n[2], nvalid = 0, npos = 0;
        int tol = test % 5;
        uint8_t p[nv], l[nv], gt[nv], gtd[nv], pm[nv], mask[nv], pr[nv], pd[nv];
        for (size_t i = 0; i < nv; i++) {
            p[i] = test % 11 == 0 ? 0 : test % 13 == 0 ? 255 : random_byte();
            uint8_t value = random_byte();
            l[i] = test % 17 == 0 ? 255 : test % 19 == 0 ? 0 : value < 30 ? 255 : value < 90 ? 254 : 0;
            gt[i] = l[i] != 255 && l[i] >= 127; mask[i] = 1 + (i % 3 == 0);
            nvalid += l[i] != 255; npos += gt[i];
        }
        reference(gt, gtd, n, tol); max_dilate(p, pm, n, tol);
        region_counts bins[3][256]; uint8_t *region = test % 2 ? mask : nullptr;
        score_histogram(p, l, gtd, pm, region, nv, bins);
        for (int th = 0; th < 256; th++) {
            for (size_t i = 0; i < nv; i++) pr[i] = p[i] >= th;
            reference(pr, pd, n, tol);
            region_counts ref[3] = {0}, direct[3] = {0};
            for (size_t i = 0; i < nv; i++) if (l[i] != 255) {
                int regions[2] = {0, region ? region[i] : 0};
                for (int j = 0; j < (region ? 2 : 1); j++) {
                    region_counts *c = &ref[regions[j]]; c->valid++; c->positive += gt[i];
                    c->tp += pr[i] && gt[i]; c->fp += pr[i] && !gt[i]; c->fn += !pr[i] && gt[i];
                    c->bpt += pr[i]; c->bph += pr[i] && gtd[i]; c->brt += gt[i]; c->brh += gt[i] && pd[i];
                }
            }
            score_threshold(p, l, gt, gtd, region, nv, nvalid, npos, th, pr, pd, n, tol, direct);
            for (int r = 0; r < 3; r++) {
                assert(!memcmp(&ref[r], &bins[r][th], sizeof(region_counts)));
                assert(!memcmp(&ref[r], &direct[r], sizeof(region_counts)));
            }
            scores++;
        }
    }
    /* An ignored prediction still recalls a neighbouring labelled positive. */
    int64_t n[3] = {1, 1, 3};
    uint8_t p[] = {255, 0, 0}, l[] = {255, 254, 0}, gtd[] = {1, 1, 1}, pm[3];
    region_counts bins[3][256];
    max_dilate(p, pm, n, 1); score_histogram(p, l, gtd, pm, nullptr, 3, bins);
    assert(bins[0][255].tp == 0 && bins[0][255].fp == 0 && bins[0][255].brh == 1 && bins[0][255].brt == 1);
    printf("grayscale filters: %d exact thin/edge/radius cases; histogram and direct scores: %d exact cases over all 256 cutoffs, ignore masks and seam/interior counters\n", filters, scores);
}
int main(void) {
    unsigned state=1234; int cases=0;
    for (int z=1; z<=12; z++) for (int y=1; y<=9; y++) for (int x=1; x<=15; x++) for (int tol=0; tol<=3; tol++) {
        int64_t n[3]={z,y,x}; size_t nv=(size_t)z*y*x;
        uint8_t a[nv], b[nv], c[nv];
        for (size_t i=0;i<nv;i++) {state=state*1664525+1013904223; a[i]=(state>>24)<64;}
        reference(a,b,n,tol); dilate(a,c,n,tol); assert(!memcmp(b,c,nv)); cases++;
    }
    printf("dilation: %d boundary, thin-volume and radius cases bit-exact\n",cases);
    for (int n=1; n<=70; n++) for (int core=2; core<=20; core++) for (int band=1; band<=core/2; band++) for (int q=0; q<n; q++) {
        int hit=0; for (int plane=core; plane<n; plane+=core) hit |= fabs(q+0.5-plane)<band;
        assert(near_seam(q,n,core,band)==hit);
    }
    puts("internal seam masks: exact including partial last tiles and outer boundaries");
    check_histogram();
}
