/* Image operations of the render pipeline (see imgops.h). */
#include "imgops.h"
#include "tiff.h"
#include "tiffw.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <jpeglib.h>   /* after stdio.h: it needs FILE */
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

/* ---- Pillow resampling (libImaging/Resample.c, 8 bits per channel) ---- */
static double pil_filter(int f, double x) {
    if (f == PIL_BOX) return x > -0.5 && x <= 0.5 ? 1.0 : 0.0;
    if (x < 0.0) x = -x;
    return x < 1.0 ? 1.0 - x : 0.0;
}
int pil_coeffs(int in, int out, int filter, int **boundsp, int32_t **kkp) {
    const float in0 = 0.f, in1 = (float)in;
    double scale, filterscale; filterscale = scale = (double)(in1 - in0) / out;
    if (filterscale < 1.0) filterscale = 1.0;
    const double support = (filter == PIL_BOX ? 0.5 : 1.0) * filterscale;
    const int ksize = (int)ceil(support) * 2 + 1;
    double *kd = calloc((size_t)out * ksize, sizeof *kd); int *bounds = malloc((size_t)out * 2 * sizeof *bounds);
    int32_t *kk = malloc((size_t)out * ksize * sizeof *kk);
    if (!kd || !bounds || !kk) { free(kd); free(bounds); free(kk); return 0; }
    for (int xx = 0; xx < out; xx++) {
        const double center = in0 + (xx + 0.5) * scale, ss = 1.0 / filterscale; double ww = 0.0;
        int xmin = (int)(center - support + 0.5); if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5); if (xmax > in) xmax = in;
        xmax -= xmin;
        double *k = kd + (size_t)xx * ksize; int x;
        for (x = 0; x < xmax; x++) { const double w = pil_filter(filter, (x + xmin - center + 0.5) * ss); k[x] = w; ww += w; }
        for (x = 0; x < xmax; x++) if (ww != 0.0) k[x] /= ww;
        for (; x < ksize; x++) k[x] = 0;
        bounds[xx * 2] = xmin; bounds[xx * 2 + 1] = xmax;
    }
    for (size_t i = 0; i < (size_t)out * ksize; i++) kk[i] = kd[i] < 0 ? (int)(-0.5 + kd[i] * (1 << 22)) : (int)(0.5 + kd[i] * (1 << 22));
    free(kd); *boundsp = bounds; *kkp = kk;
    return ksize;
}
void pil_hrow(const uint8_t *in, int ch, uint8_t *out, int out_w, int ksize, const int *bounds, const int32_t *kk) {
    for (int xx = 0; xx < out_w; xx++) {
        const int xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1]; const int32_t *k = kk + (size_t)xx * ksize;
        for (int c = 0; c < ch; c++) {
            int ss = 1 << 21; const uint8_t *p = in + (size_t)xmin * ch + c;
            for (int x = 0; x < xmax; x++) ss += p[(size_t)x * ch] * k[x];
            out[(size_t)xx * ch + c] = pil_clip8(ss);
        }
    }
}
/* the vertical pass of one output row y from the horizontally resampled rows hp (row r of hp = input row r0 + r) */
static void pil_vrow(const uint8_t *hp, size_t hstride, int r0, int n, uint8_t *out, int y, int ksize, const int *bounds, const int32_t *kk) {
    const int ymin = bounds[y * 2] - r0, ymax = bounds[y * 2 + 1]; const int32_t *k = kk + (size_t)y * ksize;
    for (int i = 0; i < n; i++) {
        int ss = 1 << 21;
        for (int r = 0; r < ymax; r++) ss += hp[(size_t)(ymin + r) * hstride + i] * k[r];
        out[i] = pil_clip8(ss);
    }
}

typedef struct {
    const uint8_t *in; int w, h, ch; uint8_t *out; int ow, oh;
    int kh, kv; int *bh, *bv; int32_t *kkh, *kkv; int rows; _Atomic int next;
} rs_job;
static void *rs_worker(void *arg) {   /* strips of rs->rows output rows: their input rows horizontally, then vertically */
    rs_job *J = arg; const int ch = J->ch; const size_t hs = (size_t)J->ow * ch;
    uint8_t *hp = nullptr; size_t cap = 0;
    for (int s; (s = atomic_fetch_add(&J->next, 1)) * J->rows < J->oh;) {
        const int y0 = s * J->rows, y1 = y0 + J->rows < J->oh ? y0 + J->rows : J->oh;
        int r0 = J->bv[y0 * 2], r1 = r0;
        for (int y = y0; y < y1; y++) { if (J->bv[y * 2] < r0) r0 = J->bv[y * 2]; if (J->bv[y * 2] + J->bv[y * 2 + 1] > r1) r1 = J->bv[y * 2] + J->bv[y * 2 + 1]; }
        if ((size_t)(r1 - r0) * hs > cap) { free(hp); cap = (size_t)(r1 - r0) * hs; hp = malloc(cap); if (!hp) abort(); }
        for (int r = r0; r < r1; r++) pil_hrow(J->in + (size_t)r * J->w * ch, ch, hp + (size_t)(r - r0) * hs, J->ow, J->kh, J->bh, J->kkh);
        for (int y = y0; y < y1; y++) pil_vrow(hp, hs, r0, (int)hs, J->out + (size_t)y * hs, y, J->kv, J->bv, J->kkv);
    }
    free(hp);
    return nullptr;
}
int pil_resize(const uint8_t *in, int w, int h, int ch, uint8_t *out, int ow, int oh, int filter, int nthreads) {
    rs_job J = {in, w, h, ch, out, ow, oh, 0, 0, nullptr, nullptr, nullptr, nullptr, 16, 0};
    J.kh = pil_coeffs(w, ow, filter, &J.bh, &J.kkh); J.kv = pil_coeffs(h, oh, filter, &J.bv, &J.kkv);
    if (!J.kh || !J.kv) { free(J.bh); free(J.kkh); free(J.bv); free(J.kkv); return -1; }
    /* Pillow skips a pass whose size is unchanged: the identity coefficients (one weight 1 << 22) give the same bytes */
    const int nt = nthreads < 1 ? 1 : nthreads > 64 ? 64 : nthreads; pthread_t th[64];
    for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, rs_worker, &J);
    for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
    free(J.bh); free(J.kkh); free(J.bv); free(J.kkv);
    return 0;
}

/* ---- the cross dilation as an L1 distance (two chamfer passes, exact for the 4-neighbour metric) ---- */
void l1_dilate_zero(const uint8_t *ct, int w, int h, int r, uint8_t *mask) {
    const int big = r + 1; uint8_t *d = mask;   /* distances capped at r + 1 (<= 255 for r < 255), then thresholded in place */
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        const size_t i = (size_t)y * w + x; int v = ct[i] == 0 ? 0 : big;
        if (v && x && d[i - 1] + 1 < v) v = d[i - 1] + 1;
        if (v && y && d[i - w] + 1 < v) v = d[i - w] + 1;
        d[i] = (uint8_t)v;
    }
    for (int y = h - 1; y >= 0; y--) for (int x = w - 1; x >= 0; x--) {
        const size_t i = (size_t)y * w + x; int v = d[i];
        if (v && x + 1 < w && d[i + 1] + 1 < v) v = d[i + 1] + 1;
        if (v && y + 1 < h && d[i + w] + 1 < v) v = d[i + w] + 1;
        d[i] = (uint8_t)v;
    }
    for (size_t i = 0; i < (size_t)w * h; i++) mask[i] = d[i] <= r;
}

/* ---- the red-ink overlay ---- */
int overlay_red_skip(const uint8_t *ct, size_t n) {
    size_t k = 0; for (size_t i = 0; i < n; i++) k += ct[i] > 0;
    return (double)k / (double)n < 0.05;
}
typedef struct {
    const uint8_t *ct, *mask; int W, H, nink, TW;
    const uint8_t *hink[3]; int kiv; int *biv; int32_t *kkiv;   /* the inks after Pillow's horizontal pass (W x ih), vertical coefficients */
    int ow, oh, kh, kv; int *bh, *bv; int32_t *kkh, *kkv; uint8_t *out; _Atomic int next;
} ov_job;
static void ov_compose_row(const ov_job *J, int y, uint8_t *row, uint8_t *ink) {   /* composite row y (TW x 3) */
    const int W = J->W; const uint8_t *c = J->ct + (size_t)y * W, *m = J->mask + (size_t)y * W;
    for (int k = 0; k < J->nink; k++) pil_vrow(J->hink[k], (size_t)W, 0, W, ink + (size_t)k * W, y, J->kiv, J->biv, J->kkiv);
    memset(row, 255, (size_t)J->TW * 3);
    for (int x = 0; x < W; x++) {
        const float cf = (float)c[x];
        float g = (cf - 40.f) * 2.2f; g = g < 0.f ? 0.f : g > 255.f ? 255.f : g;
        const uint8_t gu = (uint8_t)g; row[3 * x] = row[3 * x + 1] = row[3 * x + 2] = gu;
        float i = (float)ink[x] / 255.f; if (m[x]) i = 0.f;
        float a = (i - 0.15f) / 0.5f; a = a < 0.f ? 0.f : a > 1.f ? 1.f : a; a = a * 0.85f;
        const float ga = g * (1.f - a); uint8_t *o = row + 3 * ((size_t)W + 8 + x);
        o[0] = (uint8_t)(ga + 235.f * a); o[1] = (uint8_t)(ga + 25.f * a); o[2] = (uint8_t)(ga + 25.f * a);
    }
    for (int k = 0; k < J->nink; k++) {
        uint8_t *o = row + 3 * ((size_t)(k + 2) * (W + 8)); const uint8_t *s = ink + (size_t)k * W;
        for (int x = 0; x < W; x++) {
            float r = ((float)s[x] / 255.f) * 255.f; r = r < 0.f ? 0.f : r > 255.f ? 255.f : r;
            o[3 * x] = o[3 * x + 1] = o[3 * x + 2] = (uint8_t)r;
        }
    }
}
static void *ov_worker(void *arg) {
    ov_job *J = arg; const size_t hs = (size_t)J->ow * 3; const int S = 16;
    uint8_t *row = malloc((size_t)J->TW * 3), *ink = malloc((size_t)J->W * 3), *hp = nullptr; size_t cap = 0;
    if (!row || !ink) abort();
    for (int s; (s = atomic_fetch_add(&J->next, 1)) * S < J->oh;) {
        const int y0 = s * S, y1 = y0 + S < J->oh ? y0 + S : J->oh;
        int r0 = J->bv[y0 * 2], r1 = r0;
        for (int y = y0; y < y1; y++) { if (J->bv[y * 2] < r0) r0 = J->bv[y * 2]; if (J->bv[y * 2] + J->bv[y * 2 + 1] > r1) r1 = J->bv[y * 2] + J->bv[y * 2 + 1]; }
        if ((size_t)(r1 - r0) * hs > cap) { free(hp); cap = (size_t)(r1 - r0) * hs; hp = malloc(cap); if (!hp) abort(); }
        for (int r = r0; r < r1; r++) { ov_compose_row(J, r, row, ink); pil_hrow(row, 3, hp + (size_t)(r - r0) * hs, J->ow, J->kh, J->bh, J->kkh); }
        for (int y = y0; y < y1; y++) pil_vrow(hp, hs, r0, (int)hs, J->out + (size_t)y * hs, y, J->kv, J->bv, J->kkv);
    }
    free(row); free(ink); free(hp);
    return nullptr;
}
uint8_t *overlay_red_half(const uint8_t *ct, int W, int H, const uint8_t *const *ink, int nink, int iw, int ih, int *owp, int *ohp, int nthreads) {
    if (nink < 1 || nink > 3) return nullptr;
    ov_job J = {0}; J.ct = ct; J.W = W; J.H = H; J.nink = nink; J.TW = W * (nink + 2) + 8 * (nink + 1);
    J.ow = J.TW / 2; J.oh = H / 2;
    uint8_t *mask = malloc((size_t)W * H), *hink = malloc((size_t)nink * W * ih), *out = malloc((size_t)J.ow * J.oh * 3);
    int *bih = nullptr; int32_t *kkih = nullptr;
    const int kih = pil_coeffs(iw, W, PIL_BILINEAR, &bih, &kkih);
    J.kiv = pil_coeffs(ih, H, PIL_BILINEAR, &J.biv, &J.kkiv);
    J.kh = pil_coeffs(J.TW, J.ow, PIL_BOX, &J.bh, &J.kkh); J.kv = pil_coeffs(H, J.oh, PIL_BOX, &J.bv, &J.kkv);
    uint8_t *ret = nullptr;
    if (mask && hink && out && kih && J.kiv && J.kh && J.kv && J.ow > 0 && J.oh > 0) {
        l1_dilate_zero(ct, W, H, 20, mask); J.mask = mask;
        for (int k = 0; k < nink; k++) {
            uint8_t *hk = hink + (size_t)k * W * ih; J.hink[k] = hk;
            for (int r = 0; r < ih; r++) pil_hrow(ink[k] + (size_t)r * iw, 1, hk + (size_t)r * W, W, kih, bih, kkih);
        }
        J.out = out;
        const int nt = nthreads < 1 ? 1 : nthreads > 64 ? 64 : nthreads; pthread_t th[64];
        for (int t = 0; t < nt; t++) pthread_create(&th[t], nullptr, ov_worker, &J);
        for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
        ret = out; out = nullptr; *owp = J.ow; *ohp = J.oh;
    }
    free(mask); free(hink); free(out); free(bih); free(kkih); free(J.biv); free(J.kkiv); free(J.bh); free(J.kkh); free(J.bv); free(J.kkv);
    return ret;
}

/* ---- JPEG (Pillow's save(quality=q) defaults: baseline, 4:2:0, standard Huffman tables, JFIF) ---- */
int jpeg_write_rgb(const char *path, int w, int h, const uint8_t *rgb, int quality) {
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    struct jpeg_compress_struct ci; struct jpeg_error_mgr je;
    ci.err = jpeg_std_error(&je); jpeg_create_compress(&ci); jpeg_stdio_dest(&ci, f);
    ci.image_width = (JDIMENSION)w; ci.image_height = (JDIMENSION)h; ci.input_components = 3; ci.in_color_space = JCS_RGB;
    jpeg_set_defaults(&ci); jpeg_set_quality(&ci, quality, TRUE);
    jpeg_start_compress(&ci, TRUE);
    while (ci.next_scanline < ci.image_height) { JSAMPROW r = (JSAMPROW)(rgb + (size_t)ci.next_scanline * w * 3); jpeg_write_scanlines(&ci, &r, 1); }
    jpeg_finish_compress(&ci); jpeg_destroy_compress(&ci);
    return fclose(f) ? -1 : 0;
}

/* ---- PNG reading (8-bit, not interlaced) ---- */
static uint32_t be32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }
uint8_t *png_read_rgb(const char *path, int *wp, int *hp) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); const long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)n); if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return nullptr; }
    fclose(f);
    static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    uint8_t *idat = nullptr, *rgb = nullptr, *raw = nullptr, pal[768] = {0}; size_t ni = 0; int w = 0, h = 0, depth = 0, ct = -1, il = 0;
    if (n < 8 || memcmp(b, sig, 8)) goto done;
    for (long p = 8; p + 12 <= n;) {
        const uint32_t len = be32(b + p); const uint8_t *t = b + p + 4, *d = b + p + 8;
        if (p + 12 + (long)len > n) goto done;
        if (!memcmp(t, "IHDR", 4) && len >= 13) { w = (int)be32(d); h = (int)be32(d + 4); depth = d[8]; ct = d[9]; il = d[12]; }
        else if (!memcmp(t, "PLTE", 4)) memcpy(pal, d, len < 768 ? len : 768);
        else if (!memcmp(t, "IDAT", 4)) { uint8_t *g = realloc(idat, ni + len); if (!g) goto done; idat = g; memcpy(idat + ni, d, len); ni += len; }
        else if (!memcmp(t, "IEND", 4)) break;
        p += 12 + (long)len;
    }
    const int cc = ct == 0 ? 1 : ct == 2 ? 3 : ct == 3 ? 1 : ct == 4 ? 2 : ct == 6 ? 4 : 0;
    if (depth != 8 || !cc || il || w <= 0 || h <= 0) goto done;
    const size_t rs = (size_t)w * cc, rawn = (rs + 1) * h; uLongf dn = rawn;
    raw = malloc(rawn); if (!raw || uncompress(raw, &dn, idat, ni) != Z_OK || dn != rawn) goto done;
    uint8_t *cur = malloc(rs), *prev = calloc(rs, 1); rgb = malloc((size_t)w * h * 3);
    if (!cur || !prev || !rgb) { free(cur); free(prev); free(rgb); rgb = nullptr; goto done; }
    for (int y = 0; y < h; y++) {
        const uint8_t *s = raw + (rs + 1) * y + 1; const int ft = s[-1];
        for (size_t i = 0; i < rs; i++) {
            const int a = i >= (size_t)cc ? cur[i - cc] : 0, up = prev[i], c = i >= (size_t)cc ? prev[i - cc] : 0; int v = s[i];
            if (ft == 1) v += a; else if (ft == 2) v += up; else if (ft == 3) v += (a + up) / 2;
            else if (ft == 4) { const int pp = a + up - c, pa = abs(pp - a), pb = abs(pp - up), pc = abs(pp - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? up : c; }
            cur[i] = (uint8_t)v;
        }
        uint8_t *o = rgb + (size_t)y * w * 3;
        for (int x = 0; x < w; x++) {
            const uint8_t *q = cur + (size_t)x * cc;
            if (ct == 3) memcpy(o + 3 * x, pal + 3 * q[0], 3);
            else if (cc <= 2) o[3 * x] = o[3 * x + 1] = o[3 * x + 2] = q[0];
            else memcpy(o + 3 * x, q, 3);
        }
        uint8_t *t = prev; prev = cur; cur = t;
    }
    free(cur); free(prev); *wp = w; *hp = h;
done:
    free(b); free(idat); free(raw);
    return rgb;
}

/* ---- tifxyz rescale ---- */
float *img_read_f32(const char *path, int *w, int *h) {
    tiff *t = tiff_open_file(path); tiff_page pg;
    if (!t || tiff_page_info(t, 0, &pg) || pg.fmt != 3 || pg.bits != 32 || pg.spp != 1) { if (t) tiff_close(t); return nullptr; }
    float *d = malloc((size_t)pg.w * pg.h * 4);
    if (!d || tiff_read_page(t, 0, d)) { free(d); tiff_close(t); return nullptr; }
    *w = pg.w; *h = pg.h; tiff_close(t); return d;
}
uint8_t *img_read_u8(const char *path, int *w, int *h) {
    tiff *t = tiff_open_file(path); tiff_page pg;
    if (!t || tiff_page_info(t, 0, &pg) || pg.bits != 8 || pg.spp != 1) { if (t) tiff_close(t); return nullptr; }
    uint8_t *d = malloc((size_t)pg.w * pg.h);
    if (!d || tiff_read_page(t, 0, d)) { free(d); tiff_close(t); return nullptr; }
    *w = pg.w; *h = pg.h; tiff_close(t); return d;
}
/* meta.json with "scale": [a, b] -> [a div, b div] and "voxel_size_um": u -> u div (the rest verbatim) */
static char *meta_scaled(const char *m, double div) {
    const size_t n = strlen(m); char *o = malloc(n + 256); if (!o) return nullptr;
    const char *s = strstr(m, "\"scale\":"), *v = strstr(m, "\"voxel_size_um\":"); size_t k = 0;
    for (const char *p = m; *p;) {
        if (p == s) {
            const char *q = strchr(p, '['), *e = q ? strchr(q, ']') : nullptr; if (!e) { free(o); return nullptr; }
            char *x; const double a = strtod(q + 1, &x), b = strtod(x + 1 + strspn(x + 1, " "), nullptr);
            k += (size_t)sprintf(o + k, "\"scale\": [%.17g, %.17g]", a * div, b * div); p = e + 1; continue;
        }
        if (p == v) {
            char *x; const double u = strtod(p + 16, &x);
            k += (size_t)sprintf(o + k, "\"voxel_size_um\": %.6g", u * div); p = x; continue;
        }
        o[k++] = *p++;
    }
    o[k] = 0; return o;
}
int tifxyz_scale(const char *src, const char *dst, double div, const double off[3]) {
    char p[1400]; static const char *nm[3] = {"x", "y", "z"}; float *A[3] = {0}; int w = 0, h = 0, rc = -1;
    for (int k = 0; k < 3; k++) {
        int w2, h2; snprintf(p, sizeof p, "%s/%s.tif", src, nm[k]); A[k] = img_read_f32(p, &w2, &h2);
        if (!A[k] || (k && (w2 != w || h2 != h))) goto out;
        w = w2; h = h2;
    }
    snprintf(p, sizeof p, "mkdir -p '%s'", dst); if (system(p)) goto out;
    const float df = (float)div, of[3] = {(float)off[2], (float)off[1], (float)off[0]};   /* x, y, z */
    const size_t n = (size_t)w * h; float *o = malloc(n * 4); if (!o) goto out;
    for (int k = 0; k < 3; k++) {
        for (size_t i = 0; i < n; i++) o[i] = tifxyz_scaled(A[k][i], A[2][i], df, of[k]);
        snprintf(p, sizeof p, "%s/%s.tif", dst, nm[k]); if (tiffw_write(p, w, h, 3, o, 1)) { free(o); goto out; }
    }
    free(o);
    snprintf(p, sizeof p, "%s/meta.json", src); FILE *f = fopen(p, "rb"); if (!f) goto out;
    char mb[16384]; const size_t mn = fread(mb, 1, sizeof mb - 1, f); fclose(f); mb[mn] = 0;
    char *ms = meta_scaled(mb, div); if (!ms) goto out;
    snprintf(p, sizeof p, "%s/meta.json", dst); f = fopen(p, "w");
    if (f) { fputs(ms, f); rc = fclose(f) ? -1 : 0; }
    free(ms);
out:
    for (int k = 0; k < 3; k++) free(A[k]);
    return rc;
}
