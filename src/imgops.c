/* Image operations of the render pipeline (see imgops.h). */
#include "imgops.h"
#include "tiff.h"
#include "tiffw.h"
#include <unistd.h>

#include <dirent.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <jpeglib.h>   /* after stdio.h: it needs FILE */
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
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
    if (tfx_axis_hook && access(path, F_OK)) return tfx_axis_hook(path, w, h);   /* a .sfc-backed wrap (src/tfx.c) */
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
int tifxyz_shift(const char *src, const char *dst, double d, const double *toward) {
    char p[1400]; static const char *nm[3] = {"x", "y", "z"}; float *A[3] = {0}, *O[3] = {0}, *Nv[3] = {0}; int w = 0, h = 0, rc = -1;
    for (int k = 0; k < 3; k++) {
        int w2, h2; snprintf(p, sizeof p, "%s/%s.tif", src, nm[k]); A[k] = img_read_f32(p, &w2, &h2);
        if (!A[k] || (k && (w2 != w || h2 != h))) goto out;
        w = w2; h = h2;
    }
    const size_t n = (size_t)w * h; double vote = 0;
    for (int k = 0; k < 3; k++) { if (!(O[k] = malloc(n * 4)) || !(Nv[k] = calloc(n, 4))) goto out; memcpy(O[k], A[k], n * 4); }
    for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
        const size_t i = (size_t)r * w + c; if (!(A[2][i] >= 0)) continue;
        /* central (or one-sided) differences along the columns (u) and the rows (v) over valid neighbours; +n = u x v */
        const int c0 = c > 0 && A[2][i - 1] >= 0 ? c - 1 : c, c1 = c + 1 < w && A[2][i + 1] >= 0 ? c + 1 : c;
        const int r0 = r > 0 && A[2][i - w] >= 0 ? r - 1 : r, r1 = r + 1 < h && A[2][i + w] >= 0 ? r + 1 : r;
        if (c0 == c1 || r0 == r1) continue;
        double u[3], v[3];
        for (int k = 0; k < 3; k++) { u[k] = A[k][(size_t)r * w + c1] - A[k][(size_t)r * w + c0]; v[k] = A[k][(size_t)r1 * w + c] - A[k][(size_t)r0 * w + c]; }
        const double nx = u[1] * v[2] - u[2] * v[1], ny = u[2] * v[0] - u[0] * v[2], nz = u[0] * v[1] - u[1] * v[0], l = sqrt(nx * nx + ny * ny + nz * nz);
        if (!(l > 0)) continue;
        Nv[0][i] = (float)(nx / l); Nv[1][i] = (float)(ny / l); Nv[2][i] = (float)(nz / l);
        if (toward) vote += Nv[1][i] * (toward[0] - A[1][i]) + Nv[0][i] * (toward[1] - A[0][i]) > 0 ? 1 : -1;
    }
    const double sg = toward && vote < 0 ? -1 : 1;   /* toward (y, x): +n made the side facing that line (the scroll axis: inward) */
    for (size_t i = 0; i < n; i++) {
        if (!(A[2][i] >= 0) || (Nv[0][i] == 0 && Nv[1][i] == 0 && Nv[2][i] == 0)) continue;
        for (int k = 0; k < 3; k++) O[k][i] = (float)(A[k][i] + sg * d * Nv[k][i]);
        if (O[2][i] < 0) O[2][i] = 0;   /* (stays a valid vertex) */
    }
    if (toward) fprintf(stderr, "tifxyz-scale --normal: %s: %s (vote %.0f)\n", src, sg > 0 ? "+n faces the axis" : "+n faced away from the axis: flipped", vote);
    snprintf(p, sizeof p, "mkdir -p '%s'", dst); if (system(p)) goto out;
    for (int k = 0; k < 3; k++) { snprintf(p, sizeof p, "%s/%s.tif", dst, nm[k]); if (tiffw_write(p, w, h, 3, O[k], 1)) goto out; }
    { snprintf(p, sizeof p, "%s/meta.json", src); FILE *fi = tfx_meta_fopen(p); char mb[16384]; const size_t mn = fi ? fread(mb, 1, sizeof mb, fi) : 0; if (fi) fclose(fi);
      snprintf(p, sizeof p, "%s/meta.json", dst); FILE *fo = fi ? fopen(p, "wb") : nullptr; rc = fo && fwrite(mb, 1, mn, fo) == mn ? 0 : -1; if (fo && fclose(fo)) rc = -1; }
out:
    for (int k = 0; k < 3; k++) { free(A[k]); free(O[k]); free(Nv[k]); }
    return rc;
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
    snprintf(p, sizeof p, "%s/meta.json", src); FILE *f = tfx_meta_fopen(p); if (!f) goto out;
    char mb[16384]; const size_t mn = fread(mb, 1, sizeof mb - 1, f); fclose(f); mb[mn] = 0;
    char *ms = meta_scaled(mb, div); if (!ms) goto out;
    snprintf(p, sizeof p, "%s/meta.json", dst); f = fopen(p, "w");
    if (f) { fputs(ms, f); rc = fclose(f) ? -1 : 0; }
    free(ms);
out:
    for (int k = 0; k < 3; k++) free(A[k]);
    return rc;
}

/* ---- numpy.percentile (method linear) ---- */
static int cmp_f64(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static int cmp_f32(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
double np_percentile_f64(double *v, size_t n, double q) {
    qsort(v, n, sizeof *v, cmp_f64);
    const double vi = (double)(n - 1) * (q / 100); double pf = floor(vi); size_t p, nx;
    if (vi >= (double)(n - 1)) p = nx = n - 1; else if (vi < 0) p = nx = 0; else { p = (size_t)pf; nx = p + 1; }
    const double g = vi - (vi >= (double)(n - 1) ? -1.0 : pf), a = v[p], b = v[nx], d = b - a;   /* gamma as numpy's (beyond the end: vs index -1) */
    return g >= 0.5 ? b - d * (1 - g) : a + d * g;
}
float np_percentile_f32(float *v, size_t n, float q) {
    qsort(v, n, sizeof *v, cmp_f32);
    const float vi = (float)(n - 1) * (q / 100.f), pf = floorf(vi); size_t p, nx;
    if (vi >= (float)(n - 1)) p = nx = n - 1; else if (vi < 0) p = nx = 0; else { p = (size_t)pf; nx = p + 1; }
    const float g = vi - (vi >= (float)(n - 1) ? -1.f : pf), a = v[p], b = v[nx], d = b - a;
    return g >= 0.5f ? b - d * (1.f - g) : a + d * g;
}

/* ---- phase_score.py ---- */
long long phase_score_surface(const uint8_t *pc, const uint8_t *ps, int w, int h, const uint8_t *seam, int sw, int sh, int r0, int c0,
                              phs_acc *acc, phs_acc cls[7]) {
    phs_acc A = {0, 0, 0}, C[7] = {0};
    const int kh = seam && r0 < sh ? ((sh - r0 + 1) / 2 < h ? (sh - r0 + 1) / 2 : h) : 0;   /* S[r0:r0 + 2 h:2, c0:c0 + 2 w:2] */
    const int kw = seam && c0 < sw ? ((sw - c0 + 1) / 2 < w ? (sw - c0 + 1) / 2 : w) : 0;
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++) {
            const size_t i = (size_t)r * w + c; const int a = pc[i] - 128, b = ps[i] - 128;
            if (pc[i] == 0 || a * a + b * b <= 169) continue;   /* not rendered, or |(c, s)| <= 13 */
            const double m = hypotf((float)a, (float)b);
            A.num += a; A.den += m; A.n++;
            if (r < kh && c < kw) { const int k = seam[(size_t)(r0 + 2 * r) * sw + c0 + 2 * c]; if (k >= 1 && k <= 6) { C[k].num += a; C[k].den += m; C[k].n++; } }
        }
    *acc = A;
    if (cls) for (int k = 0; k < 7; k++) cls[k] = k >= 1 && C[k].n >= 200 ? C[k] : (phs_acc){0, 0, 0};
    return A.n;
}
static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }
/* name ~ w(.+)_r(\d+)_c(\d+)$: the sheet name (into sheet), r0, c0; 0 on a match */
static int phs_render_name(const char *nm, char *sheet, size_t cap, int *r0, int *c0) {
    const size_t L = strlen(nm); if (L < 2 || nm[0] != 'w') return -1;
    size_t e = L; while (e > 0 && nm[e - 1] >= '0' && nm[e - 1] <= '9') e--;
    if (e == L || e < 2 || nm[e - 1] != 'c' || nm[e - 2] != '_') return -1;
    const size_t ce = e - 2; size_t s = ce; while (s > 0 && nm[s - 1] >= '0' && nm[s - 1] <= '9') s--;
    if (s == ce || s < 3 || nm[s - 1] != 'r' || nm[s - 2] != '_' || s - 2 <= 1 || s - 2 - 1 >= cap) return -1;
    memcpy(sheet, nm + 1, s - 3); sheet[s - 3] = 0; *r0 = atoi(nm + s); *c0 = atoi(nm + e); return 0;
}
int phase_score_dirs(const char *pcd, const char *psd, const char *sd, FILE *out) {
    static const char *names[7] = {"", "one box", "seam ok", "spread", "conflict", "bridge", "low q"};
    DIR *dd = opendir(pcd); if (!dd) return -1;
    char **fl = nullptr; int nf = 0, cap = 0; struct dirent *de;
    while ((de = readdir(dd))) {
        const size_t L = strlen(de->d_name); if (L < 4 || strcmp(de->d_name + L - 4, ".tif")) continue;
        if (nf == cap) { cap = cap ? 2 * cap : 256; fl = realloc(fl, cap * sizeof *fl); }
        fl[nf++] = strdup(de->d_name);
    }
    closedir(dd); qsort(fl, nf, sizeof *fl, cmp_str);
    double tn = 0, td = 0, *rows = malloc((nf ? nf : 1) * sizeof *rows); int nr = 0; phs_acc all[7] = {0};
    for (int f = 0; f < nf; f++) {
        char p[2048], q[2048]; snprintf(p, sizeof p, "%s/%s", pcd, fl[f]); snprintf(q, sizeof q, "%s/%s", psd, fl[f]);
        struct stat st; if (stat(q, &st)) continue;
        int w, h, w2, h2; uint8_t *pc = img_read_u8(p, &w, &h), *ps = img_read_u8(q, &w2, &h2);
        if (!pc || !ps || w != w2 || h != h2) { fprintf(stderr, "phase-score: cannot read %s / %s\n", p, q); free(pc); free(ps); continue; }
        char stem[1024]; snprintf(stem, sizeof stem, "%.*s", (int)strlen(fl[f]) - 4, fl[f]);
        char sheet[1024]; int r0 = 0, c0 = 0, sw = 0, sh = 0; uint8_t *seam = nullptr;
        if (sd && !phs_render_name(stem, sheet, sizeof sheet, &r0, &c0)) {
            snprintf(p, sizeof p, "%s/%s/seam.tif", sd, sheet);
            if (!stat(p, &st) && !(seam = img_read_u8(p, &sw, &sh))) fprintf(stderr, "phase-score: cannot read %s\n", p);
        }
        phs_acc A, C[7]; const long long n = phase_score_surface(pc, ps, w, h, seam, sw, sh, r0, c0, &A, C);
        free(pc); free(ps);
        if (n < 1000) { free(seam); continue; }
        tn += A.num; td += A.den;
        fprintf(out, "%s: on-recto %+.3f (%lld px)", stem, A.num / A.den, n);
        if (seam) {
            fprintf(out, " | "); int np = 0;
            for (int k = 1; k <= 6; k++) {
                if (!C[k].n) continue;
                all[k].num += C[k].num; all[k].den += C[k].den; all[k].n += C[k].n;
                fprintf(out, "%s%s %+.3f", np++ ? ", " : "", names[k], C[k].num / C[k].den);
            }
        }
        fputc('\n', out); fflush(out); free(seam);
        rows[nr++] = A.num / A.den;
    }
    if (nr) {
        double *t = malloc(nr * sizeof *t); memcpy(t, rows, nr * sizeof *t); qsort(t, nr, sizeof *t, cmp_f64);
        const double med = nr % 2 ? t[nr / 2] : (t[nr / 2 - 1] + t[nr / 2]) / 2;
        memcpy(t, rows, nr * sizeof *t); const double p10 = np_percentile_f64(t, nr, 10);
        memcpy(t, rows, nr * sizeof *t); const double p90 = np_percentile_f64(t, nr, 90); free(t);
        fprintf(out, "all: on-recto %+.3f over %d surfaces (median per surface %+.3f, p10 %+.3f, p90 %+.3f)\n", tn / td, nr, med, p10, p90);
    }
    for (int k = 1; k <= 6; k++) if (all[k].n) fprintf(out, "  %s: %+.3f (%lld px)\n", names[k], all[k].num / all[k].den, all[k].n);
    for (int f = 0; f < nf; f++) free(fl[f]);
    free(fl); free(rows);
    return 0;
}

/* ---- drawing ---- */
static const uint8_t font5x7[][8] = {   /* character, then 7 rows of 5 bits (bit 4 = left column) */
    {'0', 0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}, {'1', 0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}, {'2', 0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},
    {'3', 0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E}, {'4', 0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}, {'5', 0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E},
    {'6', 0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}, {'7', 0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}, {'8', 0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},
    {'9', 0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}, {'a', 0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F}, {'b', 0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x1E},
    {'c', 0x00, 0x00, 0x0E, 0x10, 0x10, 0x11, 0x0E}, {'d', 0x01, 0x01, 0x0D, 0x13, 0x11, 0x11, 0x0F}, {'e', 0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E},
    {'f', 0x06, 0x09, 0x08, 0x1C, 0x08, 0x08, 0x08}, {'g', 0x00, 0x0F, 0x11, 0x11, 0x0F, 0x01, 0x0E}, {'h', 0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x11},
    {'i', 0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E}, {'j', 0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C}, {'k', 0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12},
    {'l', 0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E}, {'m', 0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11}, {'n', 0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11},
    {'o', 0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E}, {'p', 0x00, 0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10}, {'q', 0x00, 0x00, 0x0D, 0x13, 0x0F, 0x01, 0x01},
    {'r', 0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10}, {'s', 0x00, 0x00, 0x0E, 0x10, 0x0E, 0x01, 0x1E}, {'t', 0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06},
    {'u', 0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D}, {'v', 0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04}, {'w', 0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A},
    {'x', 0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11}, {'y', 0x00, 0x00, 0x11, 0x11, 0x0F, 0x01, 0x0E}, {'z', 0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F},
    {'(', 0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02}, {')', 0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08}, {',', 0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x08},
    {'.', 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C}, {'-', 0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00}, {':', 0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00},
};
static void put_px(uint8_t *rgb, int w, int h, int x, int y, const uint8_t col[3]) {
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    uint8_t *p = rgb + ((size_t)y * w + x) * 3; p[0] = col[0]; p[1] = col[1]; p[2] = col[2];
}
void img_text(uint8_t *rgb, int w, int h, int x, int y, const char *s, const uint8_t col[3]) {
    for (; *s; s++, x += 6) {
        const uint8_t *g = nullptr;
        for (size_t k = 0; k < sizeof font5x7 / sizeof font5x7[0]; k++) if (font5x7[k][0] == (uint8_t)*s) g = font5x7[k] + 1;
        if (g) for (int r = 0; r < 7; r++) for (int c = 0; c < 5; c++) if (g[r] >> (4 - c) & 1) put_px(rgb, w, h, x + c, y + r, col);
    }
}
static int pil_round_up(double f) { return (int)(f >= 0.0 ? floor(f + 0.5F) : -floor(fabs(f) + 0.5F)); }
void img_line2(uint8_t *rgb, int w, int h, int x0, int y0, int x1, int y1, const uint8_t col[3]) {
    const int dx = x1 - x0, dy = y1 - y0;
    if (!dx && !dy) { put_px(rgb, w, h, x0, y0, col); return; }
    const double L = hypot(dx, dy); const int mx = pil_round_up(dy / L), my = pil_round_up(dx / L);   /* width 2: ratio_max 1 / L, ratio_min 0 */
    const double V[4][2] = {{x0, y0}, {x1, y1}, {x1 + mx, y1 - my}, {x0 + mx, y0 - my}};
    int xl = x0, xh = x0, yl = y0, yh = y0;
    for (int k = 0; k < 4; k++) { const int vx = (int)V[k][0], vy = (int)V[k][1]; xl = vx < xl ? vx : xl; xh = vx > xh ? vx : xh; yl = vy < yl ? vy : yl; yh = vy > yh ? vy : yh; }
    for (int y = yl; y <= yh; y++)
        for (int x = xl; x <= xh; x++) {   /* inside the convex quad, edges included */
            int pos = 0, neg = 0;
            for (int k = 0; k < 4; k++) {
                const double *a = V[k], *b = V[(k + 1) % 4], cr = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
                pos += cr > 1e-9; neg += cr < -1e-9;
            }
            if (!pos || !neg) put_px(rgb, w, h, x, y, col);
        }
}

/* ---- ufsm ink-compare: side-by-side panels and statistics of surface renders ---- */
void img_text_scaled(uint8_t *rgb, int w, int h, int x, int y, const char *s, int sc, const uint8_t col[3]) {
    if (sc < 1) sc = 1;
    for (; *s; s++, x += 6 * sc) {
        const uint8_t *g = nullptr;
        for (size_t k = 0; k < sizeof font5x7 / sizeof font5x7[0]; k++) if (font5x7[k][0] == (uint8_t)*s) g = font5x7[k] + 1;
        if (g) for (int r = 0; r < 7; r++) for (int c = 0; c < 5; c++) if (g[r] >> (4 - c) & 1)
            for (int a = 0; a < sc; a++) for (int b = 0; b < sc; b++) put_px(rgb, w, h, x + c * sc + b, y + r * sc + a, col);
    }
}
/* masked box mean of radius r (window (2r + 1)^2 clipped to the image, over mask != 0 pixels only) via integral images; out
   is the mean where the window holds a masked pixel, 0 elsewhere */
static int masked_box_mean(const uint8_t *img, const uint8_t *mask, int w, int h, int r, float *out) {
    const size_t W1 = (size_t)w + 1; double *S = calloc(W1 * (h + 1), sizeof *S), *C = calloc(W1 * (h + 1), sizeof *C);
    if (!S || !C) { free(S); free(C); return -1; }
    for (int y = 0; y < h; y++) {
        double rs = 0, rc = 0;
        for (int x = 0; x < w; x++) {
            const size_t i = (size_t)y * w + x; const int m = mask[i] != 0;
            rs += m ? img[i] : 0; rc += m;
            S[(y + 1) * W1 + x + 1] = S[y * W1 + x + 1] + rs; C[(y + 1) * W1 + x + 1] = C[y * W1 + x + 1] + rc;
        }
    }
    for (int y = 0; y < h; y++) {
        const int y0 = y - r < 0 ? 0 : y - r, y1 = y + r + 1 > h ? h : y + r + 1;
        for (int x = 0; x < w; x++) {
            const int x0 = x - r < 0 ? 0 : x - r, x1 = x + r + 1 > w ? w : x + r + 1;
            const double s = S[y1 * W1 + x1] - S[y0 * W1 + x1] - S[y1 * W1 + x0] + S[y0 * W1 + x0];
            const double c = C[y1 * W1 + x1] - C[y0 * W1 + x1] - C[y1 * W1 + x0] + C[y0 * W1 + x0];
            out[(size_t)y * w + x] = c > 0 ? (float)(s / c) : 0.f;
        }
    }
    free(S); free(C);
    return 0;
}
int ink_stats(const uint8_t *ink, const uint8_t *mask, int w, int h, int rs, int rl, ink_stats_t *st) {
    memset(st, 0, sizeof *st);
    const size_t n = (size_t)w * h; long long hist[256] = {0}; double sum = 0;
    for (size_t i = 0; i < n; i++) if (mask[i]) { hist[ink[i]]++; sum += ink[i]; st->n++; }
    if (!st->n) return 0;
    st->mean = sum / st->n / 255.0;
    long long hi = 0; for (int v = 128; v < 256; v++) hi += hist[v];
    st->frac50 = (double)hi / st->n;
    const double q[3] = {0.50, 0.95, 0.99}; double *qo[3] = {&st->p50, &st->p95, &st->p99};
    for (int k = 0; k < 3; k++) {   /* the lowest value whose cumulative count reaches q n */
        long long c = 0; const double t = q[k] * st->n; int v = 0;
        for (; v < 255; v++) { c += hist[v]; if (c >= t) break; }
        *qo[k] = v / 255.0;
    }
    float *bs = malloc(n * sizeof *bs), *bl = malloc(n * sizeof *bl);
    if (!bs || !bl || masked_box_mean(ink, mask, w, h, rs, bs) || masked_box_mean(ink, mask, w, h, rl, bl)) { free(bs); free(bl); return -1; }
    double m1 = 0, m2 = 0;
    for (size_t i = 0; i < n; i++) if (mask[i]) { const double d = (bs[i] - bl[i]) / 255.0; m1 += d; m2 += d * d; }
    m1 /= st->n; m2 /= st->n; st->band = sqrt(m2 - m1 * m1 > 0 ? m2 - m1 * m1 : 0);
    free(bs); free(bl);
    return 0;
}
double ink_corr(const uint8_t *a, const uint8_t *b, const uint8_t *mask, size_t n) {
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0; long long k = 0;
    for (size_t i = 0; i < n; i++) if (mask[i]) { const double x = a[i], y = b[i]; sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y; k++; }
    if (k < 2) return 0;
    const double va = saa - sa * sa / k, vb = sbb - sb * sb / k;
    return va > 0 && vb > 0 ? (sab - sa * sb / k) / sqrt(va * vb) : 0;
}
void img_stretch(uint8_t *im, const uint8_t *mask, size_t n, double plo, double phi) {
    long long hist[256] = {0}, cnt = 0;
    for (size_t i = 0; i < n; i++) if (mask[i]) { hist[im[i]]++; cnt++; }
    if (!cnt) return;
    int lo = 0, hi = 255; long long c = 0;
    for (lo = 0; lo < 255; lo++) { c += hist[lo]; if (c >= plo * cnt) break; }
    c = 0; for (hi = 0; hi < 255; hi++) { c += hist[hi]; if (c >= phi * cnt) break; }
    if (hi <= lo) hi = lo + 1;
    uint8_t lut[256]; for (int v = 0; v < 256; v++) { const double g = (v - lo) * 255.0 / (hi - lo); lut[v] = (uint8_t)(g < 0 ? 0 : g > 255 ? 255 : lround(g)); }
    for (size_t i = 0; i < n; i++) im[i] = mask[i] ? lut[im[i]] : 0;
}
int img_best_window(const uint8_t *im, const uint8_t *mask, int w, int h, int win, int *x0, int *y0) {
    if (win > w) win = w;
    if (win > h) win = h;
    if (win < 1) return -1;
    const size_t W1 = (size_t)w + 1; double *S = calloc(W1 * (h + 1), sizeof *S); long long *M = calloc(W1 * (h + 1), sizeof *M);
    if (!S || !M) { free(S); free(M); return -1; }
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {   /* integral images of the masked ink and of the mask */
        const size_t i = (size_t)y * w + x, o = (size_t)(y + 1) * W1 + x + 1;
        S[o] = (mask[i] ? im[i] : 0) + S[o - 1] + S[o - W1] - S[o - W1 - 1]; M[o] = (mask[i] != 0) + M[o - 1] + M[o - W1] - M[o - W1 - 1];
    }
    const int st = win / 4 > 0 ? win / 4 : 1; double best = -1; *x0 = *y0 = 0;
    for (int y = 0; y + win <= h; y += st) for (int x = 0; x + win <= w; x += st) {
        const size_t a = (size_t)y * W1 + x, b = a + win, c = a + (size_t)win * W1, d = c + win;
        const long long m = M[d] - M[b] - M[c] + M[a]; if (m * 2 < (long long)win * win) continue;   /* at least half on the surface */
        const double s = S[d] - S[b] - S[c] + S[a]; if (s > best) { best = s; *x0 = x; *y0 = y; }
    }
    free(S); free(M);
    return best < 0 ? -1 : 0;
}
uint8_t *ink_panels_half(const uint8_t *const *img, const int *is_ct, int nimg, int w, int h, const char *const *labels, int *ow, int *oh, int nthreads) {
    return ink_panels(img, is_ct, nimg, w, h, labels, 1, ow, oh, nthreads);
}
uint8_t *ink_panels(const uint8_t *const *img, const int *is_ct, int nimg, int w, int h, const char *const *labels, int half_res, int *ow, int *oh, int nthreads) {
    const int TW = nimg * w + 8 * (nimg - 1);
    uint8_t *full = malloc((size_t)TW * h * 3);
    if (!full || nimg < 1) { free(full); return nullptr; }
    memset(full, 255, (size_t)TW * h * 3);
    for (int k = 0; k < nimg; k++) for (int y = 0; y < h; y++) {
        const uint8_t *s = img[k] + (size_t)y * w; uint8_t *o = full + ((size_t)y * TW + (size_t)k * (w + 8)) * 3;
        for (int x = 0; x < w; x++) {
            float g = is_ct[k] ? ((float)s[x] - 40.f) * 2.2f : (float)s[x];
            g = g < 0.f ? 0.f : g > 255.f ? 255.f : g;
            o[3 * x] = o[3 * x + 1] = o[3 * x + 2] = (uint8_t)g;
        }
    }
    const int sc = h / 200 > 1 ? h / 200 : 2; const uint8_t yel[3] = {255, 220, 0};
    for (int k = 0; k < nimg; k++) if (labels && labels[k]) img_text_scaled(full, TW, h, k * (w + 8) + 2 * sc, 2 * sc, labels[k], sc, yel);
    if (!half_res) { *ow = TW; *oh = h; return full; }
    *ow = TW / 2; *oh = h / 2;
    uint8_t *half = *ow > 0 && *oh > 0 ? malloc((size_t)*ow * *oh * 3) : nullptr;
    if (!half || pil_resize(full, TW, h, 3, half, *ow, *oh, PIL_BOX, nthreads)) { free(half); half = nullptr; }
    free(full);
    return half;
}
/* the rectangle x, y, cw, ch of a w x h image as a new cw x ch image; pixels outside the source are 0; nullptr on bad sizes */
uint8_t *img_crop_u8(const uint8_t *im, int w, int h, int x, int y, int cw, int ch) {
    if (!im || w < 1 || h < 1 || cw < 1 || ch < 1) return nullptr;
    uint8_t *o = calloc((size_t)cw * ch, 1); if (!o) return nullptr;
    const int x0 = x < 0 ? 0 : x, x1 = x + cw < w ? x + cw : w;
    if (x1 > x0) for (int r = 0; r < ch; r++) { const int sy = y + r; if (sy < 0 || sy >= h) continue; memcpy(o + (size_t)r * cw + (x0 - x), im + (size_t)sy * w + x0, (size_t)(x1 - x0)); }
    return o;
}
