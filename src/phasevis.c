/* ufsm phase-vis OUT.jpg CTROOT CTKEY --plane z,y,x,h,w [--axis z|y|x] [--fields LABEL=DIR,...] [--codes ROOT KEY] [--zoom 2] [--max-px 8000]
   One CT plane (CT level voxels, e.g. Paris 4 level 2 = 9.6 um) and, side by side, the same plane with the integer-turn contours of
   each fit-field set (pc / ps; a set on the CT level grid at its fields.json origin / h; the phase bilinear as a phasor, a contour
   where it crosses an integer between neighbouring output pixels, red) and of the GT winding_mod14 codes (codes 1..252, 18 per turn:
   each GT turn its own colour over the CT, so the turn boundaries are colour edges). Panels: CT | each --fields | GT, 6 white
   pixels apart; JPEG q90, the longest side <= --max-px. */
#include "imgops.h"
#include "json.h"
#include "store.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *pv_opt(int argc, char **argv, const char *k, const char *d) { for (int i = argc - 2; i >= 1; i--) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }

int cmd_phase_vis(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm phase-vis OUT.jpg CTROOT CTKEY --plane z,y,x,h,w [--axis z|y|x] [--fields LABEL=DIR,..] [--codes ROOT KEY] [--zoom 2] [--max-px 8000] [--cache DIR]\n"); return 2; }
    long long pz[5]; if (sscanf(pv_opt(argc, argv, "--plane", ""), "%lld,%lld,%lld,%lld,%lld", pz, pz + 1, pz + 2, pz + 3, pz + 4) != 5 || pz[3] < 1 || pz[4] < 1) { fprintf(stderr, "phase-vis: --plane z,y,x,h,w\n"); return 2; }
    const int H = (int)pz[3], W = (int)pz[4], Z = atoi(pv_opt(argc, argv, "--zoom", "2")) < 1 ? 1 : atoi(pv_opt(argc, argv, "--zoom", "2"));
    const char ax = pv_opt(argc, argv, "--axis", "z")[0];
    /* plane pixel (r, c) -> volume (z, y, x): z axis: (pz0, py + r, px + c); y: (pz + r, py0, px + c); x: (pz + r, py + c, px0) */
    const int64_t o[3] = {pz[0], pz[1], pz[2]}, n[3] = {ax == 'z' ? 1 : H, ax == 'z' ? H : ax == 'y' ? 1 : W, ax == 'x' ? 1 : W};
    const size_t n2 = (size_t)H * W;
    uint8_t *ct = calloc(n2, 1);
    z3 *cz = z3_open(store_open(argv[3]), argv[4], pv_opt(argc, argv, "--cache", nullptr)); if (!cz || z3_read(cz, o, n, ct, 8)) { fprintf(stderr, "phase-vis: CT: %s\n", z3_error()); return 1; }
    z3_close(cz);
    /* fields */
    char fl[4096]; snprintf(fl, sizeof fl, "%s", pv_opt(argc, argv, "--fields", "")); char *lab[8], *dir[8]; int nf = 0;
    for (char *s = fl, *tok; nf < 8 && (tok = strtok(s, ",")); s = nullptr) { char *e = strchr(tok, '='); if (!e) { fprintf(stderr, "phase-vis: --fields LABEL=DIR\n"); return 2; } *e = 0; lab[nf] = tok; dir[nf++] = e + 1; }
    int ci = 0; for (int i = 3; i + 2 < argc; i++) if (!strcmp(argv[i], "--codes")) ci = i;
    const int np = 1 + nf + (ci ? 1 : 0), gap = 6, PW = W * Z, OW = (PW + gap) * np - gap, OH = H * Z;
    uint8_t *img = malloc((size_t)OW * OH * 3); memset(img, 255, (size_t)OW * OH * 3);
    for (int p = 0; p < np; p++) for (int y = 0; y < OH; y++) for (int x = 0; x < PW; x++) {   /* the CT under every panel */
        const uint8_t g = ct[(size_t)(y / Z) * W + x / Z]; uint8_t *px = img + ((size_t)y * OW + (size_t)p * (PW + gap) + x) * 3; px[0] = px[1] = px[2] = g;
    }
    float *ph = malloc((size_t)OW * OH * 4);
    for (int f = 0; f < nf; f++) {
        char jp[1400]; snprintf(jp, sizeof jp, "%s/fields.json", dir[f]); FILE *jf = fopen(jp, "rb"); if (!jf) { fprintf(stderr, "phase-vis: cannot read %s\n", jp); return 1; }
        fseek(jf, 0, SEEK_END); long jl = ftell(jf); fseek(jf, 0, SEEK_SET); char *jt = malloc(jl + 1); if (fread(jt, 1, jl, jf) != (size_t)jl) jl = 0; jt[jl] = 0; fclose(jf);
        json *J = json_parse(jt, jl); free(jt); if (!J) { fprintf(stderr, "phase-vis: bad %s\n", jp); return 1; }
        const int h = (int)json_num(json_get(J, "h"), 2); int64_t fo[3]; for (int d = 0; d < 3; d++) fo[d] = (int64_t)json_num(json_at(json_get(J, "origin"), d), 0); json_free(J);
        /* the cell box covering the plane (+1 for bilinear) */
        int64_t c0[3], cn[3]; const int64_t hi[3] = {o[0] + n[0], o[1] + n[1], o[2] + n[2]};
        for (int d = 0; d < 3; d++) { c0[d] = (int64_t)floor((o[d] - fo[d] - 0.5 * h) / (double)h) ; if (c0[d] < 0) c0[d] = 0; cn[d] = (hi[d] - fo[d]) / h + 2 - c0[d]; if (cn[d] < 1) cn[d] = 1; }
        const size_t nc = (size_t)cn[0] * cn[1] * cn[2]; uint8_t *pc = calloc(nc, 1), *ps = calloc(nc, 1);
        store *st = store_open(dir[f]); z3 *a = z3_open(st, "pc", nullptr), *b = z3_open(st, "ps", nullptr);
        if (!a || !b || z3_read(a, c0, cn, pc, 8) || z3_read(b, c0, cn, ps, 8)) { fprintf(stderr, "phase-vis: %s: %s\n", dir[f], z3_error()); return 1; }
        z3_close(a); z3_close(b);
        for (int y = 0; y < OH; y++) for (int x = 0; x < PW; x++) {
            const double r = (y + 0.5) / Z - 0.5, c = (x + 0.5) / Z - 0.5;   /* plane pixel coords */
            double v[3] = {(double)o[0], (double)o[1], (double)o[2]};
            if (ax == 'z') { v[1] += r; v[2] += c; } else if (ax == 'y') { v[0] += r; v[2] += c; } else { v[0] += r; v[1] += c; }
            double g[3]; int64_t i0[3]; double t[3];
            for (int d = 0; d < 3; d++) { g[d] = (v[d] - fo[d] + 0.5) / h - 0.5 - c0[d]; i0[d] = (int64_t)floor(g[d]); t[d] = g[d] - i0[d]; if (i0[d] < 0) { i0[d] = 0; t[d] = 0; } if (i0[d] > cn[d] - 2) { i0[d] = cn[d] > 1 ? cn[d] - 2 : 0; t[d] = cn[d] > 1 ? 1 : 0; } }
            double sa = 0, sb = 0; int miss = 0;
            for (int k = 0; k < 8; k++) {
                const int dz = k & 1, dy = k >> 1 & 1, dx = k >> 2 & 1; int64_t zz = i0[0] + dz, yy = i0[1] + dy, xx = i0[2] + dx;
                if (zz >= cn[0]) zz = cn[0] - 1; if (yy >= cn[1]) yy = cn[1] - 1; if (xx >= cn[2]) xx = cn[2] - 1;
                const size_t q = ((size_t)zz * cn[1] + yy) * cn[2] + xx; if (!pc[q] && !ps[q]) { miss = 1; break; }
                const double w = (dz ? t[0] : 1 - t[0]) * (dy ? t[1] : 1 - t[1]) * (dx ? t[2] : 1 - t[2]); sa += w * (pc[q] - 128) / 127.0; sb += w * (ps[q] - 128) / 127.0;
            }
            ph[(size_t)y * OW + x] = miss || hypot(sa, sb) < 0.05 ? NAN : (float)(atan2(sb, sa) / (2 * M_PI));
        }
        const int p = 1 + f;
        for (int y = 0; y < OH; y++) for (int x = 0; x < PW; x++) {
            const float a0 = ph[(size_t)y * OW + x]; if (isnan(a0)) { uint8_t *px = img + ((size_t)y * OW + (size_t)p * (PW + gap) + x) * 3; px[2] = (uint8_t)fmin(255, px[2] / 2 + 70); continue; }
            int edge = 0;
            for (int k = 0; k < 2 && !edge; k++) { const int yy = y + (k == 1), xx = x + (k == 0); if (yy >= OH || xx >= PW) continue; const float a1 = ph[(size_t)yy * OW + xx]; if (isnan(a1)) continue;
                double d = a1 - a0; d -= round(d); const double u0 = a0 - floor(a0); edge = (u0 + d >= 1 || u0 + d < 0) && fabs(d) < 0.25; }
            if (edge) for (int e = 0; e < 4; e++) { const int yy = y + (e >> 1), xx = x + (e & 1); if (yy >= OH || xx >= PW) continue; uint8_t *px = img + ((size_t)yy * OW + (size_t)p * (PW + gap) + xx) * 3; px[0] = 255; px[1] = 40; px[2] = 30; }
        }
        free(pc); free(ps);
    }
    if (ci) {   /* GT codes on the CT grid */
        uint8_t *cd = calloc(n2, 1); z3 *qz = z3_open(store_open(argv[ci + 1]), argv[ci + 2], nullptr);
        if (!qz || z3_read(qz, o, n, cd, 8)) { fprintf(stderr, "phase-vis: codes: %s\n", z3_error()); return 1; }
        z3_close(qz);
        static const uint8_t pal[14][3] = {{230, 25, 75}, {60, 180, 75}, {255, 225, 25}, {0, 130, 200}, {245, 130, 48}, {145, 30, 180}, {70, 240, 240},
                                           {240, 50, 230}, {210, 245, 60}, {250, 190, 212}, {0, 128, 128}, {220, 190, 255}, {170, 110, 40}, {255, 250, 200}};
        const int p = 1 + nf;
        for (int y = 0; y < OH; y++) for (int x = 0; x < PW; x++) {
            const int c = cd[(size_t)(y / Z) * W + x / Z]; if (c < 1 || c > 252) continue; const int t = (c - 1) / 18 % 14;
            uint8_t *px = img + ((size_t)y * OW + (size_t)p * (PW + gap) + x) * 3;
            for (int e = 0; e < 3; e++) px[e] = (uint8_t)((2 * px[e] + 3 * pal[t][e]) / 5);
        }
        free(cd);
    }
    int ow = OW, oh = OH; if (img_fit_rgb(&img, &ow, &oh, atoi(pv_opt(argc, argv, "--max-px", "8000")), 8)) return 1;
    const int rc = jpeg_write_rgb(argv[2], ow, oh, img, 90);
    fprintf(stderr, "phase-vis: %s (%d x %d, %d panels)\n", argv[2], ow, oh, np);
    free(ct); free(img); free(ph);
    return rc;
}
