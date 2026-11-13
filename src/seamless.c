/* predict --seamless 2: seam-free whole-volume prediction as two passes on tile grids offset by half a tile, blended with sin^2 tile
   weights (2026-11-12, DESIGN "Latent ink head: window seams").

   Why: the latent nets' outputs carry a per-window level (inkpod-9000 ink: plane-mean steps of 10-30 / 255 at the window seams on a
   1299 corner, at halo 16 .. 256, with --gn-frozen, at level 1 and level 2, with and without the per-window dice loss): the stride-4
   trunk's context is wider than any affordable halo, so halos and --feather ramps (B <= halo) cannot hide it. Two passes, the second
   with its tile interiors anchored at S / 2 on every axis, and a per-voxel average with weights prod_d sin^2(pi u_d) (u the voxel's
   place in its tile, src/fblend.c fb_tile_w) give every tile border ~0 weight: the largest plane step on a 512^3 read across the corner
   16.9 -> 1.3 (median 0.1). Cost 2x the windows, + the blend (~20 s CPU per 1667 head box); --feather at the same quality needs a ramp
   of ~S / 2, i.e. windows ~2x as wide (8x the voxels).

   How: this process (no CUDA) runs the two passes as child processes of the same binary (/proc/self/exe, UFSM_SEAM_CHILD=1): the same
   arguments, the output <out>/.seam/{a,b}, --full-grid 1 --q 0 --levels 1 --window W --halo H --shard S / 2 and --grid-origin 0,0,0 /
   S/2,S/2,S/2 (shard S / 2: the shifted tiles stay aligned with the store's shards, one window per tile; with S-sized shards each
   shifted tile spans 8 shards and the planner runs 7.5x the windows, P4 inner 17405 vs 2311; outputs byte-identical). --gpus A,B runs
   the passes at once, one per GPU; with --split (one window over both GPUs) the passes keep --split / --gpus and run one after the other;
   otherwise both on --gpu, one after the other. Then every store of pass a (the single output, or each
   --heads store <out>/<name>) is blended with its pass-b twin (src/inkblend.c ink_blend_run) into <out>[/<name>] with the run's --q,
   --levels and --shard. A pass with <out>/.seam/<p>/.done is not rerun (resume after a blend failure); .seam is removed at the end
   (UFSM_SEAM_KEEP=1 keeps it). */
#include "inkblend.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <ftw.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static const char *sopt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static int rm_cb(const char *p, const struct stat *sb, int fl, struct FTW *f) { (void)sb; (void)fl; (void)f; return remove(p); }
static int rm_rf(const char *p) { struct stat sb; if (lstat(p, &sb)) return 0; return nftw(p, rm_cb, 64, FTW_DEPTH | FTW_PHYS); }
static int mkdir_p(const char *p) {
    char t[2048]; snprintf(t, sizeof t, "%s", p);
    for (char *s = t + 1; *s; s++) if (*s == '/') { *s = 0; if (mkdir(t, 0775) && errno != EEXIST) return -1; *s = '/'; }
    return mkdir(t, 0775) && errno != EEXIST ? -1 : 0;
}
static int is_file(const char *p) { struct stat sb; return !stat(p, &sb) && S_ISREG(sb.st_mode); }

/* the options a pass sets itself (each takes a value) */
static const char *const own[] = {"--seamless", "--gpus", "--gpu", "--q", "--levels", "--full-grid", "--grid-origin", "--shard", "--window",
                                  "--halo", "--resume", "--boxes", "--box", nullptr};
static int g_split;   /* --split z --gpus A,B: every window split over both GPUs; the passes keep --split / --gpus and run one after the other */
static int owned(const char *a) {
    if (g_split && (!strcmp(a, "--gpus") || !strcmp(a, "--gpu"))) return 0;
    for (int k = 0; own[k]; k++) if (!strcmp(a, own[k])) return 1;
    return 0;
}

/* the box of a pass (in --level voxels): its origin moved down onto the pass's tile grid (origin = shift mod S), else predict's planner
   places one window per shard instead of per tile (stride check against --grid-origin): 1667 box at shard 256, 3305 windows vs 578. A
   pass-b origin that would fall below 0 starts at shift instead (the voxels below shift come from pass a alone: ink_blend_run gives an
   absent shard weight 0) */
static long long g_lo[3], g_hi[3];
static char *pass_box(int S, int shift) {
    long long o[3], n[3];
    for (int d = 0; d < 3; d++) {
        long long lo = g_lo[d] - shift; lo = (lo >= 0 ? lo / S : -((-lo + S - 1) / S)) * S + shift; if (lo < 0) lo = shift;
        o[d] = lo; n[d] = g_hi[d] > lo ? g_hi[d] - lo : 1;
    }
    char b[160]; snprintf(b, sizeof b, "%lld,%lld,%lld,%lld,%lld,%lld", o[0], o[1], o[2], n[0], n[1], n[2]); return strdup(b);
}
static pid_t spawn_pass(int argc, char **argv, const char *pout, int W, int H, int S2, int shift, int gpu, const char *logp) {
    char **v = calloc((size_t)argc + 32, sizeof *v); int n = 0;
    static char sw[16], sh[16], ss[16], sg[48], sgpu[16];
    char *bw = strdup((snprintf(sw, sizeof sw, "%d", W), sw)), *bh = strdup((snprintf(sh, sizeof sh, "%d", H), sh)), *bs = strdup((snprintf(ss, sizeof ss, "%d", S2), ss));
    char *bg = strdup((snprintf(sg, sizeof sg, "%d,%d,%d", shift, shift, shift), sg)), *bu = strdup((snprintf(sgpu, sizeof sgpu, "%d", gpu), sgpu));
    for (int i = 0; i < 6 && i < argc; i++) v[n++] = i == 5 ? (char *)pout : argv[i];
    for (int i = 6; i < argc; i++) { if (owned(argv[i]) && i + 1 < argc) { i++; continue; } v[n++] = argv[i]; }
    char *bb = pass_box(2 * S2, shift);
    const char *add[] = {"--box", bb, "--window", bw, "--halo", bh, "--shard", bs, "--grid-origin", bg, "--full-grid", "1", "--q", "0", "--levels", "1", "--gpu", bu};
    for (size_t k = 0; k < sizeof add / sizeof *add - (g_split ? 2 : 0); k++) v[n++] = (char *)add[k];
    v[n] = nullptr;
    posix_spawn_file_actions_t fa; posix_spawn_file_actions_init(&fa);
    if (logp) { posix_spawn_file_actions_addopen(&fa, 2, logp, O_WRONLY | O_CREAT | O_TRUNC, 0644); posix_spawn_file_actions_adddup2(&fa, 2, 1); }
    const char *exe = getenv("UFSM_SEAM_EXE");   /* (tests: a stand-in for the pass, tests/test_seamless.py) */
    pid_t pid = -1; const int e = posix_spawn(&pid, exe && *exe ? exe : "/proc/self/exe", &fa, nullptr, v, environ);
    posix_spawn_file_actions_destroy(&fa); free(v);
    if (e) { fprintf(stderr, "predict --seamless: cannot start a pass: %s\n", strerror(e)); return -1; }
    return pid;
}
static void log_tail(const char *logp) {   /* a failed pass's last lines (its log may be removed with the output later) */
    FILE *f = logp ? fopen(logp, "rb") : nullptr; if (!f) return;
    fseek(f, 0, SEEK_END); long n = ftell(f), k = n > 2000 ? 2000 : n; fseek(f, n - k, SEEK_SET);
    char b[2001]; size_t r = fread(b, 1, (size_t)k, f); b[r] = 0; fclose(f);
    for (char *c = b; *c; c++) if (*c == '\r') *c = '\n';
    fprintf(stderr, "---- %s (tail)\n%s\n----\n", logp, b);
}
static int wait_pass(pid_t pid, const char *name, const char *logp) {
    int st = 0; if (waitpid(pid, &st, 0) < 0) { fprintf(stderr, "predict --seamless: wait pass %s: %s\n", name, strerror(errno)); return 1; }
    if (WIFEXITED(st) && !WEXITSTATUS(st)) return 0;
    log_tail(logp);
    if (WIFSIGNALED(st)) fprintf(stderr, "predict --seamless: pass %s killed by signal %d\n", name, WTERMSIG(st));
    else fprintf(stderr, "predict --seamless: pass %s failed (exit %d)\n", name, WEXITSTATUS(st));
    return WIFEXITED(st) && WEXITSTATUS(st) ? WEXITSTATUS(st) : 1;
}

/* the stores of a pass: directories holding a level array <dir>/<um>/zarr.json (um numeric; --levels 1: one level); rel paths */
typedef struct { char rel[256]; char lv[32]; } seam_store;
static int find_stores(const char *base, const char *rel, int depth, seam_store *o, int no, int max) {
    char d[2048]; snprintf(d, sizeof d, "%s%s%s", base, *rel ? "/" : "", rel);
    DIR *dp = opendir(d); if (!dp) return no; struct dirent *e;
    while ((e = readdir(dp))) {
        if (e->d_name[0] == '.') continue;
        char p[2400], z[2500]; snprintf(p, sizeof p, "%s/%s", d, e->d_name); snprintf(z, sizeof z, "%s/zarr.json", p);
        char *end; const double u = strtod(e->d_name, &end);
        if (*end == 0 && u > 0 && is_file(z)) { if (no < max) { snprintf(o[no].rel, sizeof o[no].rel, "%s", rel); snprintf(o[no].lv, sizeof o[no].lv, "%s", e->d_name); no++; } continue; }
        struct stat sb; if (depth < 4 && !stat(p, &sb) && S_ISDIR(sb.st_mode)) {
            char r[256]; snprintf(r, sizeof r, "%s%s%s", rel, *rel ? "/" : "", e->d_name); no = find_stores(base, r, depth + 1, o, no, max);
        }
    }
    closedir(dp); return no;
}

int predict_one_entry(int argc, char **argv);   /* src/predict.c */
/* predict --seam-inproc: a store's level arrays <dir>/<um> (numeric names with a zarr.json) removed before it is written, so the run
   replaces the store as the two-process blend does (and the GPU pyramid, which writes each level shard once, sees no older shards) */
void seam_clear_store(const char *dir) {
    DIR *dp = opendir(dir); if (!dp) return; struct dirent *e; char p[64][2400]; int n = 0;
    while ((e = readdir(dp)) && n < 64) {
        char *end; const double u = strtod(e->d_name, &end); char z[2500];
        snprintf(p[n], sizeof p[n], "%s/%s", dir, e->d_name); snprintf(z, sizeof z, "%s/zarr.json", p[n]);
        if (e->d_name[0] != '.' && *end == 0 && u > 0 && is_file(z)) n++;
    }
    closedir(dp);
    for (int i = 0; i < n; i++) rm_rf(p[i]);
}
/* in one process (UFSM_SEAM_INPROC=1): predict --seam-inproc 1 over the union of the two passes' boxes, shards S / 2, the run's --q,
   --levels; both tile grids in one window loop, each shard's two versions blended on the GPU before the writer (src/predict.c seam_t):
   no pass stores, no second start-up, no CPU blend */
static int g_fitz;   /* --fit-zarr given (in-process only) */
static int seamless_inproc(int argc, char **argv, int W, int H, int S2, int gpu) {
    g_fitz = *sopt(argc, argv, "--fit-zarr", "") != 0;
    char **v = calloc((size_t)argc + 32, sizeof *v); int n = 0;
    char sb[160], sw[16], sh[16], ss[16], sg[16]; long long o[3], e[3];
    for (int d = 0; d < 3; d++) {
        long long la = g_lo[d] - 0, lb = g_lo[d] - S2, S = 2LL * S2;
        la = (la >= 0 ? la / S : -((-la + S - 1) / S)) * S; if (la < 0) la = 0;
        lb = (lb >= 0 ? lb / S : -((-lb + S - 1) / S)) * S + S2; if (lb < 0) lb = S2;
        o[d] = g_fitz || la < lb ? la : lb;   /* (--fit-zarr: the box on grid 0, whose windows alone feed the fields: field slabs on its tiles) */
        e[d] = g_hi[d] > o[d] ? g_hi[d] - o[d] : 1;
    }
    snprintf(sb, sizeof sb, "%lld,%lld,%lld,%lld,%lld,%lld", o[0], o[1], o[2], e[0], e[1], e[2]);
    snprintf(sw, sizeof sw, "%d", W); snprintf(sh, sizeof sh, "%d", H); snprintf(ss, sizeof ss, "%d", S2); snprintf(sg, sizeof sg, "%d", gpu);
    for (int i = 0; i < 6 && i < argc; i++) v[n++] = argv[i];
    for (int i = 6; i < argc; i++) {
        if (!strcmp(argv[i], "--q") || !strcmp(argv[i], "--levels")) { v[n++] = argv[i]; if (i + 1 < argc) v[n++] = argv[++i]; continue; }
        if (owned(argv[i]) && i + 1 < argc) { i++; continue; }
        v[n++] = argv[i];
    }
    const char *add[] = {"--box", sb, "--window", sw, "--halo", sh, "--shard", ss, "--full-grid", "1", "--seam-inproc", "1", "--gpu", sg};
    for (size_t k = 0; k < sizeof add / sizeof *add - (g_split ? 2 : 0); k++) v[n++] = (char *)add[k];
    v[n] = nullptr;
    setenv("UFSM_SEAM_CHILD", "1", 1);
    fprintf(stderr, "predict --seamless 2 (in-process): window %d halo %d, tile %d, grids 0 and %d, shards %d\n", W, H, 2 * S2, S2, S2);
    const int rc = predict_one_entry(n, v);
    free(v); return rc;
}

int predict_seamless(int argc, char **argv) {
    const int mode = atoi(sopt(argc, argv, "--seamless", "0"));
    if (mode != 2) { fprintf(stderr, "predict --seamless: 2 (two passes on half-tile-shifted grids, sin^2 blend) is the one mode\n"); return 2; }
    if (argc < 6) { fprintf(stderr, "predict --seamless: needs the positional arguments (ufsm predict CKPT ROOT KEY OUT ...)\n"); return 2; }
    const int inproc = getenv("UFSM_SEAM_INPROC") && atoi(getenv("UFSM_SEAM_INPROC"));
    if (*sopt(argc, argv, "--boxes", "") || (*sopt(argc, argv, "--fit-zarr", "") && !inproc) || *sopt(argc, argv, "--feather", "")) {
        fprintf(stderr, "predict --seamless: not with --boxes or --feather; --fit-zarr only in-process (UFSM_SEAM_INPROC=1)\n"); return 2; }
    g_split = *sopt(argc, argv, "--split", "") != 0;
    { const char *b = sopt(argc, argv, "--box", nullptr); long long v[6];
      if (b && sscanf(b, "%lld,%lld,%lld,%lld,%lld,%lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 6) for (int d = 0; d < 3; d++) { g_lo[d] = v[d]; g_hi[d] = v[d] + v[3 + d]; }
      else if (b) { fprintf(stderr, "predict --seamless: --box z,y,x,nz,ny,nx\n"); return 2; }
      else for (int d = 0; d < 3; d++) { g_lo[d] = 0; g_hi[d] = 1LL << 40; } }   /* (predict clips the box to the CT level) */
    const char *ws = sopt(argc, argv, "--window", ""), *hs = sopt(argc, argv, "--halo", "");
    const int W = *ws && strcmp(ws, "auto") ? atoi(ws) : 544, H = *hs ? atoi(hs) : 16, S = W - 2 * H, S2 = S / 2;
    if (H < 0 || S <= 0 || S % 256) { fprintf(stderr, "predict --seamless: window - 2 halo must be a positive multiple of 256 (window %d, halo %d)\n", W, H); return 2; }
    int ga = atoi(sopt(argc, argv, "--gpu", "0")), gb = ga, par = 0;
    const char *gl = sopt(argc, argv, "--gpus", nullptr);
    if (gl && !g_split) { if (sscanf(gl, "%d,%d", &ga, &gb) == 2 && ga != gb) par = 1; else if (sscanf(gl, "%d", &ga) == 1) gb = ga; }
    if (inproc) return seamless_inproc(argc, argv, W, H, S2, ga);
    const char *out = argv[5]; const double q = atof(sopt(argc, argv, "--q", "8")); const int nlev = atoi(sopt(argc, argv, "--levels", "4"));
    const int shard = atoi(sopt(argc, argv, "--shard", "512")), nt = atoi(sopt(argc, argv, "--threads", "8"));
    if (shard < 128 || shard % 128 || nlev < 1 || q < 0) { fprintf(stderr, "predict --seamless: --shard a multiple of 128, --levels >= 1, --q >= 0\n"); return 2; }
    char tmp[1400], pa[1500], pb[1500], la[1600], lb[1600], da[1600], db[1600];
    snprintf(tmp, sizeof tmp, "%s/.seam", out); snprintf(pa, sizeof pa, "%s/a", tmp); snprintf(pb, sizeof pb, "%s/b", tmp);
    snprintf(la, sizeof la, "%s/a.log", tmp); snprintf(lb, sizeof lb, "%s/b.log", tmp); snprintf(da, sizeof da, "%s/.done", pa); snprintf(db, sizeof db, "%s/.done", pb);
    if (mkdir_p(tmp)) { fprintf(stderr, "predict --seamless: cannot create %s: %s\n", tmp, strerror(errno)); return 1; }
    setenv("UFSM_SEAM_CHILD", "1", 1);
    fprintf(stderr, "predict --seamless 2: window %d halo %d, tile %d, passes on grids 0 and %d (shards %d), %s; logs %s/{a,b}.log\n", W, H, S, S2, S2,
            par ? "one per GPU at once" : g_split ? "one after the other, each window split (--split, --gpus kept)" : "one after the other", tmp);
    const int need_a = !is_file(da), need_b = !is_file(db); int rc = 0;
    if (need_a) rm_rf(pa);
    if (need_b) rm_rf(pb);
    if (par) {
        const pid_t a = need_a ? spawn_pass(argc, argv, pa, W, H, S2, 0, ga, la) : 0, b = need_b ? spawn_pass(argc, argv, pb, W, H, S2, S2, gb, lb) : 0;
        if (a < 0 || b < 0) return 1;
        const int ra = a ? wait_pass(a, "a", la) : 0, rb = b ? wait_pass(b, "b", lb) : 0; rc = ra ? ra : rb;
        if (!ra && a) fclose(fopen(da, "w"));
        if (!rb && b) fclose(fopen(db, "w"));
    } else {
        if (need_a) { const pid_t a = spawn_pass(argc, argv, pa, W, H, S2, 0, ga, la); if (a < 0) return 1; if ((rc = wait_pass(a, "a", la))) goto fail; fclose(fopen(da, "w")); }
        if (need_b) { const pid_t b = spawn_pass(argc, argv, pb, W, H, S2, S2, gb, lb); if (b < 0) return 1; if ((rc = wait_pass(b, "b", lb))) goto fail; fclose(fopen(db, "w")); }
    }
    if (rc) goto fail;
    seam_store st[64]; const int ns = find_stores(pa, "", 0, st, 0, 64);
    if (!ns) { fprintf(stderr, "predict --seamless: pass a wrote no store under %s\n", pa); return 1; }
    for (int k = 0; k < ns; k++) {
        char ia[1800], ib[1800], o[1800], attrs[600]; const char *r = st[k].rel;
        snprintf(ia, sizeof ia, "%s%s%s", pa, *r ? "/" : "", r); snprintf(ib, sizeof ib, "%s%s%s", pb, *r ? "/" : "", r);
        snprintf(o, sizeof o, "%s%s%s", out, *r ? "/" : "", r);
        { char z[1900]; snprintf(z, sizeof z, "%s/%s/zarr.json", ib, st[k].lv); if (!is_file(z)) { fprintf(stderr, "predict --seamless: pass b has no %s\n", z); return 1; } }
        if (*r) rm_rf(o);
        else { char z[1900]; snprintf(z, sizeof z, "%s/%s", out, st[k].lv); rm_rf(z); }
        if (mkdir_p(o)) { fprintf(stderr, "predict --seamless: cannot create %s\n", o); return 1; }
        snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"%s probability (predict --seamless 2: two passes on grids 0 and %d, tile %d, sin^2 blend)\","
                 "\"seamless\":{\"mode\":2,\"window\":%d,\"halo\":%d,\"tile\":%d,\"grids\":[0,%d]}}}", *r ? r : "prediction", S2, S, W, H, S, S2);
        const double tl[3] = {S, S, S}, g[2][3] = {{0, 0, 0}, {S2, S2, S2}}; const int64_t org[2][3] = {{0, 0, 0}, {0, 0, 0}};
        const char *in[2] = {ia, ib};
        fprintf(stderr, "predict --seamless: blend %s (%s um)\n", *r ? r : "(the output)", st[k].lv);
        if (ink_blend_run(o, atof(st[k].lv), tl, 2, in, g, org, nlev, shard, q, nt, attrs)) { fprintf(stderr, "predict --seamless: blend of %s failed (passes kept in %s)\n", *r ? r : "the output", tmp); return 1; }
    }
    if (!getenv("UFSM_SEAM_KEEP") || !atoi(getenv("UFSM_SEAM_KEEP"))) rm_rf(tmp);
    fprintf(stderr, "predict --seamless: %d store%s written under %s\n", ns, ns == 1 ? "" : "s", out);
    return 0;
fail:
    fprintf(stderr, "predict --seamless: a pass failed; see %s/{a,b}.log (finished passes are kept: rerun resumes)\n", tmp);
    return rc ? rc : 1;
}
