/* CUDA ops: core section of the former nn.cu */
#include "nn_common.cuh"

int g_tf32 = 1;
int g_gn_stored = 0;
extern "C" void nn_set_gn_stored(int on) { g_gn_stored = on != 0; }
extern "C" int nn_get_gn_stored(void) { return g_gn_stored; }
int g_prec = 1, g_pref = 1;
extern "C" void nn_set_tf32(int on) { g_tf32 = on; g_prec = on ? g_pref : 0; }
extern "C" void nn_set_prec(int p) { g_prec = p; g_tf32 = p > 0; if (p > 0) g_pref = p; }   /* the fp8/fp4 kernels accept fp32, bf16 or fp16 storage */
extern "C" int nn_get_prec(void) { return g_prec; }
int g_layer = -1, g_lprec[NN_MAXLAYER];
int g_lprec_init = 0;
void lprec_init(void) { if (!g_lprec_init) { for (int i = 0; i < NN_MAXLAYER; i++) g_lprec[i] = -1; g_lprec_init = 1; } }
extern "C" void nn_set_layer(int id) { g_layer = id; }
extern "C" void nn_set_layer_prec(int id, int p) { lprec_init(); if (id >= 0 && id < NN_MAXLAYER) g_lprec[id] = p; }
int g_sub = -1, g_pass = 0;
unsigned g_exec_prec[NN_MAXLAYER][2][3];
void exec_prec(int pass, int p) {
    if (g_layer >= 0 && g_layer < NN_MAXLAYER)
        __atomic_fetch_or(&g_exec_prec[g_layer][g_sub > 0 ? 1 : 0][pass], 1u << p, __ATOMIC_RELAXED);
}
int g_sr = -1;
unsigned g_sr_step = 0, g_sr_ctr = 0;
int sr_on(void) { if (g_sr < 0) { const char *e = getenv("UFSM_SR"); g_sr = e ? atoi(e) : 0; } return g_sr; }
unsigned sr_seed(void) {
    unsigned h = g_sr_step * 0x9e3779b9u ^ (++g_sr_ctr) * 0x85ebca6bu;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return h | 1u;
}
extern "C" void nn_set_sr(int on) { g_sr = on; }
extern "C" void nn_set_sr_step(unsigned step) { g_sr_step = step; g_sr_ctr = 0; lp_wmemo_step(step); }
extern "C" void nn_wmemo_clear(void) { lp_wmemo_clear(); }
unsigned conv_wkey(void) { return g_layer >= 0 ? (unsigned)((g_layer * 2 + (g_sub > 0 ? g_sub : 0)) * 4 + g_pass + 1) : 0u; }
signed char g_lprec3[NN_MAXLAYER][2][3];
extern "C" void nn_set_conv(int sub) { g_sub = sub; }
extern "C" int nn_get_layer(void) { return g_layer; }
extern "C" int nn_get_conv(void) { return g_sub; }
extern "C" void nn_set_conv_prec(int id, int sub, int p_fwd, int p_bwd_data, int p_wgrad) {
    if (id < 0 || id >= NN_MAXLAYER) return;
    for (int s2 = 0; s2 < 2; s2++) if (sub < 0 || sub == s2) { g_lprec3[id][s2][0] = (signed char)(p_fwd > 0 ? p_fwd : 0); g_lprec3[id][s2][1] = (signed char)(p_bwd_data > 0 ? p_bwd_data : 0); g_lprec3[id][s2][2] = (signed char)(p_wgrad > 0 ? p_wgrad : 0); }
}
extern "C" int nn_get_conv_prec(int id, int sub, int pass) { return id >= 0 && id < NN_MAXLAYER && sub >= 0 && sub < 2 && pass >= 0 && pass < 3 ? g_lprec3[id][sub][pass] : 0; }
int g_prec_w = -1;
extern "C" void nn_set_prec_wgrad(int p) { g_prec_w = p; }
int eff_prec_pass(int pass) {
    lprec_init();
    if (!g_tf32) return 0;
    if (pass == 2 && g_prec_w >= 1) return g_prec_w;
    if (g_layer >= 0 && g_layer < NN_MAXLAYER) {
        int p3 = g_lprec3[g_layer][g_sub > 0 ? 1 : 0][pass];
        if (p3 >= 1) return p3;
        if (g_lprec[g_layer] >= 1) return g_lprec[g_layer];
    }
    return g_prec;
}
int eff_prec(void) { return eff_prec_pass(g_pass); }
int eff_prec_w(void) { return eff_prec_pass(2); }
extern "C" int nn_cur_prec(void) { return eff_prec(); }
extern "C" int nn_prec_parse(const char *s) {
    static const char *nm[] = {"fp32", "bf16", "fp8", "fp4", "fp16"};
    for (int i = 0; i < 5; i++) if (!strcmp(s, nm[i])) return i;
    char *e; long v = strtol(s, &e, 10);
    return *s && !*e && v >= 0 && v <= 4 ? (int)v : -1;
}
extern "C" const char *nn_prec_name(int p) { static const char *nm[] = {"fp32", "bf16", "fp8", "fp4", "fp16"}; return p >= 0 && p <= 4 ? nm[p] : "?"; }
extern "C" int nn_set_prec_policy(const char *pol) {
    static const char *names[] = {"enc0", "enc1", "enc2", "enc3", "down0", "down1", "down2", "dec2", "dec1", "dec0", "head"};
    lprec_init();
    for (int i = 0; i < NN_MAXLAYER; i++) { g_lprec[i] = -1; nn_set_conv_prec(i, -1, 0, 0, 0); }
    if (!pol || !*pol) return 0;
    char buf[2048]; snprintf(buf, sizeof buf, "%s", pol);
    int pos = 0;
    char *save = nullptr;
    for (char *tok = strtok_r(buf, ", ", &save); tok; tok = strtok_r(nullptr, ", ", &save)) {
        char *eq = strchr(tok, '=');
        if (!eq) { int v = nn_prec_parse(tok); if (v < 1 || pos >= 11) { fprintf(stderr, "nn_set_prec_policy: bad entry '%s'\n", tok); return -1; } g_lprec[pos++] = v; continue; }
        *eq = 0;
        char *val = eq + 1, *c1 = strchr(val, ':'), *c2 = c1 ? strchr(c1 + 1, ':') : nullptr;
        int pv[3];
        if (c1 && c2) { *c1 = *c2 = 0; pv[0] = nn_prec_parse(val); pv[1] = nn_prec_parse(c1 + 1); pv[2] = nn_prec_parse(c2 + 1); }
        else if (!c1) pv[0] = pv[1] = pv[2] = nn_prec_parse(val);
        else pv[0] = -1;
        if (pv[0] < 1 || pv[1] < 1 || pv[2] < 1) { fprintf(stderr, "nn_set_prec_policy: bad precision in '%s'\n", val); return -1; }
        int sub = -1;
        char *dot = strchr(tok, '.');
        if (dot) { if (!strcmp(dot, ".c1")) sub = 0; else if (!strcmp(dot, ".c2")) sub = 1; else { fprintf(stderr, "nn_set_prec_policy: bad conv '%s'\n", tok); return -1; } *dot = 0; }
        int lo = -1, hi = -1;
        if (!strcmp(tok, "all")) { lo = 0; hi = 10; }
        else for (int i = 0; i < 11; i++) if (!strcmp(tok, names[i])) lo = hi = i;
        if (lo < 0) { fprintf(stderr, "nn_set_prec_policy: unknown layer '%s'\n", tok); return -1; }
        for (int i = lo; i <= hi; i++) {
            if (sub < 0 && pv[0] == pv[1] && pv[1] == pv[2]) { g_lprec[i] = pv[0]; nn_set_conv_prec(i, -1, 0, 0, 0); }
            else nn_set_conv_prec(i, sub, pv[0], pv[1], pv[2]);
        }
    }
    return 0;
}
extern "C" int nn_get_tf32(void) { return g_tf32; }
int g_actbf = 1;
extern "C" void nn_set_act_bf16(int on) { g_actbf = on; }
extern "C" int nn_get_act_bf16(void) { return g_actbf; }
int g_h16 = 0;
extern "C" void nn_set_f16(int on) { g_h16 = on; }
const char *pname16(int p) { return p == 1 && g_h16 ? "fp16" : nn_prec_name(p); }   /* prec 1 is the 16-bit storage type */
extern "C" int nn_prec_manifest(char *buf, size_t n) {
    static const char *names[] = {"enc0", "enc1", "enc2", "enc3", "down0", "down1", "down2", "dec2", "dec1", "dec0", "head"};
    int sl = g_layer, ss = g_sub; size_t off = 0;
    if (!n) return 0;
    off += (size_t)snprintf(buf + off, n - off, "prec %s sr %d requested_policy:", pname16(g_prec), sr_on());
    for (int l = 0; l < 11 && off < n; l++) for (int s2 = 0; s2 < (l >= 4 && l <= 6 ? 1 : l == 10 ? 1 : 2) && off < n; s2++) {
        g_layer = l; g_sub = (l >= 4 && l <= 6) || l == 10 ? -1 : s2;
        off += (size_t)snprintf(buf + off, n - off, " %s%s=%s:%s:%s", names[l], g_sub < 0 ? "" : s2 ? ".c2" : ".c1", pname16(eff_prec_pass(0)), pname16(eff_prec_pass(1)), pname16(eff_prec_pass(2)));
    }
    g_layer = sl; g_sub = ss;
    return (int)(off < n ? off : n - 1);
}
extern "C" int nn_exec_manifest(char *buf, size_t n) {
    static const char *names[] = {"enc0", "enc1", "enc2", "enc3", "down0", "down1", "down2", "dec2", "dec1", "dec0", "head"};
    if (!n) return 0;
    size_t off = (size_t)snprintf(buf, n, "executed_compute (fwd:bwd_data:wgrad; - = not observed):");
    for (int l = 0; l < 11 && off < n; l++) for (int s2 = 0; s2 < ((l >= 4 && l <= 6) || l == 10 ? 1 : 2) && off < n; s2++) {
        off += (size_t)snprintf(buf + off, n - off, " %s%s=", names[l], l >= 4 && (l <= 6 || l == 10) ? "" : s2 ? ".c2" : ".c1");
        for (int pass = 0; pass < 3 && off < n; pass++) {
            if (pass) off += (size_t)snprintf(buf + off, n - off, ":");
            unsigned mask = __atomic_load_n(&g_exec_prec[l][s2][pass], __ATOMIC_RELAXED);
            if (!mask && off < n) off += (size_t)snprintf(buf + off, n - off, "-");
            int sep = 0;
            for (int p = 0; p < 5 && off < n; p++) if (mask & (1u << p)) {
                off += (size_t)snprintf(buf + off, n - off, "%s%s", sep ? "|" : "", pname16(p)); sep = 1;
            }
        }
    }
    return (int)(off < n ? off : n - 1);
}
extern "C" int nn_get_f16(void) { return g_h16; }
float g_gscale = 1.f;
extern "C" void nn_set_grad_scale(float s) { g_gscale = s; }
extern "C" float nn_get_grad_scale(void) { return g_gscale; }
int g_gradbf = 1;
extern "C" void nn_set_grad_bf16(int on) { g_gradbf = on; }
extern "C" int nn_get_grad_bf16(void) { return g_gradbf; }
cudaError_t g_err = cudaSuccess;
extern "C" int nn_init(int device) { if (ufsm_env_on("UFSM_ACTF32")) g_actbf = 0; if (ufsm_env_on("UFSM_GRADF32")) g_gradbf = 0; if (ufsm_env_on("UFSM_F16")) { g_h16 = 1; g_gscale = getenv("UFSM_GSCALE") ? (float)atof(getenv("UFSM_GSCALE")) : 1024.f; } return cudaSetDevice(device) == cudaSuccess ? 0 : -1; }
extern "C" const char *nn_check(void) {
    cudaError_t e = g_err;
    g_err = cudaSuccess;
    if (e == cudaSuccess) e = cudaGetLastError();
    const char *lp_error = lp_check();   /* low-precision kernels clear CUDA's last error into their own buffer */
    return e == cudaSuccess ? lp_error : cudaGetErrorString(e);
}
extern "C" void *nn_malloc(size_t n) { void *p = nullptr; CK(cudaMalloc(&p, n)); return p; }
g_reg_t g_reg[NN_MAXREG];
int g_nreg;
extern "C" void nn_storage_forget(const void *p) { for (int i = 0; i < g_nreg; i++) if (g_reg[i].p == (const char *)p) { g_reg[i] = g_reg[--g_nreg]; return; } }
extern "C" void nn_set_storage(const void *p, size_t bytes, int dt) {
    nn_storage_forget(p);
    if (!p || !dt) return;
    if (g_nreg >= NN_MAXREG) { fprintf(stderr, "nn_set_storage: registry full\n"); abort(); }
    g_reg[g_nreg].p = (const char *)p; g_reg[g_nreg].n = bytes; g_reg[g_nreg].dt = dt; g_nreg++;
}
extern "C" int nn_storage(const void *p) {
    const char *c = (const char *)p;
    for (int i = 0; i < g_nreg; i++) if (c >= g_reg[i].p && c < g_reg[i].p + g_reg[i].n) return g_reg[i].dt;
    return 0;
}
extern "C" size_t nn_mx8_bytes(shape5 s) { int bw = s.c <= 8 ? 8 : s.c <= 16 ? 16 : 32, nb = (s.c + bw - 1) / bw; return (size_t)s.n * nb * shape_spatial(s) * (bw + 1); }   /* = lp_mx8_bytes */
extern "C" size_t nn_mx4_bytes(shape5 s) { int bw = s.c <= 8 ? 8 : s.c <= 16 ? 16 : 32, nb = (s.c + bw - 1) / bw; return (size_t)s.n * nb * shape_spatial(s) * (bw / 2 + 1); }   /* = lp_mx4_bytes */
extern "C" size_t nn_mx_bytes(shape5 s, int dt) { return dt == 4 ? nn_mx4_bytes(s) : nn_mx8_bytes(s); }   /* registry dt */
extern "C" void nn_free(void *p) { if (p) { if (g_nreg) nn_storage_forget(p); CK(cudaFree(p)); } }
extern "C" void nn_zero(void *p, size_t n) { CK(cudaMemset(p, 0, n)); }
extern "C" void nn_h2d(void *d, const void *s, size_t n) { CK(cudaMemcpy(d, s, n, cudaMemcpyHostToDevice)); }
extern "C" void nn_d2h(void *d, const void *s, size_t n) { CK(cudaMemcpy(d, s, n, cudaMemcpyDeviceToHost)); }
extern "C" void nn_d2d(void *d, const void *s, size_t n) { CK(cudaMemcpy(d, s, n, cudaMemcpyDeviceToDevice)); }
extern "C" void nn_sync(void) { CK(cudaDeviceSynchronize()); }
extern "C" void *nn_host_alloc(size_t n) { void *p = nullptr; CK(cudaMallocHost(&p, n)); return p; }
extern "C" void nn_host_free(void *p) { if (p) CK(cudaFreeHost(p)); }
cudaStream_t copy_stream(void) {
    static cudaStream_t st[8]; int d = 0; cudaGetDevice(&d); d &= 7;
    if (!st[d]) CK(cudaStreamCreateWithFlags(&st[d], cudaStreamNonBlocking));
    return st[d];
}
extern "C" void nn_h2d_copy_stream(void *d, const void *s, size_t n) { CK(cudaMemcpyAsync(d, s, n, cudaMemcpyHostToDevice, copy_stream())); }   /* s must be pinned */
extern "C" void *nn_event_create(void) { cudaEvent_t e; CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); return (void *)e; }
extern "C" void nn_event_record(void *e, int on_copy_stream) { CK(cudaEventRecord((cudaEvent_t)e, on_copy_stream ? copy_stream() : 0)); }
extern "C" void nn_stream_wait(int copy_stream_waits, void *e) { CK(cudaStreamWaitEvent(copy_stream_waits ? copy_stream() : 0, (cudaEvent_t)e, 0)); }
extern "C" void nn_event_sync(void *e) { CK(cudaEventSynchronize((cudaEvent_t)e)); }
cudaEvent_t g_ev[NPROF][2];
int g_evk[NPROF], g_nev, g_ev_init;
extern "C" void nn_prof_begin(int k) {
    if (!g_ev_init) { for (int i = 0; i < NPROF; i++) { cudaEventCreate(&g_ev[i][0]); cudaEventCreate(&g_ev[i][1]); } g_ev_init = 1; }
    if (g_nev >= NPROF) return;
    g_evk[g_nev] = k; cudaEventRecord(g_ev[g_nev][0], 0);
}
extern "C" void nn_prof_end(void) { if (g_nev < NPROF) { cudaEventRecord(g_ev[g_nev][1], 0); g_nev++; } }
extern "C" void nn_prof_collect(double *out, int nk) {
    cudaDeviceSynchronize();
    for (int i = 0; i < nk; i++) out[i] = 0;
    for (int i = 0; i < g_nev; i++) { float ms = 0; cudaEventElapsedTime(&ms, g_ev[i][0], g_ev[i][1]); if (g_evk[i] < nk) out[g_evk[i]] += ms; }
    g_nev = 0;
}
extern "C" size_t nn_mem_free(void) { size_t f = 0, t = 0; cudaMemGetInfo(&f, &t); return f; }
extern "C" void ufsm_no_verify_types(const char *fn) {
    fprintf(stderr, "%s: bf16 / fp32 tensor-core kernels are not in this build (rebuild with make VERIFY=1)\n", fn); abort();
}
