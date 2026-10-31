#pragma once
/* Real machine limits for a process that may run in a container (rented GPU pods): nproc / sysconf report the host's CPUs
   (256 on a pod whose cgroup cpu.max allows 13.6) and /proc/meminfo the host's RAM (1 TB vs a 62 GB memory.max).
   ufsm_ncpu: CPUs this process may use = min(sched affinity, floor(cgroup quota / period) over this cgroup, its ancestors and
   the container's root cgroup; v2 cpu.max or v1 cpu.cfs_quota_us); UFSM_NCPU overrides. ufsm_mem_limit: min(MemTotal,
   memory.max along the same path). ufsm_mem_available: MemAvailable capped by every memory.max - memory.current
   (+ reclaimable page cache: inactive file pages, half the active ones).
   UFSM_CGROUP_ROOT (default /sys/fs/cgroup) and UFSM_PROC_ROOT (default /proc) point the readers at a fake tree (tests).
   Header-only (static inline) so every object that picks a thread count links without an extra object. */
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static inline const char *ufsm_cg_root(void) { const char *e = getenv("UFSM_CGROUP_ROOT"); return e && *e ? e : "/sys/fs/cgroup"; }
static inline const char *ufsm_proc_root(void) { const char *e = getenv("UFSM_PROC_ROOT"); return e && *e ? e : "/proc"; }

/* the cgroup v2 path of this process ("/a/b", "" when none / v1 only) */
static inline void ufsm_cg_path(char *out, size_t n) {
    char p[512], line[1024]; out[0] = 0;
    snprintf(p, sizeof p, "%s/self/cgroup", ufsm_proc_root());
    FILE *f = fopen(p, "r"); if (!f) return;
    while (fgets(line, sizeof line, f)) if (!strncmp(line, "0::", 3)) { line[strcspn(line, "\n")] = 0; snprintf(out, n, "%s", line + 3); break; }
    fclose(f);
    if (!strcmp(out, "/")) out[0] = 0;
}
/* calls fn(dir, ctx) for the cgroup directory of this process, each ancestor and the root (each once) */
static inline void ufsm_cg_walk(void (*fn)(const char *dir, void *ctx), void *ctx) {
    char cg[512], d[1100]; ufsm_cg_path(cg, sizeof cg);
    for (;;) {
        snprintf(d, sizeof d, "%s%s", ufsm_cg_root(), cg); fn(d, ctx);
        if (!cg[0]) break;
        char *s = strrchr(cg, '/'); if (!s) break; *s = 0;
    }
}
static inline int ufsm_read_ll2(const char *path, long long *a, long long *b) {   /* "A B" / "max B" (a = -1) / "A" */
    FILE *f = fopen(path, "r"); if (!f) return 0;
    char w[64] = ""; long long y = 0; int n = fscanf(f, "%63s %lld", w, &y); fclose(f);
    if (n < 1) return 0;
    *a = !strcmp(w, "max") ? -1 : atoll(w); if (b) *b = n == 2 ? y : 0;
    return 1;
}
static inline void ufsm_cg_cpu_fn(const char *d, void *ctx) {
    double *q = ctx; char p[1200]; long long a, b;
    snprintf(p, sizeof p, "%s/cpu.max", d);
    if (ufsm_read_ll2(p, &a, &b) && a > 0 && b > 0 && (*q <= 0 || (double)a / b < *q)) *q = (double)a / b;
}
/* cgroup CPU quota in CPUs (0: none): v2 cpu.max along the path, v1 cpu.cfs_quota_us / cpu.cfs_period_us at the v1 roots */
static inline double ufsm_cpu_quota(void) {
    double q = 0; ufsm_cg_walk(ufsm_cg_cpu_fn, &q);
    static const char *v1[] = {"cpu", "cpu,cpuacct", "cpuacct,cpu"};
    for (int i = 0; i < 3; i++) {
        char p[700]; long long a, b; snprintf(p, sizeof p, "%s/%s/cpu.cfs_quota_us", ufsm_cg_root(), v1[i]);
        if (!ufsm_read_ll2(p, &a, nullptr) || a <= 0) continue;
        snprintf(p, sizeof p, "%s/%s/cpu.cfs_period_us", ufsm_cg_root(), v1[i]);
        if (ufsm_read_ll2(p, &b, nullptr) && b > 0 && (q <= 0 || (double)a / b < q)) q = (double)a / b;
    }
    return q;
}
[[maybe_unused]] static int ufsm_ncpu_cache;
static inline void ufsm_sysinfo_reset(void) { ufsm_ncpu_cache = 0; }
/* CPUs this process may use (>= 1) */
static inline int ufsm_ncpu(void) {
    if (ufsm_ncpu_cache > 0) return ufsm_ncpu_cache;
    const char *e = getenv("UFSM_NCPU"); int n = e ? atoi(e) : 0;
    if (n <= 0) {
        long c = sysconf(_SC_NPROCESSORS_ONLN); n = c > 0 ? (int)c : 1;
        cpu_set_t cs; CPU_ZERO(&cs);
        if (!sched_getaffinity(0, sizeof cs, &cs)) { const int a = CPU_COUNT(&cs); if (a > 0 && a < n) n = a; }
        const double q = ufsm_cpu_quota();
        if (q > 0) { int qn = (int)(q + 1e-6); if (qn < 1) qn = 1; if (qn < n) n = qn; }
    }
    return ufsm_ncpu_cache = n < 1 ? 1 : n;
}
/* an option's thread count: the requested one, else the option's default capped by the real CPUs */
static inline int ufsm_threads(int requested, int def) {
    if (requested > 0) return requested;
    const int c = ufsm_ncpu(); return def < c ? (def < 1 ? 1 : def) : c;
}
static inline size_t ufsm_meminfo(const char *key) {   /* a /proc/meminfo field in bytes (0 unknown) */
    char p[512], line[256]; size_t v = 0, kl = strlen(key); snprintf(p, sizeof p, "%s/meminfo", ufsm_proc_root());
    FILE *f = fopen(p, "r"); if (!f) return 0;
    while (fgets(line, sizeof line, f)) if (!strncmp(line, key, kl) && line[kl] == ':') { v = (size_t)strtoull(line + kl + 1, nullptr, 10) << 10; break; }
    fclose(f); return v;
}
typedef struct { size_t lim, avail; } ufsm_mem_ctx;
static inline void ufsm_cg_mem_fn(const char *d, void *ctx) {
    ufsm_mem_ctx *m = ctx; char p[1200]; long long mx, cur;
    snprintf(p, sizeof p, "%s/memory.max", d);
    if (!ufsm_read_ll2(p, &mx, nullptr) || mx <= 0) return;
    if (!m->lim || (size_t)mx < m->lim) m->lim = (size_t)mx;
    snprintf(p, sizeof p, "%s/memory.current", d);
    if (!ufsm_read_ll2(p, &cur, nullptr)) cur = 0;
    /* memory.current includes the page cache, which the kernel reclaims before it OOM-kills: count inactive file pages as free and
       half of the active ones (a cgroup full of cached CT shards would otherwise look full) */
    snprintf(p, sizeof p, "%s/memory.stat", d);
    FILE *f = fopen(p, "r");
    if (f) { char k[64]; long long v, ia = 0, ac = 0;
             while (fscanf(f, "%63s %lld", k, &v) == 2) { if (!strcmp(k, "inactive_file")) ia = v; else if (!strcmp(k, "active_file")) ac = v; }
             fclose(f); const long long rec = ia + ac / 2; cur = cur > rec ? cur - rec : 0; }
    const size_t left = mx > cur ? (size_t)(mx - cur) : 0;
    if (left < m->avail) m->avail = left;
}
static inline void ufsm_mem_scan(ufsm_mem_ctx *m) {
    m->lim = ufsm_meminfo("MemTotal"); m->avail = ufsm_meminfo("MemAvailable"); if (!m->avail) m->avail = (size_t)-1;
    ufsm_mem_ctx c = {0, m->avail}; ufsm_cg_walk(ufsm_cg_mem_fn, &c);
    char p[700]; long long v1; snprintf(p, sizeof p, "%s/memory/memory.limit_in_bytes", ufsm_cg_root());   /* cgroup v1 */
    if (ufsm_read_ll2(p, &v1, nullptr) && v1 > 0 && v1 < (1ll << 60) && (!c.lim || (size_t)v1 < c.lim)) {
        c.lim = (size_t)v1; long long u; snprintf(p, sizeof p, "%s/memory/memory.usage_in_bytes", ufsm_cg_root());
        if (ufsm_read_ll2(p, &u, nullptr) && u < v1 && (size_t)(v1 - u) < c.avail) c.avail = (size_t)(v1 - u);
    }
    if (c.lim && (!m->lim || c.lim < m->lim)) m->lim = c.lim;
    m->avail = c.avail == (size_t)-1 ? 0 : c.avail;
}
static inline size_t ufsm_mem_limit(void) { ufsm_mem_ctx m; ufsm_mem_scan(&m); return m.lim; }
static inline size_t ufsm_mem_available(void) { ufsm_mem_ctx m; ufsm_mem_scan(&m); return m.avail; }

int cmd_sysinfo(int argc, char **argv);      /* src/sysinfo_cli.c: ufsm sysinfo [--ncpu | --avail-gb | --limit-gb] [--disk PATH]... */
