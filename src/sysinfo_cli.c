/* ufsm sysinfo [--ncpu | --avail-gb | --limit-gb] [--disk PATH]...: the real limits a pod gives this process (src/sysinfo.h):
   cgroup CPUs vs nproc, cgroup memory vs /proc/meminfo, the CUDA devices (MIG slices), the build, free disk. The single-value
   flags print one integer for scripts (unroll.sh MemAvailable gate, thread counts). */
#include "sysinfo.h"
#include <stdint.h>
#include <sys/statvfs.h>

int nn_sysinfo_gpus(FILE *o);

static void disk_line(const char *p) {
    struct statvfs s;
    if (statvfs(p, &s)) { printf("disk %s: (cannot stat)\n", p); return; }
    printf("disk %s: %.1f GB free of %.1f GB\n", p, (double)s.f_bavail * s.f_frsize / 1e9, (double)s.f_blocks * s.f_frsize / 1e9);
}
int cmd_sysinfo(int argc, char **argv) {
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--ncpu")) { printf("%d\n", ufsm_ncpu()); return 0; }
        if (!strcmp(argv[i], "--avail-gb")) { printf("%d\n", (int)(ufsm_mem_available() >> 30)); return 0; }
        if (!strcmp(argv[i], "--limit-gb")) { printf("%d\n", (int)(ufsm_mem_limit() >> 30)); return 0; }
    }
    const long on = sysconf(_SC_NPROCESSORS_ONLN); cpu_set_t cs; CPU_ZERO(&cs);
    const int aff = sched_getaffinity(0, sizeof cs, &cs) ? -1 : CPU_COUNT(&cs);
    const double q = ufsm_cpu_quota(); char cg[512]; ufsm_cg_path(cg, sizeof cg);
    printf("cpu: ufsm uses %d threads by default (nproc %ld, affinity %d, cgroup quota %s%.2f)%s\n", ufsm_ncpu(), on, aff,
           q > 0 ? "" : "none ", q, getenv("UFSM_NCPU") ? " [UFSM_NCPU]" : "");
    printf("cgroup: %s\n", cg[0] ? cg : "/");
    const size_t tot = ufsm_meminfo("MemTotal"), mav = ufsm_meminfo("MemAvailable"), lim = ufsm_mem_limit(), av = ufsm_mem_available();
    printf("memory: limit %.1f GB, available %.1f GB (/proc/meminfo: total %.1f GB, available %.1f GB)%s\n", lim / 1e9, av / 1e9, tot / 1e9, mav / 1e9,
           lim < tot ? " [cgroup limit]" : "");
    printf("build: gcc %s, C %ld\n", __VERSION__, (long)__STDC_VERSION__);
    nn_sysinfo_gpus(stdout);
    int nd = 0;
    for (int i = 2; i < argc; i++) if (!strcmp(argv[i], "--disk") && i + 1 < argc) { disk_line(argv[++i]); nd++; }
    if (!nd) disk_line(".");
    return 0;
}
