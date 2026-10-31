/* src/sysinfo.h on fake /proc + /sys/fs/cgroup trees: a pod (nproc 256, cgroup v2 cpu.max 1360000 100000 at the container
   root, memory.max 62 GB of a 1 TB host), a nested scope tighter than its parent (systemd-run --scope -p CPUQuota / MemoryMax),
   no limits ("max"), cgroup v1, UFSM_NCPU and ufsm_threads(). */
#include "sysinfo.h"
#include <assert.h>
#include <sys/stat.h>

static char root[512];
static void put(const char *rel, const char *text) {
    char p[1100]; snprintf(p, sizeof p, "%s/%s", root, rel);
    for (char *s = p + strlen(root) + 1; (s = strchr(s, '/')); s++) { *s = 0; mkdir(p, 0755); *s = '/'; }
    FILE *f = fopen(p, "w"); assert(f); fputs(text, f); fclose(f);
}
static void setup(const char *name) {
    char t[600]; snprintf(root, sizeof root, "%s/%s", getenv("TMPDIR") && *getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", name);
    snprintf(t, sizeof t, "rm -rf '%s'", root); assert(system(t) == 0); mkdir(root, 0755);
    char c[600], p[600]; snprintf(c, sizeof c, "%s/cg", root); snprintf(p, sizeof p, "%s/proc", root);
    setenv("UFSM_CGROUP_ROOT", c, 1); setenv("UFSM_PROC_ROOT", p, 1); unsetenv("UFSM_NCPU"); ufsm_sysinfo_reset();
    put("proc/meminfo", "MemTotal:       1056000000 kB\nMemFree:  9 kB\nMemAvailable:   1000000000 kB\n");
}
static int near(double a, double b) { return a > b * 0.999 && a < b * 1.001; }
int main(void) {
    long host = sysconf(_SC_NPROCESSORS_ONLN);
    /* 1. pod: container root cgroup ("0::/"), cpu.max 13.6 CPUs, memory.max 62 GB with 2 GB in use */
    setup("ufsm_test_sysinfo_pod");
    put("proc/self/cgroup", "0::/\n");
    put("cg/cpu.max", "1360000 100000\n"); put("cg/memory.max", "66571993088\n"); put("cg/memory.current", "2147483648\n");
    assert(near(ufsm_cpu_quota(), 13.6));
    assert(ufsm_ncpu() == (host < 13 ? host : 13));
    assert(ufsm_mem_limit() == 66571993088ull);
    assert(ufsm_mem_available() == 66571993088ull - 2147483648ull);
    assert(ufsm_threads(0, 16) == (host < 13 ? host : 13) && ufsm_threads(0, 4) == (host < 4 ? host : 4) && ufsm_threads(20, 16) == 20);
    /* 2. a scope below an unlimited slice: the tightest level wins (scope 6 CPUs / 10 GB; slice max) */
    setup("ufsm_test_sysinfo_scope");
    put("proc/self/cgroup", "0::/user.slice/run-1.scope\n");
    put("cg/user.slice/cpu.max", "max 100000\n"); put("cg/user.slice/memory.max", "max\n");
    put("cg/user.slice/run-1.scope/cpu.max", "600000 100000\n"); put("cg/user.slice/run-1.scope/memory.max", "10737418240\n");
    put("cg/user.slice/run-1.scope/memory.current", "1073741824\n");
    assert(near(ufsm_cpu_quota(), 6.0)); assert(ufsm_ncpu() == (host < 6 ? host : 6));
    assert(ufsm_mem_limit() == 10737418240ull && ufsm_mem_available() == 9663676416ull);
    /* 3. no limits: the host (meminfo) */
    setup("ufsm_test_sysinfo_none");
    put("proc/self/cgroup", "0::/\n"); put("cg/cpu.max", "max 100000\n"); put("cg/memory.max", "max\n");
    assert(ufsm_cpu_quota() == 0); assert(ufsm_ncpu() >= 1 && ufsm_ncpu() <= host);
    assert(ufsm_mem_limit() == 1056000000ull * 1024 && ufsm_mem_available() == 1000000000ull * 1024);
    /* 4. cgroup v1: cpu,cpuacct quota 3.5 CPUs, memory 4 GB limit / 1 GB used */
    setup("ufsm_test_sysinfo_v1");
    put("proc/self/cgroup", "4:cpu,cpuacct:/\n3:memory:/\n");
    put("cg/cpu,cpuacct/cpu.cfs_quota_us", "350000\n"); put("cg/cpu,cpuacct/cpu.cfs_period_us", "100000\n");
    put("cg/memory/memory.limit_in_bytes", "4294967296\n"); put("cg/memory/memory.usage_in_bytes", "1073741824\n");
    assert(near(ufsm_cpu_quota(), 3.5)); assert(ufsm_ncpu() == (host < 3 ? host : 3));
    assert(ufsm_mem_limit() == 4294967296ull && ufsm_mem_available() == 3221225472ull);
    /* 5. quota below one CPU still gives one thread; UFSM_NCPU overrides */
    setup("ufsm_test_sysinfo_tiny");
    put("proc/self/cgroup", "0::/\n"); put("cg/cpu.max", "50000 100000\n");
    assert(ufsm_ncpu() == 1);
    setenv("UFSM_NCPU", "7", 1); ufsm_sysinfo_reset(); assert(ufsm_ncpu() == 7); assert(ufsm_threads(0, 16) == 7);
    char t[600]; snprintf(t, sizeof t, "rm -rf '%s'/../ufsm_test_sysinfo_*", root); (void)!system(t);
    printf("test_sysinfo: ok (host %ld CPUs)\n", host);
    return 0;
}
