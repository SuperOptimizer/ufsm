/* test_xnet_oom: a failed xnet device allocation stops the process (stderr line, exit XNET_OOM_RC) instead of handing out nullptr
   (11-18: ink-ext reader3 launched on four null buffers after cudaMalloc failures -> Xid 31 on desk card 1). CPU only: the forced
   failure (UFSM_XNET_FAIL_ALLOC=1) fires before any CUDA call, and CUDA_VISIBLE_DEVICES is emptied first so nothing can open a card. */
#include "xnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int child(const char *fail, size_t bytes, char *err, size_t cap) {   /* exit status of xmalloc_dev(bytes) in a child; its stderr in err */
    fflush(stdout); int pfd[2]; if (pipe(pfd)) return -1;   /* (flush: the child flushes stdio before its _exit) */
    const pid_t pid = fork(); if (pid < 0) return -1;
    if (!pid) {
        close(pfd[0]); dup2(pfd[1], 2); setenv("UFSM_XNET_FAIL_ALLOC", fail, 1);
        void *p = xmalloc_dev(bytes);
        fprintf(stderr, "returned %p\n", p); _exit(p ? 10 : 11);   /* reaching here at all is the bug */
    }
    close(pfd[1]); size_t n = 0; ssize_t r;
    while (n + 1 < cap && (r = read(pfd[0], err + n, cap - 1 - n)) > 0) n += (size_t)r;
    err[n] = 0; close(pfd[0]);
    int st = 0; waitpid(pid, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
}
int main(void) {
    setenv("CUDA_VISIBLE_DEVICES", "", 1);
    int bad = 0; char err[4096];
    const size_t sz[] = {1, 4096, 9u << 20, 288u << 20};
    for (int i = 0; i < 4; i++) {
        const int rc = child("1", sz[i], err, sizeof err);
        const int ok = rc == XNET_OOM_RC && strstr(err, "out of memory") && strstr(err, "UFSM_XNET_FAIL_ALLOC") && !strstr(err, "returned");
        printf("forced failure, %zu bytes: exit %d %s\n", sz[i], rc, ok ? "ok" : "FAIL"); if (!ok) { bad++; fputs(err, stdout); }
    }
    printf("%s\n", bad ? "FAIL" : "PASS"); return bad != 0;
}
