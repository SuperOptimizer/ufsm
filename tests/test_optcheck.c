/* test_optcheck: src/optcheck.h, the shared option bookkeeping (mini audit opts 2, 3, 7): strict numbers (exit 2 on junk, in a child),
   the unread-flag warning (once per flag, values and prefix families skipped), more than 512 names */
#include "optcheck.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)
static int child_rc(int kind, const char *v) {   /* the exit status of parsing v in a child (-1: returned normally) */
    fflush(stderr); const pid_t p = fork();
    if (!p) { int fd = open("/dev/null", 1); if (fd >= 0) dup2(fd, 2);
              if (kind == 0) (void)opt_atoi("--x", v); else if (kind == 1) (void)opt_atof("--x", v); else (void)opt_atoll("--x", v); _exit(99); }
    int st = 0; waitpid(p, &st, 0); return WIFEXITED(st) && WEXITSTATUS(st) != 99 ? WEXITSTATUS(st) : -1;
}
int main(void) {
    CHECK(opt_atoi("--x", "512") == 512 && opt_atoi("--x", "-3") == -3 && opt_atoi("--x", " 7 ") == 7, "plain integers");
    CHECK(opt_atoi("--x", "") == 0 && opt_atoi("--x", nullptr) == 0 && opt_atof("--x", nullptr) == 0, "empty / absent: 0");
    CHECK(opt_atof("--x", "2e4") == 2e4 && opt_atof("--x", "0.25") == 0.25 && opt_atof("--x", "-1") == -1, "plain numbers");
    CHECK(opt_atoll("--x", "123456789012") == 123456789012ll, "long long");
    const char *badi[] = {"2e4", "288x", "abc", "1.5", "0,1", "99999999999"};
    for (int k = 0; k < 6; k++) CHECK(child_rc(0, badi[k]) == 2, "opt_atoi(\"%s\") must exit 2", badi[k]);
    const char *badf[] = {"1.5x", "abc", "0.1,0.2"};
    for (int k = 0; k < 3; k++) CHECK(child_rc(1, badf[k]) == 2, "opt_atof(\"%s\") must exit 2", badf[k]);
    CHECK(child_rc(0, "42") == -1 && child_rc(1, "1e-3") == -1, "valid values return");
    /* unread flags: argv[1] is the command; --known (with a value) and --fam-x (prefix family) are read, --typo and --bool-typo are not */
    opt_n("--known"); opt_p("--fam-"); opt_n("not-a-flag");
    for (int k = 0; k < 1000; k++) { char b[32]; snprintf(b, sizeof b, "--n%d", k); opt_n(b); }   /* (past the old 512 cap) */
    char *av[] = {"ufsm", "cmd", "pos", "--known", "5", "--typo", "7", "--fam-x", "1", "--n999", "-1", "--bool-typo", "--known", "--x"};
    const int ac = (int)(sizeof av / sizeof *av);
    CHECK(opt_unread("cmd", ac, av) == 3, "three unread flags (--typo, --bool-typo, --x)");
    CHECK(opt_unread("cmd", ac, av) == 0, "each warned once");
    opt_n("--late");
    char *av2[] = {"ufsm", "cmd", "--late", "1"};
    CHECK(opt_unread("cmd", 4, av2) == 0, "a name read before the check");
    if (fails) { fprintf(stderr, "test_optcheck: %d FAILED\n", fails); return 1; }
    fprintf(stderr, "test_optcheck: ok\n"); return 0;
}
