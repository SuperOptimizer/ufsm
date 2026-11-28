/* test_wrap_spacing: ufsm wrap-spacing on a synthetic wrap-check --intervals table. Concentric wrap pairs k = 0..5 at 8 z levels x 180
   rays, gap 220 um everywhere (GT right) except (a) a packed run: pair 3 on rays 40..59 of every z with gap 100 um, GT wrong (two
   wraps on one sheet); (b) one along-sheet jump: pair 1, ray 100, z 3, gap 400 um, GT right. Checks: the --out table's flag (gap < 150)
   is set on exactly the packed run, the jump interval's along score st is the |ln| jump (ln(400 / 220) = 0.598), a normal interval's
   st is 0, the radial score sr of a packed interval is ln(220 / 100), and the command returns 0. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_wrap_spacing(int argc, char **argv);

int main(void) {
    char in[128], out[128]; snprintf(in, sizeof in, "/tmp/test_wrap_spacing.%d.in.tsv", (int)getpid()); snprintf(out, sizeof out, "/tmp/test_wrap_spacing.%d.out.tsv", (int)getpid());
    FILE *f = fopen(in, "w"); if (!f) { printf("FAIL cannot write %s\n", in); return 1; }
    fprintf(f, "z\ty\tx\tr_mm\tk\tnpk\trct\tgap_um\tgtq\n");
    long npk = 0;
    for (int z = 0; z < 8; z++) for (int k = 0; k < 6; k++) for (int t = 0; t < 180; t++) {
        const double th = 2 * M_PI * t / 180, r = 100 + 25 * k, y = 1000 + r * sin(th), x = 1000 + r * cos(th);
        const int packed = k == 3 && t >= 40 && t < 60, jump = k == 1 && t == 100 && z == 3;
        const double gap = packed ? 100 : jump ? 400 : 220; npk += packed;
        fprintf(f, "%d\t%.1f\t%.1f\t%.3f\t%d\t1\t%.3f\t%.1f\t%.3f\n", 100 + 24 * z, y, x, r * 0.0096, k, gap, gap, packed ? 0.0 : 1.0);
    }
    if (fclose(f)) { printf("FAIL write\n"); return 1; }
    char *argv[] = {"ufsm", "wrap-spacing", in, "--out", out, nullptr};
    const int rc = cmd_wrap_spacing(5, argv);
    int bad = rc != 0; if (rc) printf("FAIL rc %d\n", rc);
    FILE *g = fopen(out, "r"); if (!g) { printf("FAIL no --out table\n"); return 1; }
    char line[512]; if (!fgets(line, sizeof line, g)) bad = 1;
    long n = 0, nflag = 0, flag_packed = 0; double st_jump = NAN, st_norm = NAN, sr_packed = NAN;
    while (fgets(line, sizeof line, g)) {
        float z, y, x, gap, st, sz, sr, sct, al, gtq; int k, fl;
        if (sscanf(line, "%f %f %f %d %f %f %f %f %f %f %d %f", &z, &y, &x, &k, &gap, &st, &sz, &sr, &sct, &al, &fl, &gtq) != 12) { bad = 1; break; }
        n++; nflag += fl; if (fl && gap < 150 && gtq == 0) flag_packed++;
        if (gap == 400) st_jump = st;
        if (k == 1 && gap == 220 && z == 100 && isnan(st_norm)) st_norm = st;
        if (k == 3 && gap == 100 && z == 196 && isnan(sr_packed)) sr_packed = sr;
    }
    fclose(g);
    if (n != 8 * 6 * 180) { printf("FAIL %ld rows\n", n); bad = 1; }
    if (nflag != npk || flag_packed != npk) { printf("FAIL flags %ld (packed flagged %ld) vs %ld packed\n", nflag, flag_packed, npk); bad = 1; } else printf("ok   flag = the %ld packed intervals\n", npk);
    if (!(fabs(st_jump - log(400.0 / 220)) < 0.01)) { printf("FAIL jump st %.3f\n", st_jump); bad = 1; } else printf("ok   jump st %.3f\n", st_jump);
    if (!(fabs(st_norm) < 1e-3)) { printf("FAIL normal st %.3f\n", st_norm); bad = 1; } else printf("ok   normal st %.3f\n", st_norm);
    if (!(fabs(sr_packed - log(220.0 / 100)) < 0.01)) { printf("FAIL packed sr %.3f\n", sr_packed); bad = 1; } else printf("ok   packed sr %.3f\n", sr_packed);
    remove(in); remove(out);
    printf(bad ? "FAIL\n" : "PASS\n");
    return bad;
}
