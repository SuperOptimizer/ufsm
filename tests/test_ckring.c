/* test_ckring: the trainer's checkpoint ring and snapshot retention (src/ckring.c, ops audit #10), on fake checkpoints. CPU.
   - ring: the newest N clones of last.ckpt, byte-identical; a filesystem without reflinks turns the ring off unless copies are allowed;
   - snapshots every S steps, kept at multiples of keep_every and the newest keep_last; other files in snap/ are never touched;
   - UFSM_TEST_REFLINK_DIR (a btrfs / xfs dir, e.g. /vesuvius/ufsm/tmp): the reflink path itself. */
#include "ckring.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); } else { printf("ok: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
static int exists(const char *d, const char *f) { char p[4200]; snprintf(p, sizeof p, "%s/%s", d, f); return access(p, F_OK) == 0; }
static void write_last(const char *out, int step) {
    char p[4200]; snprintf(p, sizeof p, "%s/last.ckpt", out); FILE *f = fopen(p, "wb"); fprintf(f, "UFSM{\"step\":%d}\n", step);
    for (int i = 0; i < 1000; i++) fwrite(&step, sizeof step, 1, f); fclose(f);
}
static int same(const char *a, const char *b) {
    FILE *x = fopen(a, "rb"), *y = fopen(b, "rb"); int ok = x && y, c, d;
    while (ok) { c = fgetc(x); d = fgetc(y); if (c != d) ok = 0; if (c == EOF || d == EOF) break; }
    if (x) fclose(x); if (y) fclose(y); return ok;
}
static void run(const char *root, int allow_copy, int expect_ring) {
    char out[4096]; snprintf(out, sizeof out, "%s/ckr.%d.%d", root, (int)getpid(), allow_copy); mkdir(out, 0755);
    char snap[4200], ring[4200]; snprintf(snap, sizeof snap, "%s/snap", out); snprintf(ring, sizeof ring, "%s/ring", out);
    mkdir(snap, 0755); FILE *k = fopen(strcat(strcpy((char[4300]){0}, snap), "/4000.uevaldone"), "w"); if (k) fclose(k);   /* not a checkpoint: kept */
    ck_cfg c = {3, allow_copy, 500, 2000, 2, 0};
    for (int s = 250; s <= 4500; s += 250) { write_last(out, s); ck_after_save(&c, out, s); }
    printf("-- %s (allow_copy %d)\n", out, allow_copy);
    if (expect_ring) {
        CHECK(exists(ring, "4500.ckpt") && exists(ring, "4250.ckpt") && exists(ring, "4000.ckpt") && !exists(ring, "3750.ckpt"), "ring keeps the newest 3 (4000, 4250, 4500)");
        char a[4300], b[4300]; snprintf(a, sizeof a, "%s/last.ckpt", out); snprintf(b, sizeof b, "%s/4500.ckpt", ring);
        CHECK(same(a, b), "ring/4500.ckpt == last.ckpt");
    } else CHECK(!exists(ring, "4500.ckpt"), "no reflinks, no copies allowed: the ring is off");
    /* snapshots every 500: kept 2000, 4000 (keep_every) + the newest 2 others (3500, 4500); 4000.uevaldone untouched */
    CHECK(exists(snap, "2000.ckpt") && exists(snap, "4000.ckpt") && exists(snap, "4500.ckpt") && exists(snap, "3500.ckpt"), "snapshots 2000, 4000 (every 2000) and the newest kept");
    CHECK(!exists(snap, "500.ckpt") && !exists(snap, "1000.ckpt") && !exists(snap, "3000.ckpt"), "older snapshots off the 2000 grid pruned");
    CHECK(exists(snap, "4000.uevaldone"), "other files in snap/ untouched");
    char cmd[4400]; snprintf(cmd, sizeof cmd, "rm -rf -- '%s'", out); if (fails == 0 && system(cmd)) {}
}
int main(void) {
    const char *t = getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp";
    char probe_s[4200], probe_d[4200]; snprintf(probe_s, sizeof probe_s, "%s/ckr_probe.%d", t, (int)getpid()); snprintf(probe_d, sizeof probe_d, "%s.c", probe_s);
    FILE *f = fopen(probe_s, "w"); fputs("x", f); fclose(f);
    const int reflink = ck_clone(probe_s, probe_d, 0) == 0; unlink(probe_s); unlink(probe_d);
    run(t, 0, reflink);
    run(t, 1, 1);
    const char *rd = getenv("UFSM_TEST_REFLINK_DIR");
    if (rd) { snprintf(probe_s, sizeof probe_s, "%s/ckr_probe.%d", rd, (int)getpid()); snprintf(probe_d, sizeof probe_d, "%s.c", probe_s);
              f = fopen(probe_s, "w"); fputs("x", f); fclose(f);
              CHECK(ck_clone(probe_s, probe_d, 0) == 0, "reflink in %s", rd); unlink(probe_s); unlink(probe_d); run(rd, 0, 1); }
    printf(fails ? "test_ckring: %d failure(s)\n" : "test_ckring: ok\n", fails);
    return fails != 0;
}
