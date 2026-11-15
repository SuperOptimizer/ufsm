/* ckring.c (2026-11-14, ops audit #10): the trainer's checkpoint ring and snapshot retention.
   last.ckpt is rewritten every --ckpt-every steps (tmp + rename), so the only rollback was the launch script's snap/<step>.ckpt every
   1000 steps: the try6 deg1900 rollback lost 900 steps (~1.5 h), and the script copied the whole 2.08 GB file to snap/.tmp after each
   save (8 GB written per 1000 steps). Here the trainer clones each new last.ckpt into ring/<step>.ckpt with FICLONE (a reflink: no data
   copied on btrfs / xfs, /vesuvius is btrfs) and keeps the newest N; snapshots likewise into snap/<step>.ckpt every S steps, kept when
   step % keep_every == 0 or among the newest keep_last (finished 5000s for the evals, the last few 1000s for rollback; snap/ grew
   without bound, 71 x 2.08 GB at step 71000). Filesystems without reflinks: a plain copy only when allowed (allow_copy), else the ring
   is off with one message. Only files named <digits>.ckpt are ever deleted. */
#include "ckring.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/fs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

static int copy_plain(int in, int out) {
    char *b = malloc(1 << 22); if (!b) return -1; ssize_t r; int rc = 0;
    while ((r = read(in, b, 1 << 22)) > 0) { for (ssize_t o = 0; o < r;) { const ssize_t w = write(out, b + o, r - o); if (w <= 0) { rc = -1; break; } o += w; } if (rc) break; }
    if (r < 0) rc = -1;
    free(b); return rc;
}

/* clone src to dst (via dst.tmp + rename); 0 = reflink, 1 = plain copy, -1 = failed (errno-style message printed) */
int ck_clone(const char *src, const char *dst, int allow_copy) {
    char tmp[4200]; snprintf(tmp, sizeof tmp, "%s.tmp", dst);
    const int in = open(src, O_RDONLY); if (in < 0) { fprintf(stderr, "ckring: cannot open %s: %s\n", src, strerror(errno)); return -1; }
    const int out = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644); if (out < 0) { fprintf(stderr, "ckring: cannot create %s: %s\n", tmp, strerror(errno)); close(in); return -1; }
    int how = 0;
    if (ioctl(out, FICLONE, in) != 0) {
        if (!allow_copy) { close(in); close(out); unlink(tmp); errno = EOPNOTSUPP; return -2; }
        how = 1; if (copy_plain(in, out)) how = -1;
    }
    close(in); if (close(out)) how = -1;
    if (how < 0 || rename(tmp, dst)) { fprintf(stderr, "ckring: cannot write %s: %s\n", dst, strerror(errno)); unlink(tmp); return -1; }
    return how;
}

static int cmp_long(const void *a, const void *b) { const long x = *(const long *)a, y = *(const long *)b; return (x > y) - (x < y); }

/* the steps of DIR's <digits>.ckpt files, ascending; returns the count (*steps malloc'd) */
static int list_steps(const char *dir, long **steps) {
    DIR *d = opendir(dir); *steps = nullptr; if (!d) return 0;
    int n = 0, cap = 0; struct dirent *e;
    while ((e = readdir(d))) {
        const char *s = e->d_name; size_t k = 0; while (s[k] >= '0' && s[k] <= '9') k++;
        if (!k || strcmp(s + k, ".ckpt")) continue;
        if (n == cap) { cap = cap ? 2 * cap : 64; *steps = realloc(*steps, cap * sizeof(long)); }
        (*steps)[n++] = atol(s);
    }
    closedir(d); if (n) qsort(*steps, n, sizeof(long), cmp_long); return n;
}

int ck_prune(const char *dir, int keep_last, int keep_every) {
    long *st; const int n = list_steps(dir, &st); int del = 0, kept = 0;
    for (int i = n - 1; i >= 0; i--) {   /* newest first: the grid ones stay, plus the newest keep_last others */
        if (keep_every > 0 && st[i] % keep_every == 0) continue;
        if (kept < keep_last) { kept++; continue; }
        char p[4200]; snprintf(p, sizeof p, "%s/%ld.ckpt", dir, st[i]);
        if (!unlink(p)) del++;
    }
    free(st); return del;
}

int ck_after_save(ck_cfg *c, const char *out, int step) {
    char last[4096], d[4096], dst[4200]; snprintf(last, sizeof last, "%s/last.ckpt", out); int rc = 0;
    if (c->ring > 0 && !c->ring_off) {
        snprintf(d, sizeof d, "%s/ring", out); mkdir(d, 0755); snprintf(dst, sizeof dst, "%s/%d.ckpt", d, step);
        const int h = ck_clone(last, dst, c->allow_copy);
        if (h == -2) { fprintf(stderr, "ckring: %s cannot reflink (filesystem without FICLONE): the checkpoint ring is off (--ckpt-ring-copy 1 copies)\n", out); c->ring_off = 1; }
        else if (h < 0) rc = -1;
        else ck_prune(d, c->ring, 0);
    }
    if (c->snap_every > 0 && step > 0 && step % c->snap_every == 0) {
        snprintf(d, sizeof d, "%s/snap", out); mkdir(d, 0755); snprintf(dst, sizeof dst, "%s/%d.ckpt", d, step);
        const int h = ck_clone(last, dst, 1);   /* (a snapshot is wanted either way: copied when it cannot be cloned) */
        if (h < 0) rc = -1;
        else { fprintf(stderr, "snapshot %d -> %s (%s)\n", step, dst, h ? "copied" : "reflink");
               if (c->snap_keep_last > 0 || c->snap_keep_every > 0) ck_prune(d, c->snap_keep_last, c->snap_keep_every); }
    }
    return rc;
}
