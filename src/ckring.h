#pragma once
/* ckring.h: the trainer's checkpoint ring and snapshot retention (src/ckring.c). */
typedef struct {
    int ring;             /* --ckpt-ring N: keep the newest N clones of last.ckpt in <out>/ring (0 off; default 4) */
    int allow_copy;       /* --ckpt-ring-copy 1: plain copies where reflinks are not available (default 0: the ring is then off) */
    int snap_every;       /* --snap-every S: <out>/snap/<step>.ckpt every S steps (0 off: the launch script's loop) */
    int snap_keep_every;  /* --snap-keep-every K: snapshots at multiples of K are kept for good (default 5000) */
    int snap_keep_last;   /* --snap-keep-last L: and the newest L others (default 5) */
    int ring_off;         /* (state: set when the filesystem has no reflinks and copies are not allowed) */
} ck_cfg;
int ck_clone(const char *src, const char *dst, int allow_copy);   /* 0 reflink, 1 copied, -1 error, -2 no reflink and !allow_copy */
int ck_prune(const char *dir, int keep_last, int keep_every);     /* delete DIR/<step>.ckpt except the newest keep_last and step % keep_every == 0 */
int ck_after_save(ck_cfg *c, const char *out, int step);   /* after last.ckpt of step was written: ring + snapshot + retention */
