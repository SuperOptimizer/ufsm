/* ctshell.h (inkeval 2026-11-27): a CT shell, a higher-quality partial copy of one CT level (S3 bytes re-encoded at volcomp q0 / q2 by
   latvol --exact 1) over the 128^3 chunks a reader needs; the rest still comes from the main store. Its list (shell.boxes): a header
   "# level L shard 1024 q Q (...)" and one "z y x 128 128 128k" x run of chunks a line, each inside one 1024^3 shard.
   Writers: ink-ext --shell-list (the chunks the renders read), ufsm ct-shell-list (a predict box + its window margin). Readers: ink-ext
   --ct-shell (per chunk), predict --ct-shell (the whole level, when the shell covers every stored chunk the windows read). */
#ifndef UFSM_CTSHELL_H
#define UFSM_CTSHELL_H
#include <stddef.h>
#include <stdint.h>

static inline int64_t ctshell_key(int64_t z, int64_t y, int64_t x) { return (z << 42) | (y << 21) | x; }   /* chunk indices */

/* The list of chunk keys (any order, duplicates allowed; keys outside the cgr chunk grid dropped) written to SLIST. sprev: an older list
   merged in (its grown shards, or all of them when its quality differs, into SLIST.redo). sq "auto" | "0".."9": the quality; auto =
   lossless while chunks x 128^3 x 0.7 B <= maxgb, else q2 with "DEFAULTED: ctshell q2 (lossless est N GB > M)". desc: the header's note.
   Frees keys. 0 ok; prints its own errors. *q_out: the quality written. */
int ctshell_write(const char *slist, int level, int64_t *keys, size_t n, const int64_t cgr[3], const char *sprev, const char *sq, double maxgb,
                  const char *desc, int *q_out);

/* Reads a shell list into a one-byte-per-chunk mask over the cgr chunk grid (calloc'd, *mask). *level / *q from the header (q -1: none).
   0 ok, 1 missing / unreadable / empty, 2 a box off the 128^3 chunk grid (err says which). */
int ctshell_mask(const char *path, const int64_t cgr[3], const int64_t shape[3], int *level, int *q, uint8_t **mask, size_t *nck, char *err, size_t errn);

int cmd_ct_shell_list(int argc, char **argv);
#endif
