/* Portable inference defaults embedded in a training checkpoint's JSON header. */
#pragma once
#include <stddef.h>

#define UFSM_CHECKPOINT_HEADER 16384
typedef struct {
    int version, train_window, prec, f16, act_mx4, act_mx8, grad_mx8, input_mx, input_prec, gn_stored;
    char policy[2048], optimizer[16];
} checkpoint_runtime;

/* 1: versioned settings read, 0: legacy checkpoint, -1: invalid header/settings. No CUDA required. */
int checkpoint_runtime_read(const char *path, checkpoint_runtime *r);
/* Serializes an extra object suitable for unet_save. Returns -1 on invalid values or truncation. */
int checkpoint_runtime_json(const checkpoint_runtime *r, char *out, size_t cap);
/* the checkpoint's JSON header line (malloc'ed, free it; nullptr when unreadable), e.g. for strstr of "\"unwrap\":{\"version\":2" */
char *checkpoint_header_line(const char *path);
/* the task unwrap layout of a checkpoint: 2 task unwrap2 (0-1 phase, 2-7 normal, 8 density, 9-10 winding mod 4), 1 task unwrap
   (0 recto, 1 ink, 2-3 phase, 4-9 normal, 10 density), 0 neither / unreadable */
int checkpoint_unwrap_version(const char *path);
