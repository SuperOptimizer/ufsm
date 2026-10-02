/* Portable inference defaults embedded in a training checkpoint's JSON header. */
#pragma once
#include <stddef.h>

#define UFSM_CHECKPOINT_HEADER 16384
typedef struct {
    int version, train_window, prec, f16, act_mx4, act_mx8, grad_mx8;
    char policy[2048], optimizer[16];
} checkpoint_runtime;

/* 1: versioned settings read, 0: legacy checkpoint, -1: invalid header/settings. No CUDA required. */
int checkpoint_runtime_read(const char *path, checkpoint_runtime *r);
/* Serializes an extra object suitable for unet_save. Returns -1 on invalid values or truncation. */
int checkpoint_runtime_json(const checkpoint_runtime *r, char *out, size_t cap);
