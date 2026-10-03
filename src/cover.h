#pragma once
#include "sources.h"
#include <stdint.h>

typedef struct {
    int P;
    uint64_t count;
    int64_t (*tiles)[4]; /* source index, native z/y/x origin, already shuffled */
    char sha256[65];
} cover_plan;
typedef struct {
    char sha256[65];
    uint64_t count, cursor;
    int base_step;
} cover_progress;

cover_plan *cover_load(const char *path, const sources *S, int P);
void cover_free(cover_plan *p);
/* Explicit extension: the saved plan must be complete and an unchanged prefix. */
int cover_validate_extension(const cover_plan *previous, const cover_plan *next, const cover_progress *saved);
/* 1 finite-cover checkpoint, 0 ordinary checkpoint, -1 malformed. */
int cover_checkpoint_read(const char *path, cover_progress *p);
int cover_checkpoint_extra(const char *runtime_json, const cover_progress *p, char *out, size_t cap);
