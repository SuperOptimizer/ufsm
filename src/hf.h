/* Listing helpers for the upstream trees: the HF bucket tree API (paginated via Link headers), the
   S3 ListObjectsV2 API, and local directories. Callback receives each entry's path (relative to the
   listed directory) and size; return nonzero to stop. */
#pragma once
#include "store.h"
#include <stdint.h>

typedef int (*list_cb)(const char *name, int is_dir, int64_t size, void *ud);

/* HF: root like "https://huggingface.co/buckets/scrollprize/datasets/resolve" (the same store used for
   downloads; the API URL is derived), key = directory path. */
int hf_list(store *s, const char *key, list_cb cb, void *ud);
/* S3 anonymous bucket: root like "https://vesuvius-challenge-open-data.s3.amazonaws.com", prefix = dir. */
int s3_list(store *s, const char *prefix, list_cb cb, void *ud);
/* Local directory. */
int dir_list(const char *dir, list_cb cb, void *ud);
/* Dispatch on the store root. */
int store_list(store *s, const char *key, list_cb cb, void *ud);
/* Read the token file (~/huggingfacetoken by default); returns malloc'd string or nullptr. */
char *hf_token(const char *path);
