/* Minimal JSON reader: parses a document into a node tree. Enough for zarr.json / OME metadata. */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } json_type;

typedef struct json {
    json_type type;
    double num;                 /* J_NUM, J_BOOL (0/1) */
    char *str;                  /* J_STR (owned) */
    struct json **items;        /* J_ARR / J_OBJ children */
    char **keys;                /* J_OBJ keys (owned) */
    size_t n;                   /* child count */
} json;

json *json_parse(const char *text, size_t len);   /* nullptr on error */
void json_free(json *j);
const json *json_get(const json *obj, const char *key);      /* nullptr if absent / not object */
const json *json_at(const json *arr, size_t i);              /* nullptr if out of range / not array */
const json *json_path(const json *j, const char *path);      /* "a.b.0.c" style, nullptr if absent */
double json_num(const json *j, double dflt);
const char *json_str(const json *j, const char *dflt);
