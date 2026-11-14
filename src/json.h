/* Minimal JSON reader: parses a document into a node tree. Enough for zarr.json / OME metadata. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } json_type;

typedef struct json {
    json_type type;
    double num;                 /* J_NUM, J_BOOL (0/1) */
    char *str;                  /* J_STR (owned) */
    struct json **items;        /* J_ARR / J_OBJ children */
    char **keys;                /* J_OBJ keys (owned) */
    size_t n;                   /* child count */
    unsigned char *got;         /* J_OBJ: per key, read through json_get (json_unread: keys nothing read, e.g. typos) */
} json;

json *json_parse(const char *text, size_t len);   /* nullptr on error (json_error_offset: the byte where it failed) */
size_t json_error_offset(void);   /* the byte offset of the last json_parse failure on this thread */
/* every object key of j (recursively) that no json_get has returned, as "a.b.key" / "a.3.key" paths; keys named "note" / "comment" or
   starting with '_' are free text. cb per key; returns the count */
int json_unread(const json *j, const char *path, void (*cb)(const char *path, void *u), void *u);
void json_mark_read(const json *o);   /* marks every key of the object o read (callers that iterate keys themselves) */
void json_free(json *j);
const json *json_get(const json *obj, const char *key);      /* nullptr if absent / not object */
const json *json_at(const json *arr, size_t i);              /* nullptr if out of range / not array */
const json *json_path(const json *j, const char *path);      /* "a.b.0.c" style, nullptr if absent */
double json_num(const json *j, double dflt);
const char *json_str(const json *j, const char *dflt);
/* fprintf for JSON fragments: printf's "nan" / "-nan" / "inf" / "-inf" outside strings are written as null (json_parse refuses them since
   mini/json 5; reg inkstat's empty rows broke test_reg_render 11-14). Each call's format must keep its quotes balanced. Returns bytes or -1 */
int json_fprintf(FILE *f, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
