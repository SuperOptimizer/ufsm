#pragma once
/* defaulted.h (2026-11-14, ops audit #10): a fallback that changes a result says so. count-field's direction fell back to +1 when the
   prior found no rings, without a word, and a whole unroll ran on it. Every such fallback prints one "DEFAULTED: <what> = <value>
   (<why>)" line on stderr; test_pipeline_cli asserts its golden run prints none. UFSM_STRICT_DEFAULTS=1 makes it fatal (exit 3). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define UFSM_DEFAULTED(...) do { fputs("DEFAULTED: ", stderr); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); \
    const char *e_ = getenv("UFSM_STRICT_DEFAULTS"); if (e_ && *e_ && strcmp(e_, "0")) { fputs("UFSM_STRICT_DEFAULTS: stopping\n", stderr); exit(3); } } while (0)
