/* ufsm ink-blend core (src/inkblend.c), also predict --seamless (src/seamless.c) */
#pragma once
#include <stdint.h>
int ink_blend_run(const char *out, double um, const double st[3], int n, const char *const *in, const double (*g)[3], const int64_t (*org)[3],
                  int nlev, int S, double q, int nt, const char *attrs);
