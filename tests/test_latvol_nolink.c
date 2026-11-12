/* test_latvol_nolink DIR KEY: a binary linked WITHOUT src/latvol.c (zarr3.o reaches the codec through weak references) must refuse a
   latvol array at z3_open with the "built without" error, not call a null function. Exit 0 when it does. */
#include "store.h"
#include "zarr3.h"
#include <stdio.h>
#include <string.h>
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: test_latvol_nolink DIR ARRAY-KEY\n"); return 2; }
    store *s = store_open(argv[1]); if (!s) { printf("FAIL: cannot open the store %s\n", argv[1]); return 1; }
    z3 *z = z3_open(s, argv[2], nullptr);
    if (z) { printf("FAIL: a latvol array opened in a binary without the codec\n"); return 1; }
    if (!strstr(z3_error(), "built without src/latvol.c")) { printf("FAIL: the open failed with another error: %s\n", z3_error()); return 1; }
    printf("test_latvol_nolink: ok (%s)\n", z3_error());
    return 0;
}
