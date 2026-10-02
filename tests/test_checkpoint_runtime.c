/* Checkpoint inference settings must survive a move without any sidecar files. */
#include "checkpoint.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int fails;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); fails++; } } while (0)
static void header(const char *path, const char *extra) {
    FILE *f = fopen(path, "wb");
    fprintf(f, "UFSM{\"nlev\":4,\"extra\":%s}\n", extra); fclose(f);
}
int main(void) {
    char path[128], text[8192]; snprintf(path, sizeof path, "/tmp/ufsm_runtime_%d", (int)getpid());
    checkpoint_runtime a = {.version = 1, .train_window = 512, .prec = 1, .f16 = 1, .act_mx4 = 1, .grad_mx8 = 1, .input_mx = 1, .input_prec = 4, .gn_stored = 1}, b;
    strcpy(a.policy, "all=fp4:fp4:fp4,enc0.c1=fp16"); strcpy(a.optimizer, "muon");
    CHECK(checkpoint_runtime_json(&a, text, sizeof text) == 0);
    header(path, text); CHECK(checkpoint_runtime_read(path, &b) == 1); CHECK(memcmp(&a, &b, sizeof a) == 0);
    a.input_prec = 8; CHECK(checkpoint_runtime_json(&a, text, sizeof text) == 0);
    header(path, text); CHECK(checkpoint_runtime_read(path, &b) == 1); CHECK(memcmp(&a, &b, sizeof a) == 0);
    strcpy(a.policy, "quote\" and slash\\"); CHECK(checkpoint_runtime_json(&a, text, sizeof text) == 0);
    header(path, text); CHECK(checkpoint_runtime_read(path, &b) == 1); CHECK(!strcmp(a.policy, b.policy));
    CHECK(checkpoint_runtime_json(&a, text, 8) < 0);
    a.prec = 9; CHECK(checkpoint_runtime_json(&a, text, sizeof text) < 0);
    header(path, "{}"); CHECK(checkpoint_runtime_read(path, &b) == 0);
    header(path, "{\"runtime\":{\"version\":99}}"); CHECK(checkpoint_runtime_read(path, &b) < 0);
    header(path, "{\"runtime\":null}"); CHECK(checkpoint_runtime_read(path, &b) < 0);
    header(path, "{\"runtime\":{\"version\":1,\"train_window\":512,\"prec\":1,\"f16\":1,\"act_mx4\":1,\"act_mx8\":0,\"grad_mx8\":1,\"policy\":\"\",\"optimizer\":\"muon\"}}");
    CHECK(checkpoint_runtime_read(path, &b) == 1); CHECK(b.input_mx == 1 && b.input_prec == 4 && b.gn_stored == 0);
    a.prec = 1; a.input_mx = 2; CHECK(checkpoint_runtime_json(&a, text, sizeof text) < 0);
    a.input_mx = 1; a.input_prec = 6; CHECK(checkpoint_runtime_json(&a, text, sizeof text) < 0);
    a.input_prec = 0; CHECK(checkpoint_runtime_json(&a, text, sizeof text) < 0);
    a.input_prec = 8; a.gn_stored = 2; CHECK(checkpoint_runtime_json(&a, text, sizeof text) < 0);
    FILE *f = fopen(path, "wb"); fputs("UFSM{\"extra\":{", f); fclose(f); CHECK(checkpoint_runtime_read(path, &b) < 0);
    unlink(path); printf("checkpoint runtime: %s\n", fails ? "FAIL" : "ok"); return fails != 0;
}
