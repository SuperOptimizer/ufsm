#include "cover.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void write_file(const char *p, const char *s) { FILE *f = fopen(p, "w"); assert(f); assert(fputs(s, f) >= 0); assert(!fclose(f)); }
int main(void) {
    char path[] = "/tmp/ufsm-cover-test-XXXXXX"; int fd = mkstemp(path); assert(fd >= 0); close(fd);
    source s = {.name = "fixture"}; sources S = {.n = 1, .src = &s};
    const char *valid = "{\"version\":1,\"P\":16,\"level\":0,\"count\":2,\"source_names\":[\"fixture\"],\"tiles\":[[0,0,0,0],[0,16,0,0]]}";
    write_file(path, valid); cover_plan *plan = cover_load(path, &S, 16); assert(plan && plan->count == 2 && strlen(plan->sha256) == 64);
    cover_progress p = {.count = 2, .cursor = 1, .base_step = 100}; strcpy(p.sha256, plan->sha256);
    char extra[1024], header[2048]; assert(!cover_checkpoint_extra("{\"runtime\":{}}", &p, extra, sizeof extra));
    snprintf(header, sizeof header, "UFSM{\"step\":101,\"extra\":%s}\n", extra); write_file(path, header);
    cover_progress got; assert(cover_checkpoint_read(path, &got) == 1 && got.cursor == 1 && got.base_step == 100 && !strcmp(got.sha256, p.sha256));
    snprintf(header, sizeof header, "UFSM{\"step\":102,\"extra\":%s}\n", extra); write_file(path, header); assert(cover_checkpoint_read(path, &got) < 0);
    write_file(path, "UFSM{\"step\":100,\"extra\":{}}\n"); assert(!cover_checkpoint_read(path, &got));
    p.cursor = 3; assert(cover_checkpoint_extra("{}", &p, extra, sizeof extra) < 0);
    const char *extended = "{\"version\":1,\"P\":16,\"level\":0,\"count\":3,\"source_names\":[\"fixture\"],\"tiles\":[[0,0,0,0],[0,16,0,0],[0,32,0,0]]}";
    write_file(path, extended); cover_plan *next = cover_load(path, &S, 16); assert(next);
    p.cursor = p.count;
    assert(!cover_validate_extension(plan, next, &p));
    p.cursor--; assert(!cover_validate_extension(plan, next, &p)); p.cursor++;
    p.cursor=p.count+1; assert(cover_validate_extension(plan,next,&p)); p.cursor=p.count;
    p.sha256[0] = p.sha256[0] == 'a' ? 'b' : 'a'; assert(cover_validate_extension(plan, next, &p)); strcpy(p.sha256, plan->sha256);
    assert(cover_validate_extension(plan, plan, &p));
    next->tiles[0][1]++; assert(cover_validate_extension(plan, next, &p)); next->tiles[0][1]--;
    p.base_step = 2147483646; assert(cover_validate_extension(plan, next, &p));
    cover_free(next); cover_free(plan);
    write_file(path, "{\"version\":1,\"P\":16,\"level\":0,\"count\":2,\"source_names\":[\"fixture\"],\"tiles\":[[0,0,0,0],[0,0,0,0]]}"); assert(!cover_load(path, &S, 16));
    write_file(path, valid); assert(!cover_load(path, &S, 32));
    s.name = "other"; assert(!cover_load(path, &S, 16));
    unlink(path); puts("cover identity, uniqueness, committed checkpoint cursor: ok"); return 0;
}
