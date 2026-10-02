#include "json.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>

static const json *shared;
static pthread_barrier_t barrier;
static atomic_int path_failures;
static void *path_worker(void *arg) {
    size_t id = (size_t)arg;
    const char *paths[] = {"a.2.b", "a.2.c", "deep.branch.longer.0.expected", "n", "a.1", "s"};
    const json *want[] = {json_get(json_at(json_get(shared, "a"), 2), "b"),
        json_get(json_at(json_get(shared, "a"), 2), "c"),
        json_get(json_at(json_get(json_get(json_get(shared, "deep"), "branch"), "longer"), 0), "expected"),
        json_get(shared, "n"), json_at(json_get(shared, "a"), 1), json_get(shared, "s")};
    pthread_barrier_wait(&barrier);
    for (size_t i = 0; i < 20000; i++) {
        size_t k = (id + i) % 6;
        if (json_path(shared, paths[k]) != want[k]) atomic_fetch_add(&path_failures, 1);
    }
    return nullptr;
}

int main(void) {
    const char *t = "{\"a\": [1, 2.5, {\"b\": \"x\\ny\", \"c\": true}], \"n\": null, \"s\": \"q\\\"z\", \"neg\": -3e2, \"deep\":{\"branch\":{\"longer\":[{\"expected\":72}]}}}";
    json *j = json_parse(t, strlen(t));
    assert(j && j->type == J_OBJ && j->n == 5);
    assert(json_num(json_path(j, "a.1"), 0) == 2.5);
    assert(!strcmp(json_str(json_path(j, "a.2.b"), ""), "x\ny"));
    assert(json_num(json_path(j, "a.2.c"), 0) == 1);
    assert(json_path(j, "n")->type == J_NULL);
    assert(!strcmp(json_str(json_get(j, "s"), ""), "q\"z"));
    assert(json_num(json_get(j, "neg"), 0) == -300);
    assert(json_path(j, "a.9") == nullptr && json_path(j, "zz") == nullptr);
    shared = j;
    pthread_t threads[8];
    assert(pthread_barrier_init(&barrier, nullptr, 8) == 0);
    for (size_t i = 0; i < 8; i++) assert(pthread_create(&threads[i], nullptr, path_worker, (void *)i) == 0);
    for (size_t i = 0; i < 8; i++) assert(pthread_join(threads[i], nullptr) == 0);
    pthread_barrier_destroy(&barrier);
    assert(atomic_load(&path_failures) == 0);
    json_free(j);
    assert(json_parse("{\"a\":}", 6) == nullptr);
    assert(json_parse("[1,2", 4) == nullptr);
    json *e = json_parse("  {}  ", 6);
    assert(e && e->n == 0);
    json_free(e);
    puts("json ok");
    return 0;
}
