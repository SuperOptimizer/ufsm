#include "json.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    const char *t = "{\"a\": [1, 2.5, {\"b\": \"x\\ny\", \"c\": true}], \"n\": null, \"s\": \"q\\\"z\", \"neg\": -3e2}";
    json *j = json_parse(t, strlen(t));
    assert(j && j->type == J_OBJ && j->n == 4);
    assert(json_num(json_path(j, "a.1"), 0) == 2.5);
    assert(!strcmp(json_str(json_path(j, "a.2.b"), ""), "x\ny"));
    assert(json_num(json_path(j, "a.2.c"), 0) == 1);
    assert(json_path(j, "n")->type == J_NULL);
    assert(!strcmp(json_str(json_get(j, "s"), ""), "q\"z"));
    assert(json_num(json_get(j, "neg"), 0) == -300);
    assert(json_path(j, "a.9") == nullptr && json_path(j, "zz") == nullptr);
    json_free(j);
    assert(json_parse("{\"a\":}", 6) == nullptr);
    assert(json_parse("[1,2", 4) == nullptr);
    json *e = json_parse("  {}  ", 6);
    assert(e && e->n == 0);
    json_free(e);
    puts("json ok");
    return 0;
}
