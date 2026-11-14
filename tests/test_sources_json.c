/* sources-json validation (mini/json, 2026-11-14): required keys, value types, unknown encodings, unknown / unused keys (a warning,
   fatal under sources_set_strict), duplicate keys, non-JSON numbers, an empty sources array; and the parser's number syntax. CPU only. */
#include "json.h"
#include "sources.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails;
static void check(const char *name, int ok) { printf("sources-json %-58s %s\n", name, ok ? "ok" : "FAIL"); fails += !ok; }
static sources *load(const char *dir, const char *body) {
    char p[1200]; snprintf(p, sizeof p, "%s/s.json", dir); FILE *f = fopen(p, "w"); if (!f) abort(); fputs(body, f); fclose(f);
    return sources_load(p);
}
#define SRC(extra) "{\"sources\":[{\"name\":\"a\",\"root\":\"%s\",\"ct\":\"ct\",\"um\":1" extra "}]}"
static int loads(const char *dir, const char *fmt) { char b[4096]; snprintf(b, sizeof b, fmt, dir); sources *S = load(dir, b); if (S) sources_free(S); return S != nullptr; }
int main(void) {
    char tmpl[] = "/tmp/ufsm-srcjson-XXXXXX"; const char *dir = mkdtemp(tmpl); if (!dir) return 1;
    { const char *ok[] = {"0", "-0.5e+3", "12.25", "1E2"}, *bad[] = {"nan", "inf", "0x10", "1e", "01", ".5", "-", "+1"};
      int a = 1; for (int i = 0; i < 4; i++) { json *j = json_parse(ok[i], strlen(ok[i])); a &= j != nullptr; json_free(j); }
      int b = 1; for (int i = 0; i < 8; i++) { json *j = json_parse(bad[i], strlen(bad[i])); b &= j == nullptr; json_free(j); }
      check("JSON numbers parse", a); check("nan / inf / hex / malformed numbers are errors", b);
      const char *d = "{\"a\":1,\"a\":2}"; json *j = json_parse(d, strlen(d)); check("duplicate keys are errors", !j); json_free(j); }
    check("a valid source loads", loads(dir, SRC("")));
    check("a missing root is fatal", !loads(dir, "{\"sources\":[{\"name\":\"a\",\"ct\":\"ct\",\"um\":1}]}"));
    check("a missing ct is fatal", !loads(dir, "{\"sources\":[{\"name\":\"a\",\"root\":\"%s\",\"um\":1}]}"));
    check("\"weight\": \"2\" (a string) is fatal", !loads(dir, SRC(",\"weight\":\"2\"")));
    check("a negative weight is fatal", !loads(dir, SRC(",\"weight\":-1")));
    check("\"weight\": nan is fatal", !loads(dir, SRC(",\"weight\":nan")));
    check("an unknown encoding is fatal", !loads(dir, SRC(",\"targets\":{\"recto\":{\"group\":\"labels\",\"encoding\":\"Binary\"}}")));
    check("a known encoding loads", loads(dir, SRC(",\"targets\":{\"recto\":{\"group\":\"labels\",\"encoding\":\"binary\"}}")));
    check("an empty sources array is fatal", !loads(dir, "{\"sources\":[]}"));
    check("a syntax error is fatal", !loads(dir, "{\"sources\":[{\"name\":\"a\",}]}"));
    check("an unknown key warns (loads)", loads(dir, SRC(",\"wieght\":2,\"note\":\"free text\"")));
    check("an unknown target name warns (loads)", loads(dir, SRC(",\"targets\":{\"rekto\":\"labels\"}")));
    sources_set_strict(1);
    check("an unknown key is fatal under strict", !loads(dir, SRC(",\"wieght\":2")));
    check("an unknown target name is fatal under strict", !loads(dir, SRC(",\"targets\":{\"rekto\":\"labels\"}")));
    check("note / _comment keys stay free under strict", loads(dir, SRC(",\"note\":\"x\",\"_why\":1")));
    sources_set_strict(0);
    char p[1200]; snprintf(p, sizeof p, "%s/s.json", dir); unlink(p); rmdir(dir);
    printf("test_sources_json: %s\n", fails ? "FAIL" : "ok");
    return fails != 0;
}
