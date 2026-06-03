#include "tokenizer/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

static void test_basic(const fte_tokenizer *t) {
    int ids[512];
    int n = fte_tokenizer_encode(t, "hello world", ids, 512);
    /* [CLS] hello world [SEP] = 101 7592 2088 102 */
    CHECK(n == 4);
    CHECK(ids[0] == 101);
    CHECK(ids[1] == 7592);
    CHECK(ids[2] == 2088);
    CHECK(ids[3] == 102);
}

/* golden_tokens.txt lines: "text<TAB>id1,id2,..." */
static void test_golden(const fte_tokenizer *t) {
    FILE *f = fopen("tests/data/golden_tokens.txt", "r");
    if (!f) { printf("SKIP golden tokens (no file)\n"); return; }
    char line[8192];
    int row = 0;
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\n")] = 0;
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = 0;
        const char *text = line;
        int expect[512], en = 0;
        for (char *p = strtok(tab + 1, ","); p; p = strtok(NULL, ",")) expect[en++] = atoi(p);
        int got[512];
        int gn = fte_tokenizer_encode(t, text, got, 512);
        if (gn != en) { printf("FAIL row %d: len %d != %d (\"%s\")\n", row, gn, en, text); fails++; }
        else
            for (int i = 0; i < en; i++)
                if (got[i] != expect[i]) {
                    printf("FAIL row %d pos %d: %d != %d (\"%s\")\n", row, i, got[i], expect[i], text);
                    fails++;
                    break;
                }
        row++;
    }
    fclose(f);
    printf("checked %d golden token rows\n", row);
}

int main(void) {
    fte_tokenizer *t;
    if (fte_tokenizer_load("vocab.tsv", &t) != 0) { printf("SKIP (no vocab.tsv)\n"); return 0; }
    test_basic(t);
    test_golden(t);
    fte_tokenizer_free(t);
    printf(fails ? "%d FAIL\n" : "OK\n", fails);
    return fails ? 1 : 0;
}
