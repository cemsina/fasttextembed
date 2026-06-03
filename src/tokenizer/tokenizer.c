#include "tokenizer.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* BERT WordPiece tokenizer (uncased). Targets ASCII + Latin punctuation for the
 * golden corpus; full unicode NFD accent-stripping is deferred to a later phase. */

#define HSIZE (1 << 16)
typedef struct node { char *tok; int id; struct node *next; } node;
struct fte_tokenizer { node *buckets[HSIZE]; int cls, sep, unk; };

static unsigned hash(const char *s) {
    unsigned h = 2166136261u;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 16777619u; }
    return h & (HSIZE - 1);
}
static void put(fte_tokenizer *t, const char *s, int id) {
    unsigned b = hash(s);
    node *n = malloc(sizeof(node));
    n->tok = strdup(s);
    n->id = id;
    n->next = t->buckets[b];
    t->buckets[b] = n;
}
static int get(const fte_tokenizer *t, const char *s) {
    for (node *n = t->buckets[hash(s)]; n; n = n->next)
        if (!strcmp(n->tok, s)) return n->id;
    return -1;
}

int fte_tokenizer_load(const char *path, fte_tokenizer **out) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    fte_tokenizer *t = calloc(1, sizeof(*t));
    char line[256];
    while (fgets(line, sizeof line, f)) {
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = 0;
        int id = atoi(tab + 1);
        put(t, line, id);
    }
    fclose(f);
    t->cls = get(t, "[CLS]");
    t->sep = get(t, "[SEP]");
    t->unk = get(t, "[UNK]");
    *out = t;
    return 0;
}

void fte_tokenizer_free(fte_tokenizer *t) {
    if (!t) return;
    for (int i = 0; i < HSIZE; i++) {
        node *n = t->buckets[i];
        while (n) { node *x = n->next; free(n->tok); free(n); n = x; }
    }
    free(t);
}

/* wordpiece a single normalized word (already lowercased, no spaces) into ids */
static int wordpiece(const fte_tokenizer *t, const char *w, int *ids, int cap) {
    int len = (int)strlen(w), start = 0, cnt = 0;
    char buf[300];
    if (len > 100) { if (cap > 0) ids[0] = t->unk; return cap > 0 ? 1 : 0; }
    while (start < len) {
        int end = len, id = -1;
        while (start < end) {
            int k = 0;
            if (start > 0) { buf[k++] = '#'; buf[k++] = '#'; }
            memcpy(buf + k, w + start, (size_t)(end - start));
            buf[k + end - start] = 0;
            id = get(t, buf);
            if (id >= 0) break;
            end--;
        }
        if (id < 0) { if (cnt < cap) ids[cnt++] = t->unk; return cnt; } /* whole word -> UNK */
        if (cnt < cap) ids[cnt++] = id;
        start = end;
    }
    return cnt;
}

int fte_tokenizer_encode(const fte_tokenizer *t, const char *text, int *ids, int max_len) {
    int n = 0;
    if (n < max_len) ids[n++] = t->cls;
    char word[256];
    int wl = 0;
    for (const char *p = text;; p++) {
        unsigned char c = (unsigned char)*p;
        int punct = c && !isalnum(c) && !isspace(c);
        if (c == 0 || isspace(c) || punct) {
            if (wl > 0) {
                word[wl] = 0;
                int room = max_len - 1 - n; /* reserve 1 for [SEP] */
                if (room > 0) n += wordpiece(t, word, ids + n, room);
                wl = 0;
            }
            if (punct) {
                char s[2] = {(char)tolower(c), 0};
                int id = get(t, s);
                if (id < 0) id = t->unk;
                if (n < max_len - 1) ids[n++] = id;
            }
            if (c == 0) break;
        } else if (wl < 255) {
            word[wl++] = (char)tolower(c);
        }
    }
    if (n < max_len) ids[n++] = t->sep;
    return n;
}
