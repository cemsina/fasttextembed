#ifndef FTE_TOKENIZER_H
#define FTE_TOKENIZER_H

typedef struct fte_tokenizer fte_tokenizer;

int  fte_tokenizer_load(const char *vocab_tsv, fte_tokenizer **out); /* 0 on success */
void fte_tokenizer_free(fte_tokenizer *t);

/* encode `text` into ids (incl [CLS]/[SEP]); writes up to max_len ids; returns count */
int  fte_tokenizer_encode(const fte_tokenizer *t, const char *text, int *ids_out, int max_len);

#endif
