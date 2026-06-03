#include "tensor.h"
#include "kernels/kernels.h"
#include "loader.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)
static int approx(float a, float b, float tol) { return fabsf(a - b) <= tol; }

static void test_arena(void) {
    fte_arena a;
    CHECK(fte_arena_init(&a, 1 << 20) == 0);
    float *p = fte_arena_alloc(&a, 100);
    CHECK(p);
    CHECK(((uintptr_t)p % 64) == 0);
    float *q = fte_arena_alloc(&a, 10);
    CHECK(((uintptr_t)q % 64) == 0);
    CHECK(q >= p + 100);
    fte_arena_reset(&a);
    float *r = fte_arena_alloc(&a, 5);
    CHECK(r == p);
    fte_arena_free(&a);
}

static void test_layernorm(void) {
    float x[6] = {1, 2, 3, 4, 5, 6}; /* [2,3] */
    float g[3] = {1, 1, 1}, b[3] = {0, 0, 0};
    fte_layernorm(x, g, b, 2, 3, 1e-12f);
    CHECK(approx(x[0], -1.224745f, 1e-4f));
    CHECK(approx(x[1], 0.0f, 1e-4f));
    CHECK(approx(x[2], 1.224745f, 1e-4f));
}

static void test_fastgelu(void) {
    float x[1] = {1.0f}, bias[1] = {0.0f};
    fte_fastgelu(x, bias, 1, 1);
    CHECK(approx(x[0], 0.8411920f, 1e-5f));
}

static void test_softmax(void) {
    float x[4] = {1, 1, 1, 1};
    fte_softmax_rows(x, 1, 4, 2);
    CHECK(approx(x[0], 0.5f, 1e-6f));
    CHECK(approx(x[1], 0.5f, 1e-6f));
    CHECK(x[2] == 0.0f && x[3] == 0.0f);
}

static void test_matmul(void) {
    float A[6] = {1, 2, 3, 4, 5, 6}, B[6] = {1, 2, 3, 4, 5, 6}, C[4];
    fte_matmul(A, B, C, 2, 3, 2);
    CHECK(approx(C[0], 22, 1e-4f));
    CHECK(approx(C[1], 28, 1e-4f));
    CHECK(approx(C[2], 49, 1e-4f));
    CHECK(approx(C[3], 64, 1e-4f));
}

static void test_loader(void) {
    fte_weights w;
    if (fte_weights_open("model.fte", &w) != 0) { printf("SKIP loader (no model.fte)\n"); return; }
    CHECK(w.hdr->hidden == 384 && w.hdr->layers == 12);
    CHECK(fte_weight(&w, "embeddings.word_embeddings.weight") != NULL);
    CHECK(fte_weight(&w, "does.not.exist") == NULL);
    fte_weights_close(&w);
}

int main(void) {
    test_arena();
    test_layernorm();
    test_fastgelu();
    test_softmax();
    test_matmul();
    test_loader();
    printf(fails ? "%d FAIL\n" : "OK\n", fails);
    return fails ? 1 : 0;
}
