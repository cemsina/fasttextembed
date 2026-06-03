#ifndef FTE_THREADPOOL_H
#define FTE_THREADPOOL_H

/* Minimal fork/join thread pool for intra-document parallelism (low single-query
 * latency). Between documents the workers park on a condvar (no CPU use). Wrapping a
 * document in fte_pool_begin/fte_pool_end keeps them spinning, so each matmul's
 * parallel_for dispatches via a cheap atomic (no per-call wakeup). parallel_for splits
 * [0,n) into coarse chunks consumed dynamically; bodies write disjoint output, so
 * there are no reductions/races and results are deterministic. */
typedef struct fte_pool fte_pool;

fte_pool *fte_pool_create(int nthreads);   /* nthreads total incl. caller; NULL if <=1 */
void      fte_pool_destroy(fte_pool *p);
int       fte_pool_threads(const fte_pool *p);

void      fte_pool_begin(fte_pool *p);     /* enter active (spinning) region */
void      fte_pool_end(fte_pool *p);       /* leave active region (workers park) */

/* Run body(arg, start, end) over a partition of [0,n). Blocks until done.
 * If p is NULL, n<=1, or called outside begin/end, runs serially on the caller. */
void      fte_pool_parallel_for(fte_pool *p, int n,
                                void (*body)(void *arg, int start, int end), void *arg);

#endif
