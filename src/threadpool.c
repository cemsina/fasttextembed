#include "threadpool.h"

#ifdef FTE_NO_THREADS
/* Single-threaded build (e.g. WebAssembly): no pthreads, everything runs on the caller. */
fte_pool *fte_pool_create(int n) { (void)n; return 0; }
void fte_pool_destroy(fte_pool *p) { (void)p; }
int  fte_pool_threads(const fte_pool *p) { (void)p; return 1; }
void fte_pool_begin(fte_pool *p) { (void)p; }
void fte_pool_end(fte_pool *p) { (void)p; }
void fte_pool_parallel_for(fte_pool *p, int n, void (*body)(void *, int, int), void *arg) {
    (void)p;
    if (n > 0) body(arg, 0, n);
}
#else

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

#if defined(__aarch64__) || defined(__arm__)
#define CPU_RELAX() __asm__ __volatile__("yield")
#elif defined(__x86_64__) || defined(__i386__)
#define CPU_RELAX() __asm__ __volatile__("pause")
#else
#define CPU_RELAX() ((void)0)
#endif

/* Spin-then-block fork/join pool. Inside a document, dispatches are µs apart, so workers
 * that just finished a chunk are still in their spin window and pick up the next dispatch
 * with ~ns latency (no syscall). If the gap exceeds the spin window (idle between docs, or
 * a long serial section), they block on a condvar instead of busy-spinning — which is what
 * made an unbounded spin pool oversubscribe cores and blow up tail latency. */

#define SPIN_LIMIT 20000   /* ~tens of µs of bounded spinning before blocking */

struct fte_pool {
    pthread_t *threads;
    int nworkers;                  /* spawned; total parallelism = nworkers+1 */

    pthread_mutex_t mu;
    pthread_cond_t cv_work, cv_done;
    int nblocked;                  /* workers currently blocked on cv_work (under mu) */
    int main_waiting;              /* main blocked on cv_done (under mu) */

    void (*body)(void *, int, int);
    void *arg;
    int nchunks, csz, n;

    atomic_ulong epoch;            /* bumped per dispatch */
    atomic_int next;               /* next chunk index (work-stealing) */
    atomic_int pending;            /* workers still draining */
    atomic_int engaged;            /* inside begin/end: workers spin, never block */
    atomic_int stop;
};

/* Work-stealing drain: dynamic chunk grab balances load across asymmetric cores,
 * so a slow (efficiency-core / preempted) thread doesn't stall the barrier. */
static void drain(fte_pool *p) {
    int c;
    while ((c = atomic_fetch_add_explicit(&p->next, 1, memory_order_relaxed)) < p->nchunks) {
        int s = c * p->csz;
        int e = s + p->csz;
        if (e > p->n) e = p->n;
        p->body(p->arg, s, e);
    }
}

static void set_qos(void) {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0); /* bias to P-cores */
#endif
}

static void *worker(void *v) {
    fte_pool *p = v;
    set_qos();
    unsigned long last = 0;
    for (;;) {
        int spins = 0;
        for (;;) {                 /* acquire next dispatch: spin, then block */
            if (atomic_load_explicit(&p->stop, memory_order_acquire)) return NULL;
            if (atomic_load_explicit(&p->epoch, memory_order_acquire) != last) break;
            /* engaged (inside a ParallelSection): stay hot, never block — the next
             * loop's dispatch arrives within µs, so we catch it without a syscall. */
            if (atomic_load_explicit(&p->engaged, memory_order_acquire)) { CPU_RELAX(); continue; }
            if (++spins < SPIN_LIMIT) { CPU_RELAX(); continue; }
            pthread_mutex_lock(&p->mu);
            while (!atomic_load(&p->stop) && !atomic_load(&p->engaged) &&
                   atomic_load_explicit(&p->epoch, memory_order_acquire) == last) {
                p->nblocked++;
                pthread_cond_wait(&p->cv_work, &p->mu);
                p->nblocked--;
            }
            pthread_mutex_unlock(&p->mu);
            spins = 0;
        }
        last = atomic_load_explicit(&p->epoch, memory_order_acquire);

        drain(p);

        if (atomic_fetch_sub_explicit(&p->pending, 1, memory_order_acq_rel) == 1) {
            pthread_mutex_lock(&p->mu);
            if (p->main_waiting) pthread_cond_signal(&p->cv_done);
            pthread_mutex_unlock(&p->mu);
        }
    }
}

fte_pool *fte_pool_create(int nthreads) {
    if (nthreads <= 1) return NULL;
    fte_pool *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->nworkers = nthreads - 1;
    p->threads = malloc((size_t)p->nworkers * sizeof(pthread_t));
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv_work, NULL);
    pthread_cond_init(&p->cv_done, NULL);
    atomic_init(&p->epoch, 0);
    atomic_init(&p->pending, 0);
    atomic_init(&p->engaged, 0);
    atomic_init(&p->stop, 0);
    atomic_init(&p->next, 0);
    set_qos();                     /* the caller participates too */
    int spawned = 0;
    for (int i = 0; i < p->nworkers; i++)
        if (pthread_create(&p->threads[spawned], NULL, worker, p) == 0) spawned++;
    p->nworkers = spawned;
    return p;
}

void fte_pool_destroy(fte_pool *p) {
    if (!p) return;
    pthread_mutex_lock(&p->mu);
    atomic_store(&p->stop, 1);
    pthread_cond_broadcast(&p->cv_work);
    pthread_mutex_unlock(&p->mu);
    for (int i = 0; i < p->nworkers; i++) pthread_join(p->threads[i], NULL);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv_work);
    pthread_cond_destroy(&p->cv_done);
    free(p->threads);
    free(p);
}

int  fte_pool_threads(const fte_pool *p) { return p ? p->nworkers + 1 : 1; }

/* ParallelSection: keep workers hot (spinning, not blocking) across the many small
 * dependent loops of one forward pass, so per-loop dispatch costs ~0. Safe because the
 * pool is sized to the performance cores and threads carry P-core QoS — no oversubscription. */
void fte_pool_begin(fte_pool *p) {
    if (!p || p->nworkers == 0) return;
    pthread_mutex_lock(&p->mu);
    atomic_store_explicit(&p->engaged, 1, memory_order_release);
    pthread_cond_broadcast(&p->cv_work);   /* wake any blocked workers into the spin */
    pthread_mutex_unlock(&p->mu);
}

void fte_pool_end(fte_pool *p) {
    if (!p || p->nworkers == 0) return;
    atomic_store_explicit(&p->engaged, 0, memory_order_release);
}

void fte_pool_parallel_for(fte_pool *p, int n,
                           void (*body)(void *, int, int), void *arg) {
    if (n <= 0) return;
    if (!p || p->nworkers == 0) { body(arg, 0, n); return; }
    int total = p->nworkers + 1;
    /* ~3 chunks per thread: enough granularity for work-stealing to balance
     * stragglers, few enough to keep the shared counter cheap. */
    int nchunks = total * 3;
    if (nchunks > n) nchunks = n;
    int csz = (n + nchunks - 1) / nchunks;
    if (csz < 1) csz = 1;
    p->body = body; p->arg = arg; p->n = n; p->csz = csz;
    p->nchunks = (n + csz - 1) / csz;
    atomic_store_explicit(&p->next, 0, memory_order_relaxed);
    atomic_store_explicit(&p->pending, p->nworkers, memory_order_relaxed);

    /* publish dispatch; wake any blocked workers (none if all still spinning) */
    pthread_mutex_lock(&p->mu);
    atomic_fetch_add_explicit(&p->epoch, 1, memory_order_release);
    if (p->nblocked > 0) pthread_cond_broadcast(&p->cv_work);
    pthread_mutex_unlock(&p->mu);

    drain(p);   /* caller helps */

    /* wait for workers: spin, then block */
    int spins = 0;
    while (atomic_load_explicit(&p->pending, memory_order_acquire) > 0) {
        if (++spins < SPIN_LIMIT) { CPU_RELAX(); continue; }
        pthread_mutex_lock(&p->mu);
        p->main_waiting = 1;
        while (atomic_load_explicit(&p->pending, memory_order_acquire) > 0)
            pthread_cond_wait(&p->cv_done, &p->mu);
        p->main_waiting = 0;
        pthread_mutex_unlock(&p->mu);
        break;
    }
}

#endif /* FTE_NO_THREADS */
