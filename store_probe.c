/* store_probe: does a cache that every store passes through stall the stores of k streams written in lockstep
 * once more of their lines fall into one of its sets than it has ways; if so, its set period and ways.
 *
 *   store_probe decide                          the answer: "none", or "structure line L sets N ways W"
 *   store_probe grid S1,S2,... KMIN KMAX        every (stride, k) point listed, k unbounded (diagnostic)
 *
 * The pattern. k streams, stream p writing the S bytes from p * G on, one 16-byte store per visit, the k streams
 * in order within a visit (a radix pass's R outputs), S / 16 visits a sweep (S a power of two). Three layouts of
 * one point (S, k):
 *   conflict  G = S: the k lines of a visit lie S apart, so a structure whose set period P divides S gets all k in
 *             one set, one of period P = S * 2^a gets ceil(k / 2^a) in its fullest set;
 *   free_a    G = S + line, free_b  G = S + 3 * line: an odd count of lines (S >= 2 lines), so the k lines of a
 *             visit fall into k distinct sets of every power-of-two structure of k sets or more, whatever its
 *             period.
 * Every layout writes S bytes per stream with the same instructions, in a region of k S bytes plus the free
 * layouts' gaps of one or three lines between streams; they differ in which lines one visit writes together. The
 * loop holds no load, so no load can alias a store's address bits.
 *
 * The bounds. Strides S run over the powers of two from 2 lines to a page: a line apart, any layout puts k
 * distinct lines in one set of a one-set (fully associative) structure, which no layout avoids; past a page,
 * virtual addresses no longer choose the bits a physically indexed structure sets by. At stride S the probe tries
 * k up to K(S) = (l1d_ways - 1) * max(1, min(l1d_span, page) / S): then the conflict layout's region (k S bytes)
 * fits the first-level data cache with one way of every set left for the process's other lines (its stack, the
 * clock's data), and no visit puts more lines into one of its sets than its ways; so a write-back first-level cache
 * serves every store of the conflict layout, and its set conflicts and misses are never read as a structure (the
 * free layouts' gaps can only slow them). A region filling every way of every set is slower in the conflict layout
 * on some cores (within the last streams of a full first level), which is the first level, not a structure past it. Where a
 * way of the first-level cache spans more than a page, the address bits past the page that pick its set are the
 * physical page's, which no layout chooses, so the bound counts the sets one page spans. A structure of fewer bytes
 * than the first-level cache is in reach (its ways + 1 <= K(P)).
 *
 * The decision at a point. Each layout is timed n times, the three rotating their order every round. The point
 * overflows when the Mann-Whitney count U of (conflict, free) sample pairs with the conflict one slower reaches the
 * critical count (the least U that equally likely orderings of the 3n samples reach with probability at most
 * family_false_rate over the most points the probe may test, test_for), and the conflict layout's median sample is
 * slower than the free layouts' by at least the stores of one line per visit (see the size of an effect below).
 *
 * The decision of the probe. At each stride: does k = K(S) overflow; if so, the least k that does, by bisection
 * (overflow is monotone in k: one more stream never empties a set). The ways W are the least knee minus one, the
 * period P the least stride whose knee is that; every stride must then show the knee the model predicts,
 * W * max(1, P / S) + 1, or none where that exceeds K(S); otherwise the answer is none (the stalls are not those
 * of one set-associative structure). A stride where K(S) exceeds W * P / line lines is not checked: there the free
 * layouts overflow the structure too. The line is the first-level data cache's (the coherence line). */
#if defined(__linux__)
#define _GNU_SOURCE
#endif
#include <math.h>

#include "probe_os.h"

/* A policy: at most one probe in a thousand on a machine whose store path has no set-associative structure
 * reports one through a chance ordering of its samples. */
static const double family_false_rate = 1e-3;
/* The size of an effect that counts. A program that avoids overflowing a line by reordering its passes pays at
 * least the stores that write that line once more: line / 16 stores of 16 bytes. So a point overflows only where its
 * conflict layout is slower than the free ones by at least that per visit, a ratio of 1 + (line / 16) / k for a
 * visit of k stores (one overflowed line at the knee); a smaller effect, however certain, is one a change of passes
 * costs more to avoid. A sample spans 4 k_most clock
 * granularities, so that the clock's quantization moves the ratio of two medians by at most 2 / (4 k_most), half
 * the least such threshold. */

enum { conflict, free_a, free_b, layouts };

struct machine {
  size_t line, l1d_ways, l1d_sets, l1d_span, page;
  int ways_reported; /* 0: l1d_ways is size / page, the least of a cache indexed within the page */
  double granularity; /* seconds: the least nonzero step between two clock reads */
};

/* The first-level data cache of the CPU this runs on, the page and the clock's granularity. Where the OS reports no
 * ways, the bound takes the least ways of a cache indexed within the page, size / page, and a span of one page: that
 * keeps the conflict layout within the first level if the cache is so indexed (the usual virtually indexed,
 * physically tagged design), and is said in the output. */
static int query_machine(struct machine *m) {
  memset(m, 0, sizeof *m);
  struct cache caches[most_caches];
  const struct cache *l1d = find_cache(caches, os_caches(caches), 1, 'D');
  m->page = page_size();
  if (!l1d || l1d->line == 0 || l1d->size == 0) return 0;
  m->line = l1d->line;
  m->ways_reported = l1d->ways != 0;
  m->l1d_ways = l1d->ways ? l1d->ways : (l1d->size + m->page - 1) / m->page;
  m->l1d_sets = l1d->size / (m->l1d_ways * m->line);
  m->l1d_span = m->line * m->l1d_sets;
  m->granularity = clock_granularity();
  return 1;
}

/* reps sweeps of k streams G apart, each S bytes of 16-byte stores in lockstep. */
static NOINLINE void sweep(unsigned char *base, size_t G, unsigned k, size_t S, unsigned reps) {
  for (unsigned r = 0; r < reps; ++r) {
    v16 value = V16_SPLAT(r);
    for (size_t off = 0; off < S; off += 16) {
      unsigned char *q = base + off;
      for (unsigned p = 0; p < k; ++p, q += G) V16_STORE(q, value);
      COMPILER_BARRIER(); /* keeps the visits in order: no interchange of the two loops */
    }
  }
}

/* One sweep untimed, so that the lines of the layout timed before are gone from the sample, then reps timed. */
static double time_sweeps(unsigned char *base, size_t G, unsigned k, size_t S, unsigned reps) {
  sweep(base, G, k, S, 1);
  double t0 = now();
  sweep(base, G, k, S, reps);
  return now() - t0;
}

struct point {
  size_t stride;
  unsigned k, reps, n, critical, pairs_slower;
  double least;                 /* seconds a sample lasts at least */
  double t[layouts][64];        /* seconds per sample, n of each */
  double conflict_median, free_median, need; /* need: the least conflict / free median ratio that counts */
  int overflows;
};

static size_t layout_distance(const struct machine *m, size_t S, int layout) {
  return layout == conflict ? S : layout == free_a ? S + m->line : S + 3 * m->line;
}

static size_t region_bytes(const struct machine *m, size_t S, unsigned k) { return k * layout_distance(m, S, free_b); }

static double minimum(const double *x, unsigned n) {
  double v = x[0];
  for (unsigned i = 1; i < n; ++i) v = x[i] < v ? x[i] : v;
  return v;
}
static double maximum(const double *x, unsigned n) {
  double v = x[0];
  for (unsigned i = 1; i < n; ++i) v = x[i] > v ? x[i] : v;
  return v;
}

static int ascending(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static double median(const double *x, unsigned n) {
  double sorted[128];
  memcpy(sorted, x, n * sizeof *x);
  qsort(sorted, n, sizeof *sorted, ascending);
  return n % 2 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}

static void measure(const struct machine *m, unsigned char *base, struct point *pt) {
  size_t G[layouts];
  for (int l = 0; l < layouts; ++l) G[l] = layout_distance(m, pt->stride, l);
  pt->reps = 1;
  while (time_sweeps(base, G[free_a], pt->k, pt->stride, pt->reps) < pt->least) pt->reps *= 2;
  for (unsigned r = 0; r < pt->n; ++r)
    for (int j = 0; j < layouts; ++j) {
      int l = (int)((r + (unsigned)j) % layouts);
      pt->t[l][r] = time_sweeps(base, G[l], pt->k, pt->stride, pt->reps);
    }
  pt->pairs_slower = 0;
  for (unsigned i = 0; i < pt->n; ++i)
    for (unsigned j = 0; j < pt->n; ++j)
      pt->pairs_slower += (pt->t[conflict][i] > pt->t[free_a][j]) + (pt->t[conflict][i] > pt->t[free_b][j]);
  double both_free[128];
  memcpy(both_free, pt->t[free_a], pt->n * sizeof(double));
  memcpy(both_free + pt->n, pt->t[free_b], pt->n * sizeof(double));
  pt->conflict_median = median(pt->t[conflict], pt->n);
  pt->free_median = median(both_free, 2 * pt->n);
  pt->need = 1 + (double)(m->line / 16) / pt->k;
  pt->overflows = pt->pairs_slower >= pt->critical && pt->conflict_median >= pt->free_median * pt->need;
}

static void print_point(const struct point *pt) {
  double c = minimum(pt->t[conflict], pt->n), a = minimum(pt->t[free_a], pt->n), b = minimum(pt->t[free_b], pt->n);
  double free = a < b ? a : b, stores = (double)pt->reps * pt->k * (double)(pt->stride / 16);
  printf("point S %5zu k %3u reps %5u n %u | ns/store min conflict %.4f free_a %.4f free_b %.4f | max conflict "
         "%.4f free_a %.4f free_b %.4f | min conflict/free %.3f b/a %.3f | median conflict/free %.4f need %.4f | U %u of "
         "%u, critical %u | %s\n",
         pt->stride, pt->k, pt->reps, pt->n, 1e9 * c / stores, 1e9 * a / stores, 1e9 * b / stores,
         1e9 * maximum(pt->t[conflict], pt->n) / stores, 1e9 * maximum(pt->t[free_a], pt->n) / stores,
         1e9 * maximum(pt->t[free_b], pt->n) / stores, c / free, b / a, pt->conflict_median / pt->free_median,
         pt->need, pt->pairs_slower, 2 * pt->n * pt->n,
         pt->critical, pt->overflows ? "OVERFLOW" : "-");
  fflush(stdout);
}

/* The fraction of the orderings of n conflict and m free samples, all equally likely, whose count of
 * (conflict, free) pairs with the conflict sample slower is at least u (the Mann-Whitney statistic's tail). */
static double tail_fraction(unsigned n, unsigned m, unsigned u) {
  const unsigned most = n * m + 1;
  double *ways = calloc((size_t)2 * (m + 1) * most, sizeof *ways); /* ways[i % 2][j][v]: i conflict, j free, U = v */
  if (!ways) return 1;
#define WAYS(i, j, v) ways[(((size_t)((i) % 2) * (m + 1)) + (j)) * most + (v)]
  for (unsigned i = 0; i <= n; ++i)
    for (unsigned j = 0; j <= m; ++j)
      for (unsigned v = 0; v < most; ++v) {
        /* the slowest of i + j samples is a conflict one, slower than all j free ones, or a free one */
        double w = (i == 0 || j == 0) ? (v == 0) : 0;
        if (i > 0 && j > 0) w = (v >= j ? WAYS(i - 1, j, v - j) : 0) + WAYS(i, j - 1, v);
        WAYS(i, j, v) = w;
      }
  double all = 0, at_least = 0;
  for (unsigned v = 0; v < most; ++v) {
    all += WAYS(n, m, v);
    if (v >= u) at_least += WAYS(n, m, v);
  }
#undef WAYS
  free(ways);
  return at_least / all;
}

/* A point's test: n samples of each layout; it overflows when at least `critical` of the (conflict, free)
 * pairs have the conflict sample slower. `critical` is the least count reached by chance with probability at most
 * family_false_rate / points; n is the least that keeps the test passing with one disturbed sample in each group
 * (a conflict sample faster than every free one and a free sample slower than every conflict one, as an interrupt
 * makes), so that a preemption now and then does not hide a structure. */
struct test {
  unsigned n, critical;
};

static struct test test_for(double points) {
  const double rate = family_false_rate / points;
  for (unsigned n = 2; n <= 64 / 2; ++n) {
    unsigned m = 2 * n, critical = n * m + 1;
    while (critical > 0 && tail_fraction(n, m, critical - 1) <= rate) --critical;
    if (critical <= n * m - (n + m - 1)) return (struct test){n, critical};
  }
  return (struct test){32, 32 * 64};
}

static unsigned ceil_log2(unsigned x) {
  unsigned b = 0;
  while ((1u << b) < x) ++b;
  return b;
}

static unsigned k_bound(const struct machine *m, size_t S) {
  size_t chosen = m->l1d_span < m->page ? m->l1d_span : m->page; /* the span virtual addresses choose sets in */
  size_t share = chosen > S ? chosen / S : 1;
  return (unsigned)((m->l1d_ways - 1) * share); /* one way of every set left for the process's other lines */
}

/* Base pages only: every page alike for every layout. */
static unsigned char *base_pages(size_t bytes) {
  int huge;
  return map(bytes, 0, &huge);
}

static void print_machine(const struct machine *m) {
  printf("machine line %zu l1d %zu ways x %zu sets (span %zu)%s page %zu clock granularity %.1f ns\n", m->line,
         m->l1d_ways, m->l1d_sets, m->l1d_span,
         m->ways_reported ? "" : " (ways not reported: the bound takes size / page)", m->page, 1e9 * m->granularity);
}

static int grid(const struct machine *m, char *strides, unsigned kmin, unsigned kmax) {
  size_t list[32];
  unsigned count = 0;
  for (char *s = strtok(strides, ","); s && count < 32; s = strtok(NULL, ",")) list[count++] = strtoul(s, NULL, 10);
  size_t bytes = 0;
  for (unsigned i = 0; i < count; ++i) {
    size_t f = region_bytes(m, list[i], kmax);
    bytes = f > bytes ? f : bytes;
  }
  struct test test = test_for((double)count * (kmax - kmin + 1));
  const double least = 4.0 * kmax * m->granularity;
  unsigned char *base = base_pages(bytes);
  if (!base) return 1;
  for (unsigned i = 0; i < count; ++i)
    for (unsigned k = kmin; k <= kmax; ++k) {
      struct point pt = {.stride = list[i], .k = k, .n = test.n, .critical = test.critical, .least = least};
      measure(m, base, &pt);
      print_point(&pt);
    }
  return 0;
}

static int decide(const struct machine *m) {
  double t0 = now();
  size_t strides[32] = {0};
  unsigned count = 0, knee[32];
  double at_bound[32]; /* median conflict / free time at k = K(S) */
  double worst_points = 0;
  size_t bytes = 0;
  for (size_t S = 2 * m->line; S <= m->page && count < 32; S *= 2) {
    strides[count++] = S;
    worst_points += 1 + ceil_log2(k_bound(m, S) - 1);
    size_t f = region_bytes(m, S, k_bound(m, S));
    bytes = f > bytes ? f : bytes;
  }
  struct test test = test_for(worst_points);
  const double sample = 4.0 * k_bound(m, strides[0]) * m->granularity;
  printf("decide strides %zu..%zu, %u samples a layout per point, overflow at U >= %u of %u (at most %.0f points)\n",
         strides[0], strides[count - 1], test.n, test.critical, 2 * test.n * test.n, worst_points);
  unsigned char *base = base_pages(bytes);
  if (!base) return 1;
  unsigned points = 0;
  for (unsigned i = 0; i < count; ++i) {
    unsigned hi = k_bound(m, strides[i]), lo = 1;
    struct point pt = {.stride = strides[i], .k = hi, .n = test.n, .critical = test.critical, .least = sample};
    measure(m, base, &pt);
    print_point(&pt);
    ++points;
    knee[i] = 0;
    at_bound[i] = pt.conflict_median / pt.free_median;
    if (!pt.overflows) continue;
    while (hi - lo > 1) { /* invariant: lo does not overflow, hi does */
      struct point mid = {.stride = strides[i], .k = (lo + hi) / 2, .n = test.n, .critical = test.critical,
                          .least = sample};
      measure(m, base, &mid);
      print_point(&mid);
      ++points;
      if (mid.overflows) hi = mid.k; else lo = mid.k;
    }
    knee[i] = hi;
  }
  unsigned least = 0;
  for (unsigned i = 0; i < count; ++i)
    if (knee[i] && (!least || knee[i] < least)) least = knee[i];
  size_t period = 0;
  for (unsigned i = 0; i < count && least; ++i)
    if (knee[i] == least) { period = strides[i]; break; }
  int consistent = 1;
  for (unsigned i = 0; i < count && least; ++i) {
    size_t share = period > strides[i] ? period / strides[i] : 1;
    unsigned expected = (unsigned)((least - 1) * share + 1);
    unsigned want = expected <= k_bound(m, strides[i]) ? expected : 0;
    /* At k = K(S) the fullest set holds ceil(K / share) lines, those past the ways overflow each visit: the cost of
     * one overflowed line, in stores of the free layouts, is the extra time of a visit of K stores over them. */
    unsigned K = k_bound(m, strides[i]), fullest = (K + (unsigned)share - 1) / (unsigned)share;
    double over = fullest > least - 1 ? fullest - (least - 1) : 0;
    /* Past ways * sets lines a visit, the free layouts fill every set past its ways too, so the ratio at K(S)
     * no longer compares a conflict with none, and the stride tests nothing. */
    int clean = K <= (least - 1) * (period / m->line);
    printf("knee S %5zu: %u (model %u%s), at k %u: conflict/free %.3f, %.0f lines over, %.1f stores per line\n",
           strides[i], knee[i], want, clean ? "" : ", not checked: free layouts overflow too", K, at_bound[i], over,
           over > 0 ? (at_bound[i] - 1) * K / over : 0.0);
    consistent &= !clean || knee[i] == want;
  }
  double wall = now() - t0;
  if (!least)
    printf("answer none\n");
  else if (!consistent)
    printf("answer none (knees inconsistent with one structure of %u ways, period %zu)\n", least - 1, period);
  else
    printf("answer structure line %zu sets %zu ways %u\n", m->line, period / m->line, least - 1);
  printf("cost wall %.4f s, %u points, %zu bytes mapped, peak rss %ld KiB\n", wall, points, bytes, peak_rss_kib());
  return 0;
}

int main(int argc, char **argv) {
  struct machine m;
  setvbuf(stdout, NULL, _IOLBF, 1 << 12);
  place();
  if (!query_machine(&m)) {
    fprintf(stderr, "store_probe: the OS reports no first-level data cache\n");
    return 1;
  }
  print_machine(&m);
  if (argc >= 2 && strcmp(argv[1], "decide") == 0) return decide(&m);
  if (argc >= 5 && strcmp(argv[1], "grid") == 0) return grid(&m, argv[2], atoi(argv[3]), atoi(argv[4]));
  fprintf(stderr, "usage: store_probe decide | store_probe grid S1,S2,... KMIN KMAX\n");
  return 2;
}
