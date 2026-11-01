/* crowd_probe: what one visit of a pass pays when it stores more whole lines into one set of the first-level data
 * cache than the set has ways: the least crowding that costs (the knee), and the extra time per crowded line, in
 * nanoseconds and in instructions of a reference loop shaped like an FFT pass.
 *
 *   crowd_probe fmas                  the fused multiply-adds one source iteration of reference_loop issues
 *   crowd_probe run REF_INSNS         the probe; REF_INSNS = instructions one source iteration of reference_loop
 *                                     retires, counted in this binary's disassembly (loop_insns.py); 0 = time only
 *
 * The pattern. k streams G bytes apart, each G bytes long. A visit stores one whole line into each stream (line / 16
 * stores of 16 bytes), the k streams in order; the next visit stores the next line of every stream; G / line visits
 * make a sweep. Three layouts of one k:
 *   conflict  G = D, the span of one way of the first-level data cache (size / ways): the k lines of a visit lie D
 *             apart and share one set;
 *   free_a    G = D + line, free_b  G = D + 3 lines: an odd count of lines apart, so the k lines of a visit fall into
 *             k distinct sets while k <= sets.
 * In every layout each stream covers D / line consecutive lines, one per set, so a sweep stores exactly k lines into
 * every set: the layouts have the same footprint per set, the same instructions and the same bytes, and differ only
 * in how many of one visit's lines share a set. At k <= ways all three fit the first level; past it a sweep misses it
 * alike in all three (k lines cycled through each set of `ways`), and the conflict layout alone asks for more lines
 * of one set at once than the set holds. conflict / free is that crowding's cost.
 *
 * Where the OS reports no ways, D is taken as the page: a first-level cache indexed by address bits within the page
 * (the usual virtually indexed, physically tagged design) has a way span of at most a page, so lines a page apart
 * share a set. Where the span exceeds the page, the set bits past the page are the physical page's; the probe asks
 * Linux for transparent huge pages so they follow the virtual ones, and says where the system did not grant them
 * (the conflict layout then spreads over span / page sets at random).
 *
 * The range. k from 2 to size / D + 36, the reported ways (or the least ways of a cache indexed within the page)
 * plus 36: through the ways, the absorption past them that line fill buffers give, and far enough past both to read
 * a per-line slope.
 *
 * A point. Each layout is timed n times, the three rotating their order every round, with one sample of the
 * reference loop per round. The point is crowded when the Mann-Whitney count U of (conflict, free) sample pairs with
 * the conflict one slower reaches the critical count (chance reaches it with probability at most family_false_rate
 * over every point of the probe) and the conflict median is slower than the free median by at least one line of
 * stores per visit: ratio >= 1 + 1 / k (an effect smaller than storing one of the visit's lines again is one no
 * reordering of a pass could avoid for less). The knee is the least k from which every larger k is crowded (one more
 * stream never empties a set).
 *
 * The price. For k at or past the knee, the extra time per visit (conflict median - free median) / visits, divided by
 * the visit's crowded lines k - W (W the reported ways, else knee - 1), and the least-squares slope of the extra time
 * per visit over k (the marginal price of one more crowded line, which needs no W). Each in nanoseconds and in
 * reference instructions: the reference loop's median nanoseconds per instruction of the same point's rounds.
 *
 * The reference loop: a radix-2 butterfly over interleaved complex single-precision vectors, out of place, all
 * arrays within the first level: per vector two unit-stride data loads, two twiddle loads, one shuffle (swap of real
 * and imaginary parts), one multiply, one fused multiply-add, one add, one subtract and two unit-stride stores. Its
 * instructions per iteration are counted from the compiled binary, never assumed.
 *
 * The noise floor of a point: median free_b / free_a, two layouts that crowd nothing. */
#if defined(__linux__)
#define _GNU_SOURCE
#endif
#include <math.h>

#include "probe_os.h"

/* A policy: at most one probe in a thousand on a machine whose stores do not mind crowding reports a knee through a
 * chance ordering of its samples. */
static const double family_false_rate = 1e-3;
/* A policy: a sample lasts at least this many clock granularities, so that quantization moves a median by at most
 * 0.1 % (two reads per sample). */
static const double sample_granularities = 2000;
/* The probe's reach past the ways (see the range). */
static const unsigned reach_past_ways = 36;

enum { conflict, free_a, free_b, layouts };

struct machine {
  size_t line, l1d_size, l1d_ways, page, span; /* ways 0: not reported; span: D */
  int huge;                                     /* 1: the arena is on huge pages */
  double granularity;                           /* seconds: the least nonzero step between two clock reads */
};

/* ---- the reference loop ---- */

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define REFERENCE_ISA "x86-64 AVX + FMA3, 8 floats a vector"
enum { reference_floats = 8, reference_fmas = 1 };
#if defined(_MSC_VER) && !defined(__clang__)
static int reference_supported(void) {
  int r[4];
  __cpuid(r, 1);
  int fma = r[2] >> 12 & 1, osxsave = r[2] >> 27 & 1, avx = r[2] >> 28 & 1;
  return fma && osxsave && avx && (_xgetbv(0) & 6) == 6;
}
#define REFERENCE_TARGET
#else
static int reference_supported(void) { return __builtin_cpu_supports("avx") && __builtin_cpu_supports("fma"); }
#define REFERENCE_TARGET __attribute__((target("avx,fma")))
#endif
/* Exported, so that its symbol survives in every object format's disassembly. */
REFERENCE_TARGET NOINLINE void reference_loop(float *y, const float *x, const float *twiddle, size_t n) {
#if defined(__clang__)
#pragma clang loop unroll(disable) vectorize(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
  for (size_t i = 0; i < n; ++i) {
    __m256 a = _mm256_load_ps(x + 8 * i), b = _mm256_load_ps(x + 8 * (n + i));
    __m256 re = _mm256_load_ps(twiddle + 8 * i), im = _mm256_load_ps(twiddle + 8 * (n + i));
    __m256 t = _mm256_fmadd_ps(_mm256_permute_ps(b, 0xB1), im, _mm256_mul_ps(b, re));
    _mm256_store_ps(y + 8 * i, _mm256_add_ps(a, t));
    _mm256_store_ps(y + 8 * (n + i), _mm256_sub_ps(a, t));
  }
}
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define REFERENCE_ISA "AArch64 Advanced SIMD, 4 floats a vector"
enum { reference_floats = 4, reference_fmas = 1 };
static int reference_supported(void) { return 1; }
NOINLINE void reference_loop(float *y, const float *x, const float *twiddle, size_t n) {
#if defined(__clang__)
#pragma clang loop unroll(disable) vectorize(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
  for (size_t i = 0; i < n; ++i) {
    float32x4_t a = vld1q_f32(x + 4 * i), b = vld1q_f32(x + 4 * (n + i));
    float32x4_t re = vld1q_f32(twiddle + 4 * i), im = vld1q_f32(twiddle + 4 * (n + i));
    float32x4_t t = vfmaq_f32(vmulq_f32(b, re), vrev64q_f32(b), im);
    vst1q_f32(y + 4 * i, vaddq_f32(a, t));
    vst1q_f32(y + 4 * (n + i), vsubq_f32(a, t));
  }
}
#else
/* Other ISAs: one complex value a step in scalar code, whose two products contract to fused multiply-adds. */
#define REFERENCE_ISA "scalar, 2 floats a step"
enum { reference_floats = 2, reference_fmas = 2 };
static int reference_supported(void) { return 1; }
NOINLINE void reference_loop(float *y, const float *x, const float *twiddle, size_t n) {
#if defined(__clang__)
#pragma clang loop unroll(disable) vectorize(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
  for (size_t i = 0; i < n; ++i) {
    float ar = x[2 * i], ai = x[2 * i + 1], br = x[2 * (n + i)], bi = x[2 * (n + i) + 1];
    float wr = twiddle[2 * i], wi = twiddle[2 * i + 1];
    float tr = br * wr - bi * wi, ti = br * wi + bi * wr;
    y[2 * i] = ar + tr, y[2 * i + 1] = ai + ti;
    y[2 * (n + i)] = ar - tr, y[2 * (n + i) + 1] = ai - ti;
  }
}
#endif

/* The reference's arrays: n vectors in each half of x, twiddle and y, all within the first level. */
struct reference {
  float *x, *twiddle, *y;
  size_t n;
  unsigned calls;
  double insns; /* instructions one source iteration retires (from the disassembly); 0: unknown */
};

/* One sample of `calls` calls: seconds. */
static double time_reference(const struct reference *r) {
  double t0 = now();
  for (unsigned c = 0; c < r->calls; ++c) {
    reference_loop(r->y, r->x, r->twiddle, r->n);
    COMPILER_BARRIER();
  }
  return now() - t0;
}

/* Seconds per instruction of a sample, or per iteration where the count is unknown. */
static double per_instruction(const struct reference *r, double seconds) {
  return seconds / ((double)r->calls * (double)r->n * (r->insns > 0 ? r->insns : 1));
}

/* ---- the crowding pattern ---- */

/* reps sweeps of k streams G apart, each `lines` lines long, one whole line of each stream per visit. */
static NOINLINE void sweep(unsigned char *base, size_t G, unsigned k, size_t lines, size_t line, unsigned reps) {
  for (unsigned r = 0; r < reps; ++r) {
    v16 value = V16_SPLAT(r);
    for (size_t u = 0; u < lines; ++u) {
      unsigned char *q = base + u * line;
      for (unsigned p = 0; p < k; ++p, q += G)
        for (size_t b = 0; b < line; b += 16) V16_STORE(q + b, value);
      COMPILER_BARRIER(); /* keeps the visits in order: no interchange of the loops */
    }
  }
}

static size_t distance(const struct machine *m, int layout) {
  return m->span + (layout == conflict ? 0 : layout == free_a ? m->line : 3 * m->line);
}

/* One sweep untimed, so that the layout timed before leaves no lines behind, then reps timed: seconds. */
static double time_layout(const struct machine *m, unsigned char *base, int layout, unsigned k, unsigned reps) {
  size_t G = distance(m, layout), lines = m->span / m->line;
  sweep(base, G, k, lines, m->line, 1);
  double t0 = now();
  sweep(base, G, k, lines, m->line, reps);
  return now() - t0;
}

static int ascending(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static double median(const double *x, unsigned n) {
  double sorted[256];
  memcpy(sorted, x, n * sizeof *x);
  qsort(sorted, n, sizeof *sorted, ascending);
  return n % 2 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
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

/* n samples a layout and the critical count: the least count chance reaches with probability at most
 * family_false_rate / points; n the least that keeps a real effect crowded with one disturbed sample in each group
 * (a conflict sample faster than every free one and a free sample slower than every conflict one). */
struct test {
  unsigned n, critical;
};

static struct test test_for(double points) {
  const double rate = family_false_rate / points;
  for (unsigned n = 2; n <= 32; ++n) {
    unsigned m = 2 * n, critical = n * m + 1;
    while (critical > 0 && tail_fraction(n, m, critical - 1) <= rate) --critical;
    if (critical <= n * m - (n + m - 1)) {
      struct test t = {n, critical};
      return t;
    }
  }
  struct test t = {32, 32 * 64};
  return t;
}

struct point {
  unsigned k, reps, pairs_slower, crowded;
  double t[layouts][32], ref[32]; /* seconds per sample; seconds per reference instruction */
  double conflict, free, floor, ns_per_insn; /* medians: ns per visit; free_b / free_a; ns per instruction */
};

static void measure(const struct machine *m, unsigned char *base, const struct reference *ref, const struct test *test,
                    double least, struct point *pt) {
  pt->reps = 1;
  while (time_layout(m, base, free_a, pt->k, pt->reps) < least) pt->reps *= 2;
  for (unsigned r = 0; r < test->n; ++r) {
    for (int j = 0; j < layouts; ++j) {
      int l = (int)((r + (unsigned)j) % layouts);
      pt->t[l][r] = time_layout(m, base, l, pt->k, pt->reps);
    }
    pt->ref[r] = per_instruction(ref, time_reference(ref));
  }
  pt->pairs_slower = 0;
  for (unsigned i = 0; i < test->n; ++i)
    for (unsigned j = 0; j < test->n; ++j)
      pt->pairs_slower += (pt->t[conflict][i] > pt->t[free_a][j]) + (pt->t[conflict][i] > pt->t[free_b][j]);
  double both[64];
  memcpy(both, pt->t[free_a], test->n * sizeof(double));
  memcpy(both + test->n, pt->t[free_b], test->n * sizeof(double));
  double visits = (double)pt->reps * (double)(m->span / m->line);
  pt->conflict = 1e9 * median(pt->t[conflict], test->n) / visits;
  pt->free = 1e9 * median(both, 2 * test->n) / visits;
  pt->floor = median(pt->t[free_b], test->n) / median(pt->t[free_a], test->n);
  pt->ns_per_insn = 1e9 * median(pt->ref, test->n);
  pt->crowded = pt->pairs_slower >= test->critical && pt->conflict >= pt->free * (1 + 1.0 / pt->k);
}

static int run(double ref_insns) {
  double t0 = now();
  struct machine m = {0};
  struct cache caches[most_caches];
  place();
  const struct cache *l1d = find_cache(caches, os_caches(caches), 1, 'D');
  if (l1d) m.line = l1d->line, m.l1d_size = l1d->size, m.l1d_ways = l1d->ways;
  m.page = page_size();
  if (m.line == 0 || m.l1d_size == 0) {
    fprintf(stderr, "crowd_probe: the OS reports no first-level data cache size or line\n");
    return 1;
  }
  m.span = m.l1d_ways ? m.l1d_size / m.l1d_ways : m.page;
  const unsigned implied = (unsigned)(m.l1d_size / m.span), kmax = implied + reach_past_ways;
  const size_t bytes = (size_t)kmax * distance(&m, free_b);
  unsigned char *base = map(bytes, 1, &m.huge);
  m.granularity = clock_granularity();
  if (!base) return 1;
  printf("machine line %zu l1d %zu B ways %zu%s, span D %zu B (%s), page %zu, arena %zu B on %s pages, clock "
         "granularity %.1f ns\n",
         m.line, m.l1d_size, m.l1d_ways, m.l1d_ways ? "" : " (not reported)", m.span,
         m.l1d_ways ? "size / ways" : "the page: ways not reported", m.page, bytes, m.huge ? "huge" : "base",
         1e9 * m.granularity);
  if (m.span > m.page && !m.huge)
    printf("warning: span %zu B > page %zu B on base pages: the conflict layout's sets follow physical pages\n", m.span,
           m.page);

  /* The reference's arrays: 64-byte aligned, n vectors per half of x, twiddle and y (3 * 2 * 64 vectors of at most
   * 32 bytes: 12 KiB, within every first level). */
  struct reference ref = {0};
  ref.n = 64;
  ref.insns = ref_insns;
  const size_t half = ref.n * reference_floats;
  unsigned char *raw = malloc(3 * 2 * half * sizeof(float) + 64);
  if (!raw) return 1;
  float *arrays = (float *)(((uintptr_t)raw + 63) & ~(uintptr_t)63);
  ref.x = arrays, ref.twiddle = arrays + 2 * half, ref.y = arrays + 4 * half;
  for (size_t i = 0; i < 2 * half; ++i) {
    ref.x[i] = (float)(i % 7) - 3; /* finite, never subnormal: out of place, the inputs stay as written */
    ref.twiddle[i] = (float)cos(0.01 * (double)i);
  }
  if (!reference_supported()) {
    printf("reference: %s not supported by this CPU; prices in ns only\n", REFERENCE_ISA);
    ref.insns = 0;
  }
  const double least = sample_granularities * m.granularity;
  for (ref.calls = 1; time_reference(&ref) < least;) ref.calls *= 2;
  printf("reference: %s, %zu iterations a call, %u calls a sample, %.3f instructions an iteration (%s)\n",
         REFERENCE_ISA, ref.n, ref.calls, ref.insns, ref.insns > 0 ? "from the disassembly" : "not given");

  const struct test test = test_for(kmax - 1);
  printf("range k 2..%u (%s ways %u + %u), %u samples a layout per point, crowded at U >= %u of %u and conflict/free "
         ">= 1 + 1/k, sample >= %.1f us\n",
         kmax, m.l1d_ways ? "reported" : "implied by size / page:", implied, reach_past_ways, test.n, test.critical,
         2 * test.n * test.n, 1e6 * least);
  static struct point points[256];
  unsigned count = 0;
  for (unsigned k = 2; k <= kmax && count < 256; ++k, ++count) {
    struct point *pt = &points[count];
    pt->k = k;
    measure(&m, base, &ref, &test, least, pt);
    printf("point k %3u reps %5u | ns/visit conflict %8.2f free %8.2f | conflict/free %.3f floor free_b/free_a %.3f "
           "| U %4u | ns/insn %.4f | %s\n",
           k, pt->reps, pt->conflict, pt->free, pt->conflict / pt->free, pt->floor, pt->pairs_slower,
           ref.insns > 0 ? pt->ns_per_insn : 0.0, pt->crowded ? "CROWDED" : "-");
    fflush(stdout);
  }

  unsigned knee = 0; /* the least k from which every larger k is crowded */
  for (unsigned i = count; i-- > 0 && points[i].crowded;) knee = points[i].k;
  double floor_most = 0, floors[256];
  for (unsigned i = 0; i < count; ++i) {
    floors[i] = fabs(points[i].floor - 1);
    floor_most = floors[i] > floor_most ? floors[i] : floor_most;
  }
  printf("noise floor |free_b/free_a - 1|: median %.4f, largest %.4f over %u points\n", median(floors, count),
         floor_most, count);
  if (!knee) {
    printf("answer knee none in k 2..%u\n", kmax);
  } else {
    const unsigned W = m.l1d_ways ? (unsigned)m.l1d_ways : knee - 1;
    double per_line_ns[256], per_line_insns[256], sx = 0, sy = 0, sxx = 0, sxy = 0, sz = 0;
    unsigned past = 0;
    for (unsigned i = 0; i < count; ++i) {
      const struct point *pt = &points[i];
      if (pt->k < knee) continue;
      double extra = pt->conflict - pt->free;
      if (pt->k > W) {
        per_line_ns[past] = extra / (pt->k - W);
        per_line_insns[past] = per_line_ns[past] / pt->ns_per_insn;
        ++past;
      }
      sx += pt->k, sy += extra, sxx += (double)pt->k * pt->k, sxy += pt->k * extra, sz += pt->ns_per_insn;
    }
    const unsigned fitted = kmax - knee + 1;
    double slope = fitted > 1 ? (fitted * sxy - sx * sy) / (fitted * sxx - sx * sx) : 0, ns_insn = sz / fitted;
    double low = per_line_ns[0], high = per_line_ns[0];
    for (unsigned i = 1; i < past; ++i) {
      low = per_line_ns[i] < low ? per_line_ns[i] : low;
      high = per_line_ns[i] > high ? per_line_ns[i] : high;
    }
    printf("answer knee %u (ways %u %s; crowded lines at the knee %d)\n", knee, W,
           m.l1d_ways ? "reported" : "taken as knee - 1", (int)knee - (int)W);
    if (past)
      printf("price per crowded line (extra / (k - %u), k %u..%u): median %.2f ns (%.2f .. %.2f)", W,
             knee > W ? knee : W + 1, kmax, median(per_line_ns, past), low, high);
    if (past && ref.insns > 0)
      printf(", median %.1f reference instructions", median(per_line_insns, past));
    if (past) printf("\n");
    printf("price slope d(extra ns per visit)/dk over k %u..%u: %.2f ns", knee, kmax, slope);
    if (ref.insns > 0) printf(", %.1f reference instructions (at %.4f ns per instruction)", slope / ns_insn, ns_insn);
    printf("\n");
  }
  printf("cost wall %.3f s, %u points, %zu bytes mapped, peak rss %ld KiB\n", now() - t0, count, bytes, peak_rss_kib());
  return 0;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IOLBF, 1 << 12);
  if (argc == 2 && strcmp(argv[1], "fmas") == 0) return printf("%d\n", reference_fmas) < 0;
  if (argc == 3 && strcmp(argv[1], "run") == 0) return run(atof(argv[2]));
  fprintf(stderr, "usage: crowd_probe fmas | crowd_probe run REF_INSNS\n");
  return 2;
}
