/* streamer_probe: from timing alone, how many pages of concurrent streams the second level's stream prefetcher
 * follows before the lines of a pass arrive on demand, and what a line on demand costs against a prefetched one:
 * {page, stores, paired, demand_factor}.
 *
 *   streamer_probe decide REF_STORES REF_PAIRED MOST_PAGES
 *   streamer_probe grid KIND STREAM_BYTES SAMPLE_BYTES ROUNDS K1,K2,...  (diagnostic: every K timed, no test)
 *
 * REF_STORES and REF_PAIRED are page counts the caller takes as within any prefetcher's reach: q is measured there
 * as each kind's reference, so a capacity at or below its reference cannot be told apart from every count losing
 * alike. MOST_PAGES is the most store pages one visit of the caller's passes reaches; a capacity above it changes
 * nothing the caller does. A run as published: decide 10 4 100.
 *
 * The pattern. A pass of K streams: stream k writes the B bytes from k B on, one whole line per visit (line / 16
 * stores of 16 bytes), the K streams in order within a visit, B / line visits. B = 2 pages + 1 line: each stream
 * crosses into new pages within a pass, as a pass's streams longer than a page do, and stream k lies k lines further
 * into its page than stream 0, so the streams cross pages at different visits and no two lines of a visit share an
 * in-page offset (no set of any cache indexed within the page takes two of them while K <= page / line). Streams
 * that share in-page offsets add set conflicts that are no page effect. Each visit's K lines lie in K distinct pages:
 * K is the pass's count of pages. Three kinds:
 *   stores   the K streams store only, to lines that lie past the last level;
 *   paired   each visit first loads the line at the same offset of K source bands (the source's slot, laid out as
 *            the destination's) that miss the second level and hit the last: K store pages and K load pages, the
 *            paired capacity (measured with as many load pages as store pages);
 *   memory   the same with the source past the last level too: if store and load pages from memory count together
 *            against the stores' capacity, its knee is at ceil(stores / 2) (a check of that model; it feeds no
 *            field).
 * Two layouts of one point, the same instructions over the same lines of the same slot:
 *   lockstep line (k, u) at k B + u line: K pages at once, each walked by one stream;
 *   linear   line u K + k: the slot written front to back, one page at once.
 * The loop is one 2-D walk whose two steps are its arguments, so the layouts differ only in the order the lines
 * are visited, i.e. in how many pages the streams run through at once.
 *
 * Where the lines lie. The destination cycles through past_llc_multiple (6, a policy) times the second and last levels
 * together, slot after slot, so a line is written a lap after its last touch and comes from memory. The memory kind
 * takes its source slot from the same cycle. The paired kind's source cycles through L2 + one slot of its own: between
 * two reads of a source line as many destination lines are written as source lines read, so twice the cycle passes
 * through the caches, at least twice the second level (missed) and at most the last level where 2 (L2 + slot) <= LLC
 * (kept under LRU); a machine where that does not hold answers no paired capacity. Memory is mapped with transparent
 * huge pages where the system grants them, so a lockstep pass's K pages cost no more translations than a linear one's.
 *
 * The decision at a point. n rounds; in each both layouts are timed (order alternating by round, one untimed sample of
 * each before the first round), a sample storing at least the second level's bytes in whole passes. A sample's
 * write-backs from the last level are of lines both layouts wrote, which pulls q toward 1 (a bias toward none); samples
 * past the last level would cost it tens of times over. q = lockstep time / linear time, a ratio within one round so a
 * slow drift of the clock frequency cancels. q at K is compared with q at the reference R of its kind (REF_STORES for
 * stores, REF_PAIRED for paired and memory), measured once: K is past the streamer when (a) the Mann-Whitney count of
 * (q_K, q_R) pairs with q_K larger reaches the critical count for a family false-positive rate of family_false_rate
 * over the most points the probe may test, and (b) median q_K / median q_R >= 3 / 2: demand_factor, the factor a line
 * past the streamer costs, is a whole number, so a loss that rounds to a factor of 1 is none.
 *
 * The decision of the probe, per kind: test K = 2 R, 4 R, ..., MOST_PAGES until one is past; if none is, there is
 * no capacity in range (none); else bisect between the last kept K and it for the least past K (one more page of
 * streams never restores what fewer lost). The capacity is that K; demand_factor is the rounded median of
 * median q_K / median q_R over the stores kind's past points.
 */
#if defined(__linux__)
#define _GNU_SOURCE
#endif
#include <math.h>

#include "probe_os.h"

/* A policy: at most one probe in a thousand on a machine whose prefetcher follows any number of pages reports a
 * capacity through a chance ordering of its samples. */
static const double family_false_rate = 1e-3;
/* A policy for a region that lies past the last level: six times the second and last levels together, so a line
 * written is a lap of the region from its last touch. */
static const size_t past_llc_multiple = 6;

enum kind { stores, paired, memory, kinds };
static const char *const kind_name[kinds] = {"stores", "paired", "memory"};
enum layout { lockstep, linear, layouts };

struct machine {
  size_t line, page, l2, llc;
  double granularity; /* seconds: the least nonzero step between two clock reads */
};

/* The line (the first-level data cache's), the second and the last level of the CPU this runs on, as the OS reports
 * them (the last level is the second where the OS reports no higher one), the page and the clock's granularity. */
static int query_machine(struct machine *m) {
  memset(m, 0, sizeof *m);
  struct cache caches[most_caches];
  unsigned count = os_caches(caches);
  const struct cache *l1d = find_cache(caches, count, 1, 'D'), *l2 = find_cache(caches, count, 2, 'D'), *llc = l2;
  for (unsigned level = 3; level < 8; ++level)
    if (find_cache(caches, count, level, 'U')) llc = find_cache(caches, count, level, 'U');
  m->page = page_size();
  if (!l1d || !l2 || l1d->line == 0 || l2->size == 0) return 0;
  m->line = l1d->line, m->l2 = l2->size, m->llc = llc->size;
  m->granularity = clock_granularity();
  return 1;
}

/* The two walks, one 2-D loop each: line (o, i) at o * outer + i * inner, `count` lines in order (o over
 * count / inner_n). */
static NOINLINE void store_walk(unsigned char *dst, size_t outer_n, size_t outer, size_t inner_n, size_t inner,
                                size_t line, v16 value) {
  for (size_t o = 0; o < outer_n; ++o) {
    unsigned char *row = dst + o * outer;
    for (size_t i = 0; i < inner_n; ++i) {
      unsigned char *at = row + i * inner;
      for (size_t b = 0; b < line; b += 16) V16_STORE(at + b, value);
    }
    COMPILER_BARRIER(); /* keeps the walk's order: no interchange of the loops */
  }
}

static NOINLINE void copy_walk(unsigned char *dst, const unsigned char *src, size_t outer_n, size_t outer,
                               size_t inner_n, size_t inner, size_t line, v16 add) {
  for (size_t o = 0; o < outer_n; ++o) {
    size_t row = o * outer;
    for (size_t i = 0; i < inner_n; ++i) {
      size_t at = row + i * inner;
      for (size_t b = 0; b < line; b += 16) V16_STORE(dst + at + b, V16_ADD(V16_LOAD(src + at + b), add));
    }
    COMPILER_BARRIER();
  }
}

struct arena {
  const struct machine *m;
  size_t stream;              /* B: bytes of one stream in one slot */
  unsigned char *dst, *src;   /* the past-LLC cycle; the paired kind's source cycle */
  int huge;                   /* both on huge pages (Linux, where granted) */
  size_t dst_bytes, src_bytes, dst_at, src_at;
  unsigned value;
};

static size_t slot_bytes(const struct arena *a, unsigned k) { return (size_t)k * a->stream; }

static unsigned char *next_slot(unsigned char *base, size_t bytes, size_t *at, size_t slot) {
  if (*at + slot > bytes) *at = 0;
  unsigned char *p = base + *at;
  *at += slot;
  return p;
}

/* One pass of K streams of `kind` over the next slot(s), in `layout`. */
static void pass(struct arena *a, enum kind kind, enum layout layout, unsigned k) {
  const size_t line = a->m->line, slot = slot_bytes(a, k), units = a->stream / line;
  const size_t outer = layout == lockstep ? line : (size_t)k * line, inner = layout == lockstep ? a->stream : line;
  unsigned char *d = next_slot(a->dst, a->dst_bytes, &a->dst_at, slot);
  v16 value = V16_SPLAT(++a->value);
  if (kind == stores) {
    store_walk(d, units, outer, k, inner, line, value);
    return;
  }
  const unsigned char *s = kind == memory ? next_slot(a->dst, a->dst_bytes, &a->dst_at, slot)
                                          : next_slot(a->src, a->src_bytes, &a->src_at, slot);
  copy_walk(d, s, units, outer, k, inner, line, value);
}

/* Seconds per line stored by `passes` passes. */
static double sample(struct arena *a, enum kind kind, enum layout layout, unsigned k, unsigned passes) {
  double t0 = now();
  for (unsigned p = 0; p < passes; ++p) pass(a, kind, layout, k);
  return (now() - t0) / ((double)passes * (double)(slot_bytes(a, k) / a->m->line));
}

static int ascending(const void *x, const void *y) {
  double p = *(const double *)x, q = *(const double *)y;
  return (p > q) - (p < q);
}

static double median(const double *x, unsigned n) {
  double sorted[128];
  memcpy(sorted, x, n * sizeof *x);
  qsort(sorted, n, sizeof *sorted, ascending);
  return n % 2 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}

struct point {
  enum kind kind;
  unsigned k, passes, n;
  double t[layouts][64]; /* seconds per line, n of each */
  double q[64];          /* lockstep / linear, per round */
  double q_median;
};

/* The passes a sample takes: the least whole number whose stores reach sample_bytes. */
static unsigned passes_for(const struct arena *a, unsigned k, size_t sample_bytes) {
  size_t slot = slot_bytes(a, k);
  return (unsigned)((sample_bytes + slot - 1) / slot);
}

static void measure(struct arena *a, struct point *pt, size_t sample_bytes) {
  pt->passes = passes_for(a, pt->k, sample_bytes);
  for (int l = 0; l < layouts; ++l) sample(a, pt->kind, (enum layout)l, pt->k, pt->passes);
  for (unsigned r = 0; r < pt->n; ++r)
    for (int j = 0; j < layouts; ++j) {
      int l = (int)((r + (unsigned)j) % layouts);
      pt->t[l][r] = sample(a, pt->kind, (enum layout)l, pt->k, pt->passes);
    }
  for (unsigned r = 0; r < pt->n; ++r) pt->q[r] = pt->t[lockstep][r] / pt->t[linear][r];
  pt->q_median = median(pt->q, pt->n);
}

/* The fraction of the orderings of n and m samples, all equally likely, whose count of (first, second) pairs with
 * the first larger is at least u (the Mann-Whitney statistic's tail). */
static double tail_fraction(unsigned n, unsigned m, unsigned u) {
  const unsigned most = n * m + 1;
  double *ways = calloc((size_t)2 * (m + 1) * most, sizeof *ways);
  if (!ways) return 1;
#define WAYS(i, j, v) ways[(((size_t)((i) % 2) * (m + 1)) + (j)) * most + (v)]
  for (unsigned i = 0; i <= n; ++i)
    for (unsigned j = 0; j <= m; ++j)
      for (unsigned v = 0; v < most; ++v) {
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

/* n rounds per point and the critical count: the least count chance reaches with probability at most
 * family_false_rate / points; n the least that keeps a real loss past with one disturbed round in each group (a
 * q_K below every q_R and a q_R above every q_K, as an interrupt makes). */
struct test {
  unsigned n, critical;
};

static struct test test_for(double points) {
  const double rate = family_false_rate / points;
  for (unsigned n = 2; n <= 64; ++n) {
    unsigned critical = n * n + 1;
    while (critical > 0 && tail_fraction(n, n, critical - 1) <= rate) --critical;
    if (critical <= n * n - (2 * n - 1)) return (struct test){n, critical};
  }
  return (struct test){64, 64 * 64};
}

static unsigned ceil_log2(unsigned x) {
  unsigned b = 0;
  while ((1u << b) < x) ++b;
  return b;
}

/* The paired kind's source: cycles of at least the second level and at most half the last. */
static int paired_measurable(const struct machine *m, size_t most_slot) { return 2 * (m->l2 + most_slot) <= m->llc; }

static int open_arena(struct arena *a, const struct machine *m, unsigned most_k, size_t stream) {
  memset(a, 0, sizeof *a);
  a->m = m;
  a->stream = stream;
  size_t most_slot = slot_bytes(a, most_k);
  a->dst_bytes = past_llc_multiple * (m->l2 + m->llc);
  if (a->dst_bytes < 4 * most_slot) a->dst_bytes = 4 * most_slot;
  a->src_bytes = m->l2 + most_slot;
  int dst_huge, src_huge;
  a->dst = map(a->dst_bytes, 1, &dst_huge);
  a->src = map(a->src_bytes, 1, &src_huge);
  a->huge = dst_huge && src_huge;
  return a->dst && a->src;
}

static void print_machine(const struct machine *m, const struct arena *a) {
  printf("machine line %zu page %zu l2 %zu llc %zu clock granularity %.1f ns | stream %zu B, destination cycle %zu B, "
         "paired source %zu B (%s), %s pages\n",
         m->line, m->page, m->l2, m->llc, 1e9 * m->granularity, a->stream, a->dst_bytes, a->src_bytes,
         paired_measurable(m, a->src_bytes - m->l2) ? "misses L2, hits LLC" : "not within half the LLC",
         a->huge ? "huge" : "base");
}


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

static void print_point(const struct point *pt, double q_ref, unsigned u, unsigned critical, const char *verdict) {
  printf("point %-6s K %3u passes %4u n %2u | ns/line median lockstep %7.3f linear %7.3f | q median %.3f min %.3f "
         "max %.3f | q/q_ref %.3f | U %u critical %u | %s\n",
         kind_name[pt->kind], pt->k, pt->passes, pt->n, 1e9 * median(pt->t[lockstep], pt->n),
         1e9 * median(pt->t[linear], pt->n), pt->q_median, minimum(pt->q, pt->n), maximum(pt->q, pt->n),
         q_ref > 0 ? pt->q_median / q_ref : 0, u, critical, verdict);
  fflush(stdout);
}

static enum kind kind_of(const char *name) {
  for (int k = 0; k < kinds; ++k)
    if (strcmp(name, kind_name[k]) == 0) return (enum kind)k;
  return kinds;
}

static int grid(const struct machine *m, const char *kind_text, size_t stream, size_t sample_bytes, unsigned rounds,
                char *list) {
  unsigned ks[128], count = 0, most = 0;
  for (char *s = strtok(list, ","); s && count < 128; s = strtok(NULL, ",")) {
    ks[count] = (unsigned)strtoul(s, NULL, 10);
    most = ks[count] > most ? ks[count] : most;
    ++count;
  }
  enum kind kind = kind_of(kind_text);
  if (kind == kinds || rounds == 0 || rounds > 64) return 2;
  struct arena a;
  if (stream % m->line || !open_arena(&a, m, most, stream)) return 1;
  print_machine(m, &a);
  double q_first = 0;
  for (unsigned i = 0; i < count; ++i) {
    struct point pt = {.kind = kind, .k = ks[i], .n = rounds};
    measure(&a, &pt, sample_bytes);
    if (i == 0) q_first = pt.q_median;
    print_point(&pt, q_first, 0, 0, "-");
  }
  printf("cost peak rss %ld KiB\n", peak_rss_kib());
  return 0;
}

struct capacity {
  unsigned pages;      /* 0: none in range */
  double factor;       /* median over the past points of median q_K / median q_R */
  unsigned points;
};

/* Whether `pt` is past the streamer against the reference rounds `ref`. */
static int past(const struct point *pt, const struct point *ref, const struct test *test, unsigned *u) {
  *u = 0;
  for (unsigned i = 0; i < pt->n; ++i)
    for (unsigned j = 0; j < ref->n; ++j) *u += pt->q[i] > ref->q[j];
  return *u >= test->critical && pt->q_median >= 1.5 * ref->q_median;
}

/* Tests K = 2 R, 4 R, ... and MOST_PAGES last until one is past, then bisects between it and the last kept K:
 * the least past K. A loss that fades again at larger K (none seen) cannot hide a knee below it. */
static struct capacity find_capacity(struct arena *a, enum kind kind, unsigned reference, unsigned most,
                                     const struct test *test, size_t sample_bytes) {
  struct capacity c = {0, 0, 0};
  struct point ref = {.kind = kind, .k = reference, .n = test->n};
  measure(a, &ref, sample_bytes);
  print_point(&ref, ref.q_median, 0, test->critical, "reference");
  double ratios[64];
  unsigned past_points = 0, u = 0, lo = reference, hi = 0; /* invariant: lo kept; hi past where not 0 */
  for (unsigned k = 2 * reference; !hi; k = k >= most ? 0 : 2 * k < most ? 2 * k : most) {
    if (k == 0) return c;
    struct point pt = {.kind = kind, .k = k, .n = test->n};
    measure(a, &pt, sample_bytes);
    ++c.points;
    int is_past = past(&pt, &ref, test, &u);
    print_point(&pt, ref.q_median, u, test->critical, is_past ? "PAST" : "kept");
    if (is_past) {
      hi = k;
      ratios[past_points++] = pt.q_median / ref.q_median;
    } else {
      lo = k;
    }
  }
  while (hi - lo > 1) {
    struct point mid = {.kind = kind, .k = (lo + hi) / 2, .n = test->n};
    measure(a, &mid, sample_bytes);
    ++c.points;
    int is_past = past(&mid, &ref, test, &u);
    print_point(&mid, ref.q_median, u, test->critical, is_past ? "PAST" : "kept");
    if (is_past) {
      hi = mid.k;
      ratios[past_points++] = mid.q_median / ref.q_median;
    } else {
      lo = mid.k;
    }
  }
  c.pages = hi;
  c.factor = median(ratios, past_points);
  return c;
}

/* The most points find_capacity tests from `reference`: the doublings to MOST_PAGES, then a bisection of the
 * widest gap they leave (MOST_PAGES less the last doubling below it, or half of a doubled K). */
static unsigned most_points(unsigned reference, unsigned most) {
  unsigned doublings = 0, last = reference;
  for (unsigned k = 2 * reference;; k = 2 * k < most ? 2 * k : most) {
    ++doublings;
    if (k >= most) break;
    last = k;
  }
  unsigned gap = most - last > last / 2 ? most - last : last / 2;
  return doublings + ceil_log2(gap);
}

static int decide(const struct machine *m, unsigned ref_stores, unsigned ref_paired, unsigned most) {
  double t0 = now();
  const size_t sample_bytes = m->l2;
  struct arena a;
  if (most <= ref_stores || most <= ref_paired || !open_arena(&a, m, most, 2 * m->page + m->line)) return 1;
  print_machine(m, &a);
  double mapped = now() - t0;
  const int with_paired = paired_measurable(m, slot_bytes(&a, most));
  const double worst = most_points(ref_stores, most) + 2.0 * most_points(ref_paired, most);
  struct test test = test_for(worst);
  printf("decide K up to %u, %u rounds a point, past at U >= %u of %u and q/q_ref >= 1.5 (at most %.0f points), "
         "sample >= %zu B stored\n",
         most, test.n, test.critical, test.n * test.n, worst, sample_bytes);
  struct capacity c[kinds] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  c[stores] = find_capacity(&a, stores, ref_stores, most, &test, sample_bytes);
  if (with_paired) c[paired] = find_capacity(&a, paired, ref_paired, most, &test, sample_bytes);
  c[memory] = find_capacity(&a, memory, ref_paired, most, &test, sample_bytes);
  for (int k = 0; k < kinds; ++k)
    printf("knee %-6s %u (q/q_ref past it %.2f, %u points)%s\n", kind_name[k], c[k].pages, c[k].factor, c[k].points,
           k == paired && !with_paired ? " not measurable: 2 (L2 + slot) > LLC" : "");
  if (c[stores].pages)
    printf("check memory: knee %u, model ceil(stores / 2) = %u\n", c[memory].pages, (c[stores].pages + 1) / 2);
  double wall = now() - t0;
  if (!c[stores].pages)
    printf("answer none\n");
  else
    printf("answer streamer page %zu stores %u paired %u demand_factor %.0f\n", m->page, c[stores].pages,
           c[paired].pages, floor(c[stores].factor + 0.5));
  unsigned points = c[stores].points + c[paired].points + c[memory].points;
  printf("cost wall %.4f s (mapping and filling %.4f s), %u points + 3 references, %zu bytes mapped, peak rss %ld "
         "KiB\n",
         wall, mapped, points, a.dst_bytes + a.src_bytes, peak_rss_kib());
  return 0;
}

int main(int argc, char **argv) {
  struct machine m;
  setvbuf(stdout, NULL, _IOLBF, 1 << 12);
  place();
  if (!query_machine(&m)) {
    fprintf(stderr, "streamer_probe: the OS reports no line or second level\n");
    return 1;
  }
  if (argc == 5 && strcmp(argv[1], "decide") == 0)
    return decide(&m, (unsigned)atoi(argv[2]), (unsigned)atoi(argv[3]), (unsigned)atoi(argv[4]));
  if (argc == 7 && strcmp(argv[1], "grid") == 0)
    return grid(&m, argv[2], strtoul(argv[3], NULL, 10), strtoul(argv[4], NULL, 10), (unsigned)atoi(argv[5]), argv[6]);
  fprintf(stderr, "usage: streamer_probe decide REF_STORES REF_PAIRED MOST_PAGES | "
                  "streamer_probe grid stores|paired|memory STREAM_BYTES SAMPLE_BYTES ROUNDS K1,K2,...\n");
  return 2;
}
