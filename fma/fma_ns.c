// Counter-free variant of fma.c: FMAs retired per core cycle at 128, 256 and 512 bits and the dependent FMA latency,
// with the core cycle taken from a chain of dependent 64-bit integer adds (latency one cycle on every x86 core) timed
// just before and just after each FMA round. FMAs per cycle = FMAs per ns x ns per add; latency in cycles = ns per
// dependent FMA / ns per add. Where perf_event_open works, the hardware-cycle result of the same round is printed
// beside it.
#include <immintrin.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

static int fd = -1;
static uint64_t cycles(void) { uint64_t v; if (fd < 0 || read(fd, &v, sizeof v) != sizeof v) return 0; return v; }
static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static double pmu_;  // hardware-cycle result of the last round, 0 without a counter
#define ITER 20000000L
#define ADDS 100000000L

static float sink_;
static float m_ = 0.999999f, c_ = 1e-7f;
// Each kernel holds only the FMA loop and no call, so no accumulator is live across a call and the compiler keeps all
// of them in registers; the timers are read around the kernel call in the wrapper.
#define KERNEL(NAME, TARGET, T, FMA, SET)                                                                    \
  __attribute__((target(TARGET), noinline)) static float NAME(void) {                                            \
    T a0 = SET(1), a1 = SET(1), a2 = SET(1), a3 = SET(1), a4 = SET(1), a5 = SET(1), a6 = SET(1), a7 = SET(1),\
      a8 = SET(1), a9 = SET(1), a10 = SET(1), a11 = SET(1), m = SET(m_), c = SET(c_);                        \
    for (long i = 0; i < ITER; ++i) {                                                                        \
      a0 = FMA(a0, m, c); a1 = FMA(a1, m, c); a2 = FMA(a2, m, c); a3 = FMA(a3, m, c);                        \
      a4 = FMA(a4, m, c); a5 = FMA(a5, m, c); a6 = FMA(a6, m, c); a7 = FMA(a7, m, c);                        \
      a8 = FMA(a8, m, c); a9 = FMA(a9, m, c); a10 = FMA(a10, m, c); a11 = FMA(a11, m, c);                    \
      __asm__ volatile("" : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3), "+v"(a4), "+v"(a5), "+v"(a6), "+v"(a7), \
                       "+v"(a8), "+v"(a9), "+v"(a10), "+v"(a11));                                           \
    }                                                                                                        \
    T s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11;                                       \
    float lanes[16]; memcpy(lanes, &s, sizeof s); return lanes[0];                                           \
  }
KERNEL(k512, "avx512f,fma", __m512, _mm512_fmadd_ps, _mm512_set1_ps)
KERNEL(k256, "avx2,fma", __m256, _mm256_fmadd_ps, _mm256_set1_ps)
KERNEL(k128, "avx2,fma", __m128, _mm_fmadd_ps, _mm_set1_ps)
__attribute__((target("avx2,fma"), noinline)) static float klat(void) {
  __m256 a = _mm256_set1_ps(1), m = _mm256_set1_ps(m_), c = _mm256_set1_ps(c_);
  for (long i = 0; i < ITER; ++i) { a = _mm256_fmadd_ps(a, m, c); __asm__ volatile("" : "+v"(a)); }
  float lanes[8]; memcpy(lanes, &a, sizeof a); return lanes[0];
}

// Times one kernel: FMAs (or links of the dependent chain) per ns in the return value, per hardware cycle in pmu_.
#define TIMED(NAME, KERN, COUNT)                                                                             \
  static double NAME(void) {                                                                                 \
    double n0 = now_ns(); uint64_t t0 = cycles();                                                            \
    float s = KERN();                                                                                         \
    uint64_t t1 = cycles(); double n1 = now_ns();                                                            \
    sink_ += s;                                                                                              \
    pmu_ = t1 > t0 ? (double)(COUNT) / (double)(t1 - t0) : 0;                                                \
    return (double)(COUNT) / (n1 - n0);                                                                      \
  }
TIMED(w512, k512, 12 * ITER)
TIMED(w256, k256, 12 * ITER)
TIMED(w128, k128, 12 * ITER)
// Latency is the time of one dependent link, the reciprocal of the others' rate.
TIMED(rlat, klat, ITER)

// Nanoseconds per link of a dependent add chain. Four independent FMAs of the measured width per trip of eight adds
// keeps the core in the frequency licence that width's FMA rounds run under, so the chain reads the clock the FMAs
// saw; it adds no dependence to the chain.
#define ADD_CHAIN(NAME, TARGET, T, FMA, SET)                                                                 \
  __attribute__((target(TARGET))) static double NAME(void) {                                                 \
    uint64_t x = 1, c = 3;                                                                                   \
    T v0 = SET(1), v1 = SET(1), v2 = SET(1), v3 = SET(1), m = SET(m_), k = SET(c_);                                                                  \
    double n0 = now_ns();                                                                                    \
    for (long i = 0; i < ADDS / 8; ++i) {                                                                    \
      v0 = FMA(v0, m, k); v1 = FMA(v1, m, k); v2 = FMA(v2, m, k); v3 = FMA(v3, m, k);                        \
      __asm__ volatile("add %1,%0\n add %1,%0\n add %1,%0\n add %1,%0\n add %1,%0\n add %1,%0\n add %1,%0\n"  \
                       " add %1,%0" : "+r"(x), "+r"(c) :: "cc");                                              \
      __asm__ volatile("" : "+v"(v0), "+v"(v1), "+v"(v2), "+v"(v3));                                                                      \
    }                                                                                                        \
    double n1 = now_ns();                                                                                    \
    float sink[16]; T v = v0 + v1 + v2 + v3; memcpy(sink, &v, sizeof v); sink_ += sink[0] + (float)x;                                 \
    return (n1 - n0) / (double)ADDS;                                                                         \
  }
ADD_CHAIN(add512, "avx512f,fma", __m512, _mm512_fmadd_ps, _mm512_set1_ps)
ADD_CHAIN(add256, "avx2,fma", __m256, _mm256_fmadd_ps, _mm256_set1_ps)
ADD_CHAIN(add128, "avx2,fma", __m128, _mm_fmadd_ps, _mm_set1_ps)

int main(void) {
  struct perf_event_attr pe; memset(&pe, 0, sizeof pe);
  pe.type = PERF_TYPE_HARDWARE; pe.size = sizeof pe; pe.config = PERF_COUNT_HW_CPU_CYCLES;
  pe.exclude_kernel = 1; pe.exclude_hv = 1;
  fd = (int)syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
  if (fd >= 0) ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
  int has512 = __builtin_cpu_supports("avx512f");
  for (int r = 0; r < 3; ++r) {
    double a1 = add128(), x128 = w128(), p128 = pmu_, a2 = add128();
    double b1 = add256(), x256 = w256(), p256 = pmu_, b2 = add256();
    double c1 = has512 ? add512() : 0.0, x512 = has512 ? w512() : 0.0, p512 = pmu_, c2 = has512 ? add512() : 0.0;
    double d1 = add128(), lat = 1 / rlat(), plat = pmu_ > 0 ? 1 / pmu_ : 0, d2 = add128();
    printf("add-clock FMAs per cycle: 128-bit %.2f  256-bit %.2f  512-bit %.2f  dependent FMA latency %.2f cycles  "
           "(clock %.2f %.2f %.2f GHz)", x128 * (a1 + a2) / 2, x256 * (b1 + b2) / 2, x512 * (c1 + c2) / 2,
           lat / ((d1 + d2) / 2), 2 / (a1 + a2), 2 / (b1 + b2), has512 ? 2 / (c1 + c2) : 0.0);
    if (fd >= 0) printf("  | hardware cycles: %.2f %.2f %.2f %.2f", p128, p256, p512, plat);
    printf("\n");
  }
  return sink_ == 12345.f;
}
