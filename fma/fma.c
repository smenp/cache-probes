// FMAs retired per core cycle at 128, 256 and 512 bits, and the latency of one dependent FMA chain, counted in core
// cycles (perf user cycles), so the clock's licence and turbo do not enter. Twelve independent accumulators per
// width exceed latency x units on every x86 core with at most two FMA pipes of latency at most six.
#include <immintrin.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

static int fd;
static uint64_t cycles(void) { uint64_t v; if (read(fd, &v, sizeof v) != sizeof v) return 0; return v; }
#define ITER 20000000L

#define BODY(T, FMA, SET)                                                                                    \
  T a0 = SET(1), a1 = SET(1), a2 = SET(1), a3 = SET(1), a4 = SET(1), a5 = SET(1), a6 = SET(1), a7 = SET(1),  \
    a8 = SET(1), a9 = SET(1), a10 = SET(1), a11 = SET(1), m = SET(m_), c = SET(c_);                          \
  uint64_t t0 = cycles();                                                                                    \
  for (long i = 0; i < ITER; ++i) {                                                                          \
    a0 = FMA(a0, m, c); a1 = FMA(a1, m, c); a2 = FMA(a2, m, c); a3 = FMA(a3, m, c);                          \
    a4 = FMA(a4, m, c); a5 = FMA(a5, m, c); a6 = FMA(a6, m, c); a7 = FMA(a7, m, c);                          \
    a8 = FMA(a8, m, c); a9 = FMA(a9, m, c); a10 = FMA(a10, m, c); a11 = FMA(a11, m, c);                      \
    __asm__ volatile("" : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3), "+v"(a4), "+v"(a5), "+v"(a6), "+v"(a7),   \
                     "+v"(a8), "+v"(a9), "+v"(a10), "+v"(a11));                                              \
  }                                                                                                          \
  uint64_t t1 = cycles();                                                                                    \
  T s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11;                                         \
  float sink[16]; memcpy(sink, &s, sizeof s); sink_ += sink[0];                                              \
  return (double)(12 * ITER) / (double)(t1 - t0);

static float sink_;
static float m_ = 0.999999f, c_ = 1e-7f;
__attribute__((target("avx512f,fma"))) static double w512(void) { BODY(__m512, _mm512_fmadd_ps, _mm512_set1_ps) }
__attribute__((target("avx2,fma"))) static double w256(void) { BODY(__m256, _mm256_fmadd_ps, _mm256_set1_ps) }
__attribute__((target("avx2,fma"))) static double w128(void) { BODY(__m128, _mm_fmadd_ps, _mm_set1_ps) }
__attribute__((target("avx2,fma"))) static double latency(void) {
  __m256 a = _mm256_set1_ps(1), m = _mm256_set1_ps(m_), c = _mm256_set1_ps(c_);
  uint64_t t0 = cycles();
  for (long i = 0; i < ITER; ++i) { a = _mm256_fmadd_ps(a, m, c); __asm__ volatile("" : "+v"(a)); }
  uint64_t t1 = cycles();
  float sink[8]; memcpy(sink, &a, sizeof a); sink_ += sink[0];
  return (double)(t1 - t0) / (double)ITER;
}

int main(void) {
  struct perf_event_attr pe; memset(&pe, 0, sizeof pe);
  pe.type = PERF_TYPE_HARDWARE; pe.size = sizeof pe; pe.config = PERF_COUNT_HW_CPU_CYCLES;
  pe.exclude_kernel = 1; pe.exclude_hv = 1;
  fd = (int)syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
  if (fd < 0) { perror("perf_event_open"); return 1; }
  ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
  for (int r = 0; r < 3; ++r)
    printf("FMAs per cycle: 128-bit %.2f  256-bit %.2f  512-bit %.2f  dependent FMA latency %.2f cycles\n",
           w128(), w256(), __builtin_cpu_supports("avx512f") ? w512() : 0.0, latency());
  return sink_ == 12345.f;
}
