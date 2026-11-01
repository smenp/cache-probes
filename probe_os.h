/* probe_os.h: what the three probes ask of the operating system and the compiler, one #if block per OS.
 *
 *   now()              seconds, a monotonic clock (clock_gettime, clock_gettime_nsec_np, QueryPerformanceCounter)
 *   clock_granularity  the least nonzero step between two reads of now()
 *   place()            one logical CPU for this thread, printed: the last the process may use on Linux and Windows;
 *                      on macOS, which has no affinity API, the highest quality of service (performance cores first)
 *   map(bytes, huge)   zeroed-then-filled memory; huge = 1 asks Linux for transparent huge pages, 0 forbids them;
 *                      returns whether they were granted (never elsewhere)
 *   os_caches()        every cache record the OS reports for the CPU this runs on, printed as "os cache: ..." lines
 *   peak_rss_kib()     the process's peak resident memory
 *
 * And 16-byte vectors for loads and stores of exactly 16 bytes: GCC vector extensions where the compiler has them,
 * SSE2 or NEON intrinsics under MSVC. */
#ifndef PROBE_OS_H
#define PROBE_OS_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>
#else
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define NOINLINE __declspec(noinline)
#define COMPILER_BARRIER() _ReadWriteBarrier()
#if defined(_M_X64)
#include <intrin.h>
typedef __m128i v16;
#define V16_SPLAT(b) _mm_set1_epi8((char)(b))
#define V16_ADD(a, b) _mm_add_epi8((a), (b))
#define V16_LOAD(p) _mm_load_si128((const __m128i *)(p))
#define V16_STORE(p, v) _mm_store_si128((__m128i *)(p), (v))
#elif defined(_M_ARM64)
#include <arm_neon.h>
typedef uint8x16_t v16;
#define V16_SPLAT(b) vdupq_n_u8((uint8_t)(b))
#define V16_ADD(a, b) vaddq_u8((a), (b))
#define V16_LOAD(p) vld1q_u8((const uint8_t *)(p))
#define V16_STORE(p, v) vst1q_u8((uint8_t *)(p), (v))
#endif
#else
#define NOINLINE __attribute__((noinline))
#define COMPILER_BARRIER() __asm__ volatile("" ::: "memory")
typedef unsigned char v16 __attribute__((vector_size(16), aligned(16)));
#define V16_SPLAT(b) ((v16){0} + (unsigned char)(b))
#define V16_ADD(a, b) ((a) + (b))
#define V16_LOAD(p) (*(const v16 *)(p))
#define V16_STORE(p, v) (*(v16 *)(p) = (v))
#endif

/* One cache as the OS reports it; ways 0: not reported (or fully associative, said in the printed record). */
struct cache {
  unsigned level;
  char type; /* 'D' data, 'I' instruction, 'U' unified */
  size_t size, line, ways;
};
enum { most_caches = 16 };

#if defined(_WIN32)
static inline double now(void) {
  static double tick = 0;
  LARGE_INTEGER t;
  if (tick == 0) {
    QueryPerformanceFrequency(&t);
    tick = 1.0 / (double)t.QuadPart;
  }
  QueryPerformanceCounter(&t);
  return (double)t.QuadPart * tick;
}
static inline void place(void) {
  DWORD_PTR process, system;
  GetProcessAffinityMask(GetCurrentProcess(), &process, &system);
  int last = -1;
  for (int c = 0; c < (int)(8 * sizeof process); ++c)
    if (process >> c & 1) last = c;
  if (last >= 0) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << last);
  Sleep(0); /* lets the scheduler move the thread before the first timing */
  printf("placement: pinned to logical CPU %d (SetThreadAffinityMask), running on %lu\n", last,
         GetCurrentProcessorNumber());
}
static inline size_t page_size(void) {
  SYSTEM_INFO s;
  GetSystemInfo(&s);
  return s.dwPageSize;
}
static inline unsigned char *map(size_t bytes, int huge, int *granted) {
  (void)huge;
  *granted = 0;
  unsigned char *p = VirtualAlloc(NULL, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (p) memset(p, 1, bytes);
  return p;
}
static inline long peak_rss_kib(void) {
  PROCESS_MEMORY_COUNTERS c;
  return K32GetProcessMemoryInfo(GetCurrentProcess(), &c, sizeof c) ? (long)(c.PeakWorkingSetSize >> 10) : -1;
}
/* GetLogicalProcessorInformationEx's cache records: every one printed, those of this thread's processor kept. */
static inline unsigned os_caches(struct cache *out) {
  DWORD bytes = 0;
  GetLogicalProcessorInformationEx(RelationCache, NULL, &bytes);
  unsigned char *buffer = malloc(bytes);
  unsigned count = 0;
  if (!buffer ||
      !GetLogicalProcessorInformationEx(RelationCache, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer, &bytes)) {
    free(buffer);
    return 0;
  }
  PROCESSOR_NUMBER self;
  GetCurrentProcessorNumberEx(&self);
  static const char type_letter[] = {'U', 'I', 'D', 'T'};
  for (DWORD at = 0; at < bytes;) {
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX r = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buffer + at);
    const CACHE_RELATIONSHIP *c = &r->Cache;
    int mine = c->GroupMask.Group == self.Group && (c->GroupMask.Mask >> self.Number & 1);
    printf("os cache: GetLogicalProcessorInformationEx level %u type %c size %lu line %u associativity %u%s group %u "
           "mask %#llx%s\n",
           c->Level, c->Type <= CacheTrace ? type_letter[c->Type] : '?', c->CacheSize, c->LineSize, c->Associativity,
           c->Associativity == CACHE_FULLY_ASSOCIATIVE ? " (fully associative)" : "", c->GroupMask.Group,
           (unsigned long long)c->GroupMask.Mask, mine ? " (this thread's)" : "");
    if (mine && count < most_caches && c->Type <= CacheData) {
      struct cache k = {c->Level, type_letter[c->Type], c->CacheSize, c->LineSize,
                        c->Associativity == CACHE_FULLY_ASSOCIATIVE ? 0 : c->Associativity};
      out[count++] = k;
    }
    at += r->Size;
  }
  free(buffer);
  return count;
}
#elif defined(__APPLE__)
static inline double now(void) { return 1e-9 * (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static inline void place(void) {
  int failed = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  printf("placement: no affinity API on macOS; QOS_CLASS_USER_INTERACTIVE %s\n", failed ? "refused" : "set");
}
static inline size_t page_size(void) { return (size_t)sysconf(_SC_PAGESIZE); }
static inline unsigned char *map(size_t bytes, int huge, int *granted) {
  (void)huge;
  *granted = 0;
  void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (p == MAP_FAILED) return NULL;
  memset(p, 1, bytes);
  return p;
}
static inline long peak_rss_kib(void) {
  struct rusage u;
  getrusage(RUSAGE_SELF, &u);
  return u.ru_maxrss >> 10; /* bytes on macOS */
}
static inline size_t sysctl_size(const char *name) {
  uint64_t v = 0;
  size_t len = sizeof v;
  if (sysctlbyname(name, &v, &len, NULL, 0) != 0) return 0;
  if (len == sizeof(uint32_t)) v = (uint32_t)v;
  printf("os cache: sysctl %s %llu\n", name, (unsigned long long)v);
  return (size_t)v;
}
/* The performance cores' caches (hw.perflevel0) where the system describes core types, else the single set of
 * keys. macOS reports sizes and the line, no associativity. */
static inline unsigned os_caches(struct cache *out) {
  static const struct {
    unsigned level;
    char type;
    const char *perflevel, *plain;
  } keys[] = {{1, 'D', "hw.perflevel0.l1dcachesize", "hw.l1dcachesize"},
              {1, 'I', "hw.perflevel0.l1icachesize", "hw.l1icachesize"},
              {2, 'U', "hw.perflevel0.l2cachesize", "hw.l2cachesize"},
              {3, 'U', "hw.perflevel0.l3cachesize", "hw.l3cachesize"}};
  size_t line = sysctl_size("hw.cachelinesize");
  unsigned count = 0;
  for (size_t i = 0; i < sizeof keys / sizeof *keys; ++i) {
    size_t size = sysctl_size(keys[i].perflevel);
    if (size == 0) size = sysctl_size(keys[i].plain);
    if (size) {
      struct cache k = {keys[i].level, keys[i].type, size, line, 0};
      out[count++] = k;
    }
  }
  return count;
}
#else
static inline double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}
static inline void place(void) {
  cpu_set_t set;
  int last = -1;
  if (sched_getaffinity(0, sizeof set, &set) == 0)
    for (int c = 0; c < CPU_SETSIZE; ++c)
      if (CPU_ISSET(c, &set)) last = c;
  if (last >= 0) {
    CPU_ZERO(&set);
    CPU_SET(last, &set);
    sched_setaffinity(0, sizeof set, &set);
  }
  printf("placement: pinned to logical CPU %d (sched_setaffinity), running on %d\n", last, sched_getcpu());
}
static inline size_t page_size(void) { return (size_t)sysconf(_SC_PAGESIZE); }
/* huge = 1: transparent huge pages where the system grants them; huge = 0: base pages only, every page alike for
 * every layout. A huge page backs only a whole 2 MiB extent, aligned, that the advice covers, so the region is mapped,
 * advised and touched in whole aligned extents. *granted: the region's mapping holds huge pages. */
static inline unsigned char *map(size_t bytes, int huge, int *granted) {
  const size_t two_mib = (size_t)2 << 20, whole = (bytes + two_mib - 1) / two_mib * two_mib;
  unsigned char *p = mmap(NULL, whole + two_mib, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return NULL;
  unsigned char *aligned = (unsigned char *)(((uintptr_t)p + two_mib - 1) & ~(uintptr_t)(two_mib - 1));
  madvise(aligned, whole, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
  memset(aligned, 1, whole);
  FILE *f = fopen("/proc/self/smaps", "r");
  char text[256];
  int mine = 0;
  long kib = 0;
  while (f && fgets(text, sizeof text, f)) {
    unsigned long start, end;
    if (sscanf(text, "%lx-%lx ", &start, &end) == 2)
      mine = start <= (uintptr_t)aligned && (uintptr_t)aligned < end;
    else if (mine && strncmp(text, "AnonHugePages:", 14) == 0)
      kib = strtol(text + 14, NULL, 10);
  }
  if (f) fclose(f);
  *granted = kib > 0;
  return aligned;
}
static inline long peak_rss_kib(void) {
  struct rusage u;
  getrusage(RUSAGE_SELF, &u);
  return u.ru_maxrss;
}
static inline size_t sysfs_value(int cpu, int index, const char *name, char *text, size_t room) {
  char path[128];
  snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cache/index%d/%s", cpu, index, name);
  FILE *f = fopen(path, "r");
  text[0] = 0;
  if (!f) return 0;
  if (!fgets(text, (int)room, f)) text[0] = 0;
  fclose(f);
  text[strcspn(text, "\n")] = 0;
  char *end = NULL;
  size_t v = strtoul(text, &end, 10);
  return end && *end == 'K' ? v << 10 : end && *end == 'M' ? v << 20 : v;
}
/* Every sysfs cache entry of the CPU this runs on, printed and kept. */
static inline unsigned os_caches(struct cache *out) {
  static const char *const names[] = {"level", "type", "size", "coherency_line_size", "ways_of_associativity",
                                      "number_of_sets", "shared_cpu_list"};
  int cpu = sched_getcpu();
  unsigned count = 0;
  char text[64];
  for (int i = 0; count < most_caches; ++i) {
    size_t level = sysfs_value(cpu, i, "level", text, sizeof text);
    if (!text[0]) break;
    printf("os cache: sysfs cpu%d/index%d", cpu, i);
    for (size_t n = 0; n < sizeof names / sizeof *names; ++n) {
      sysfs_value(cpu, i, names[n], text, sizeof text);
      printf(" %s %s", names[n], text[0] ? text : "-");
    }
    printf("\n");
    sysfs_value(cpu, i, "type", text, sizeof text);
    struct cache k = {(unsigned)level, text[0] == 'D' ? 'D' : text[0] == 'I' ? 'I' : 'U',
                      sysfs_value(cpu, i, "size", text, sizeof text),
                      sysfs_value(cpu, i, "coherency_line_size", text, sizeof text),
                      sysfs_value(cpu, i, "ways_of_associativity", text, sizeof text)};
    out[count++] = k;
  }
  return count;
}
#endif

static inline double clock_granularity(void) {
  double least = 1;
  for (int seen = 0; seen < 64; ++seen) {
    double a = now(), b;
    while ((b = now()) == a) {}
    if (b - a < least) least = b - a;
  }
  return least;
}

/* The first cache of `level` whose type is `type`, or unified where `type` is 'D' (a unified cache holds data);
 * NULL if the OS reports none. */
static inline const struct cache *find_cache(const struct cache *c, unsigned count, unsigned level, char type) {
  for (unsigned i = 0; i < count; ++i)
    if (c[i].level == level && (c[i].type == type || (type == 'D' && c[i].type == 'U'))) return &c[i];
  return NULL;
}

#endif
