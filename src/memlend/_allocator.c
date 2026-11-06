// Segment allocator behind the MemPool used during a loan (see region.py and DESIGN.md).
//
// The caching allocator asks for whole segments (multiples of 2 MiB). memlend_malloc serves each one from the run of
// contiguous unlent 2 MiB pages of the region lent on that device that costs least: the number of distinct blocks in
// it that hold data and were not touched yet in this loan (empty blocks and already-touched blocks are free). Ties go to
// the far end of the region. It falls back to cuMemAlloc (ordinary free GPU memory) only when no run is long enough.
//
// The CUDA driver is opened at runtime (it is already loaded by PyTorch), so building needs only a C compiler.

#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

typedef unsigned long long CUdeviceptr;
typedef int CUresult;
typedef void* CUstream;
typedef void* CUcontext;
typedef void* CUevent;

#define DRIVER_FUNCTIONS(X)                                                       \
  X(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain", CUcontext*, int)       \
  X(cuCtxSetCurrent, "cuCtxSetCurrent", CUcontext)                               \
  X(cuMemAlloc, "cuMemAlloc_v2", CUdeviceptr*, size_t)                           \
  X(cuMemFree, "cuMemFree_v2", CUdeviceptr)                                      \
  X(cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2", void*, CUdeviceptr, size_t, CUstream) \
  X(cuStreamCreate, "cuStreamCreate", CUstream*, unsigned int)                   \
  X(cuStreamSynchronize, "cuStreamSynchronize", CUstream)                        \
  X(cuStreamDestroy, "cuStreamDestroy_v2", CUstream)                             \
  X(cuEventCreate, "cuEventCreate", CUevent*, unsigned int)                      \
  X(cuEventRecord, "cuEventRecord", CUevent, CUstream)                           \
  X(cuEventDestroy, "cuEventDestroy_v2", CUevent)                                \
  X(cuStreamWaitEvent, "cuStreamWaitEvent", CUstream, CUevent, unsigned int)

#define DECLARE(name, symbol, ...) static CUresult (*name)(__VA_ARGS__);
DRIVER_FUNCTIONS(DECLARE)

// Returns 0 on success, -1 if libcuda.so.1 or one of its symbols is missing.
int memlend_init(void) {
  void* h = dlopen("libcuda.so.1", RTLD_NOW);
  if (!h) return -1;
#define LOAD(name, symbol, ...) \
  if (!(name = (CUresult(*)(__VA_ARGS__))dlsym(h, symbol))) return -1;
  DRIVER_FUNCTIONS(LOAD)
  return 0;
}

#define PAGE (2ull * 1024 * 1024)
#define MAX_DEVICES 64

typedef struct Region {
  int device;
  CUdeviceptr base;
  int64_t npages, pages_per_block;
  uint8_t* used;         // page currently backs a segment of a pool
  uint8_t* skip;         // per block: declared empty for this loan (holds nothing, never backed up)
  uint8_t* touched;      // per block: handed out during the current loan
  int64_t* cost_prefix;  // scratch: number of costly blocks among the first b
  // Lazy backup: before a non-empty block is first handed out, it is copied to `backup` on `copy_stream`, and every
  // stream that asks for memory in it waits for that copy (its event).
  char* backup;
  CUevent* events;
  CUstream copy_stream;
  int dropped;  // the owner is gone; destroy once no page is used
  struct Region* next;
} Region;

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static Region* regions;                // all live regions: frees can arrive after the loan that made the segment ended
static Region* lender[MAX_DEVICES];    // region serving mallocs during the current loan on each device
static int64_t fallback_bytes[MAX_DEVICES], fallback_peak[MAX_DEVICES];

Region* memlend_create(int device, size_t nbytes, int64_t block_size) {
  // PyTorch's runtime calls use the device's primary context; the calling thread may not have it current yet.
  CUcontext ctx;
  Region* r = calloc(1, sizeof(Region));
  if (cuDevicePrimaryCtxRetain(&ctx, device) != 0 || cuCtxSetCurrent(ctx) != 0 || cuMemAlloc(&r->base, nbytes) != 0) {
    free(r);
    return NULL;
  }
  r->device = device;
  r->npages = nbytes / PAGE;
  r->pages_per_block = block_size / PAGE;
  int64_t nblocks = r->npages / r->pages_per_block;
  r->used = calloc(r->npages, 1);
  r->skip = calloc(nblocks, 1);
  r->touched = calloc(nblocks, 1);
  r->cost_prefix = malloc((nblocks + 1) * sizeof(int64_t));
  pthread_mutex_lock(&mu);
  r->next = regions;
  regions = r;
  pthread_mutex_unlock(&mu);
  return r;
}

static int64_t used_pages(Region* r) {
  int64_t n = 0;
  for (int64_t i = 0; i < r->npages; i++) n += r->used[i];
  return n;
}

static void destroy(Region* r) {
  for (Region** p = &regions; *p; p = &(*p)->next)
    if (*p == r) {
      *p = r->next;
      break;
    }
  cuMemFree(r->base);
  if (r->copy_stream) cuStreamDestroy(r->copy_stream);
  if (r->events)
    for (int64_t b = 0; b < r->npages / r->pages_per_block; b++) cuEventDestroy(r->events[b]);
  free(r->events), free(r->used), free(r->skip), free(r->touched), free(r->cost_prefix), free(r);
}

// The owner dropped the region. If leaked borrower segments still live in it, free it when the last one goes.
void memlend_destroy(Region* r) {
  pthread_mutex_lock(&mu);
  if (used_pages(r))
    r->dropped = 1;
  else
    destroy(r);
  pthread_mutex_unlock(&mu);
}

CUdeviceptr memlend_base(Region* r) { return r->base; }

// Start a loan of r on its device. backup (pinned host, optional) enables lazy backup; skip (one byte per block,
// optional) marks blocks that hold nothing.
void memlend_begin(Region* r, char* backup, const uint8_t* skip) {
  int64_t nblocks = r->npages / r->pages_per_block;
  pthread_mutex_lock(&mu);
  memset(r->touched, 0, nblocks);
  if (skip) memcpy(r->skip, skip, nblocks);
  else memset(r->skip, 0, nblocks);
  r->backup = backup;
  if (backup && !r->copy_stream) {
    cuStreamCreate(&r->copy_stream, 1 /* CU_STREAM_NON_BLOCKING */);
    r->events = malloc(nblocks * sizeof(CUevent));
    for (int64_t b = 0; b < nblocks; b++) cuEventCreate(&r->events[b], 2 /* CU_EVENT_DISABLE_TIMING */);
  }
  fallback_peak[r->device] = fallback_bytes[r->device];
  lender[r->device] = r;
  pthread_mutex_unlock(&mu);
}

void memlend_end(Region* r) {
  pthread_mutex_lock(&mu);
  lender[r->device] = NULL;
  pthread_mutex_unlock(&mu);
}

void memlend_touched(Region* r, uint8_t* out) { memcpy(out, r->touched, r->npages / r->pages_per_block); }
int64_t memlend_fallback_peak(int device) { return fallback_peak[device]; }

// First page of the run of n contiguous unlent pages with the fewest distinct blocks that hold data and were not
// touched yet in this loan; among equal costs, the one nearest the far end. -1 if no n unlent pages are contiguous.
static int64_t find_window(Region* r, int64_t n) {
  int64_t ppb = r->pages_per_block, nblocks = r->npages / ppb;
  r->cost_prefix[0] = 0;
  for (int64_t b = 0; b < nblocks; b++) r->cost_prefix[b + 1] = r->cost_prefix[b] + !(r->skip[b] || r->touched[b]);
  int64_t best = -1, best_cost = INT64_MAX, free_len = 0;
  for (int64_t p = r->npages - 1; p >= 0; p--) {  // from the far end, so the first window of a cost is the farthest
    free_len = r->used[p] ? 0 : free_len + 1;     // unlent pages from p upward
    if (free_len < n) continue;
    int64_t cost = r->cost_prefix[(p + n - 1) / ppb + 1] - r->cost_prefix[p / ppb];
    if (cost < best_cost) {
      best = p, best_cost = cost;
      if (cost == 0) break;
    }
  }
  return best;
}

void* memlend_malloc(ssize_t size, int device, void* stream) {
  int64_t k = (size + PAGE - 1) / PAGE;
  pthread_mutex_lock(&mu);
  Region* r = lender[device];
  int64_t start = r ? find_window(r, k) : -1;
  if (start >= 0) {
    memset(r->used + start, 1, k);
    for (int64_t b = start / r->pages_per_block; b <= (start + k - 1) / r->pages_per_block; b++) {
      int first = !r->touched[b];
      r->touched[b] = 1;
      if (!r->backup || r->skip[b]) continue;
      if (first) {
        size_t bs = r->pages_per_block * PAGE;
        cuMemcpyDtoHAsync(r->backup + b * bs, r->base + b * bs, bs, r->copy_stream);
        cuEventRecord(r->events[b], r->copy_stream);
      }
      cuStreamWaitEvent((CUstream)stream, r->events[b], 0);  // the borrower's kernels on this stream follow the copy
    }
    pthread_mutex_unlock(&mu);
    return (void*)(r->base + start * PAGE);
  }
  CUdeviceptr p = 0;
  if (cuMemAlloc(&p, size) == 0) {
    fallback_bytes[device] += size;
    if (fallback_bytes[device] > fallback_peak[device]) fallback_peak[device] = fallback_bytes[device];
  }
  pthread_mutex_unlock(&mu);
  return (void*)p;
}

void memlend_free(void* ptr, size_t size, int device, void* stream) {
  CUdeviceptr p = (CUdeviceptr)ptr;
  pthread_mutex_lock(&mu);
  for (Region* r = regions; r; r = r->next)
    if (p >= r->base && p < r->base + r->npages * PAGE) {
      memset(r->used + (p - r->base) / PAGE, 0, (size + PAGE - 1) / PAGE);
      if (r->dropped && !used_pages(r)) destroy(r);
      pthread_mutex_unlock(&mu);
      return;
    }
  cuMemFree(p);
  fallback_bytes[device] -= size;
  pthread_mutex_unlock(&mu);
}
