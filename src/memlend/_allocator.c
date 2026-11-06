// Segment allocator behind the MemPool used during a loan (see region.py and DESIGN.md).
//
// The caching allocator asks for whole segments (multiples of 2 MiB). memlend_malloc serves each one from a
// contiguous run of free 2 MiB pages of the region lent on that device, chosen in lend order, and falls back to
// cuMemAlloc (ordinary free GPU memory) when no run is long enough. Every block a served run overlaps is marked touched.
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

#define PAGE (2ull << 20)
#define MAX_DEVICES 64

typedef struct Region {
  int device;
  CUdeviceptr base;
  int64_t npages, pages_per_block;
  int32_t* order;    // pages in lend order
  int64_t* block_order;  // blocks in lend order
  uint8_t* used;     // page currently backs a segment of a pool
  uint8_t* touched;  // per block: handed out during the current loan
  uint8_t* active;   // scratch for the run search
  int64_t* end;      // scratch: for an active run [a, b], end[a] = b and end[b] = a
  // Lazy backup: before a block is first handed out, it is copied to `backup` on `copy_stream`, and the stream that
  // asked for the memory waits for that copy (`sync_copies`: the host waits instead). `prefetch` more blocks, next in
  // lend order, are kept copied ahead of demand. Blocks marked `skip` hold nothing and are never copied.
  char* backup;
  int sync_copies;
  int64_t prefetch, cursor, ahead;
  uint8_t* skip;
  uint8_t* saved;  // copy-out enqueued during this loan
  CUevent* events;  // per block: recorded after its copy-out
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
  r->order = malloc(r->npages * sizeof(int32_t));
  r->block_order = malloc(r->npages / r->pages_per_block * sizeof(int64_t));
  r->skip = calloc(r->npages / r->pages_per_block, 1);
  r->saved = calloc(r->npages / r->pages_per_block, 1);
  r->used = calloc(r->npages, 1);
  r->touched = calloc(r->npages / r->pages_per_block, 1);
  r->active = malloc(r->npages);
  r->end = malloc(r->npages * sizeof(int64_t));
  for (int64_t i = 0; i < r->npages; i++) r->order[i] = (int32_t)(r->npages - 1 - i);  // far end first
  for (int64_t b = 0; b < r->npages / r->pages_per_block; b++) r->block_order[b] = r->npages / r->pages_per_block - 1 - b;
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
  free(r->events), free(r->block_order), free(r->skip), free(r->saved);
  free(r->order), free(r->used), free(r->touched), free(r->active), free(r->end), free(r);
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

// block_order: every block index once, first lent first.
void memlend_set_order(Region* r, const int64_t* block_order) {
  pthread_mutex_lock(&mu);
  int64_t i = 0;
  for (int64_t b = 0; b < r->npages / r->pages_per_block; b++) {
    r->block_order[b] = block_order[b];
    for (int64_t p = 0; p < r->pages_per_block; p++) r->order[i++] = (int32_t)(block_order[b] * r->pages_per_block + p);
  }
  pthread_mutex_unlock(&mu);
}

static void copy_out(Region* r, int64_t b) {
  size_t bs = r->pages_per_block * PAGE;
  cuMemcpyDtoHAsync(r->backup + b * bs, r->base + b * bs, bs, r->copy_stream);
  cuEventRecord(r->events[b], r->copy_stream);
  r->saved[b] = 1;
}

static void prefetch(Region* r) {
  int64_t nblocks = r->npages / r->pages_per_block;
  while (r->ahead < r->prefetch && r->cursor < nblocks) {
    int64_t b = r->block_order[r->cursor++];
    if (r->skip[b] || r->saved[b] || r->touched[b]) continue;
    copy_out(r, b);
    r->ahead++;
  }
}

// Start a loan of r on its device. backup (pinned host, optional) enables lazy backup; skip (one byte per block,
// optional) marks blocks that hold nothing.
void memlend_begin(Region* r, char* backup, const uint8_t* skip, int sync_copies, int64_t prefetch_blocks) {
  int64_t nblocks = r->npages / r->pages_per_block;
  pthread_mutex_lock(&mu);
  memset(r->touched, 0, nblocks);
  memset(r->saved, 0, nblocks);
  if (skip) memcpy(r->skip, skip, nblocks);
  else memset(r->skip, 0, nblocks);
  r->backup = backup;
  r->sync_copies = sync_copies;
  r->prefetch = prefetch_blocks;
  r->cursor = r->ahead = 0;
  if (backup && !r->copy_stream) {
    cuStreamCreate(&r->copy_stream, 1 /* CU_STREAM_NON_BLOCKING */);
    r->events = malloc(nblocks * sizeof(CUevent));
    for (int64_t b = 0; b < nblocks; b++) cuEventCreate(&r->events[b], 2 /* CU_EVENT_DISABLE_TIMING */);
  }
  if (backup) prefetch(r);
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
void memlend_saved(Region* r, uint8_t* out) { memcpy(out, r->saved, r->npages / r->pages_per_block); }
int64_t memlend_fallback_peak(int device) { return fallback_peak[device]; }

// First page of a window of k free pages within the shortest lend-order prefix that has one, or -1.
static int64_t find_run(Region* r, int64_t k) {
  memset(r->active, 0, r->npages);
  for (int64_t i = 0; i < r->npages; i++) {
    int64_t p = r->order[i];
    if (r->used[p]) continue;
    int64_t a = (p > 0 && r->active[p - 1]) ? r->end[p - 1] : p;
    int64_t b = (p + 1 < r->npages && r->active[p + 1]) ? r->end[p + 1] : p;
    r->active[p] = 1;
    r->end[a] = b;
    r->end[b] = a;
    // Both sides were shorter than k before p joined them, so any k-window of [a, b] contains p.
    if (b - a + 1 >= k) return a > p - k + 1 ? a : p - k + 1;
  }
  return -1;
}

void* memlend_malloc(ssize_t size, int device, void* stream) {
  int64_t k = (size + PAGE - 1) / PAGE;
  pthread_mutex_lock(&mu);
  Region* r = lender[device];
  int64_t start = r ? find_run(r, k) : -1;
  if (start >= 0) {
    memset(r->used + start, 1, k);
    int wait_host = 0;
    for (int64_t b = start / r->pages_per_block; b <= (start + k - 1) / r->pages_per_block; b++) {
      if (!r->touched[b] && r->saved[b]) r->ahead--;  // a prefetched block is now in use
      r->touched[b] = 1;
      if (!r->backup || r->skip[b]) continue;
      if (!r->saved[b]) copy_out(r, b);
      if (r->sync_copies)
        wait_host = 1;
      else
        cuStreamWaitEvent((CUstream)stream, r->events[b], 0);  // the borrower's kernels on this stream follow the copy
    }
    if (wait_host) cuStreamSynchronize(r->copy_stream);
    if (r->backup) prefetch(r);
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
