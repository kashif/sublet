// Segment allocator behind the MemPool used during a loan (see region.py).
//
// The region is one address range backed by one physical 2 MiB page per page of the range. The caching allocator asks
// for whole segments (multiples of 2 MiB); memlend_malloc gives each one a fresh address range and maps unlent pages of
// the region lent on that device behind it, side by side, wherever they sit in the region: first pages of blocks that
// cost nothing (declared empty, or already lent in this loan), then pages of blocks holding data, whole blocks at a time;
// each from the far end of the region. When the region has no unlent page left, the rest of the segment is new device
// memory. Data blocks are copied to the backup buffer before their first page is lent (lazy backup).
//
// The CUDA driver is opened at runtime (it is already loaded by PyTorch), so building needs only a C compiler.

#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;
typedef int CUresult;
typedef void* CUstream;
typedef void* CUcontext;
typedef void* CUevent;
typedef struct {
  int type;  // CU_MEM_LOCATION_TYPE_DEVICE
  int id;
} CUmemLocation;
typedef struct {
  int type;                  // CU_MEM_ALLOCATION_TYPE_PINNED
  int requestedHandleTypes;  // CU_MEM_HANDLE_TYPE_NONE
  CUmemLocation location;
  void* win32HandleMetaData;
  unsigned char compressionType, gpuDirectRDMACapable;
  unsigned short usage;
  unsigned char reserved[4];
} CUmemAllocationProp;
typedef struct {
  CUmemLocation location;
  int flags;  // CU_MEM_ACCESS_FLAGS_PROT_READWRITE
} CUmemAccessDesc;

#define DRIVER_FUNCTIONS(X)                                                                                 \
  X(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain", CUcontext*, int)                                 \
  X(cuCtxSetCurrent, "cuCtxSetCurrent", CUcontext)                                                         \
  X(cuCtxSynchronize, "cuCtxSynchronize")                                                                  \
  X(cuMemAddressReserve, "cuMemAddressReserve", CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) \
  X(cuMemAddressFree, "cuMemAddressFree", CUdeviceptr, size_t)                                             \
  X(cuMemCreate, "cuMemCreate", CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long) \
  X(cuMemRelease, "cuMemRelease", CUmemGenericAllocationHandle)                                            \
  X(cuMemMap, "cuMemMap", CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long)   \
  X(cuMemUnmap, "cuMemUnmap", CUdeviceptr, size_t)                                                         \
  X(cuMemSetAccess, "cuMemSetAccess", CUdeviceptr, size_t, const CUmemAccessDesc*, size_t)                 \
  X(cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2", void*, CUdeviceptr, size_t, CUstream)                       \
  X(cuStreamCreate, "cuStreamCreate", CUstream*, unsigned int)                                             \
  X(cuStreamDestroy, "cuStreamDestroy_v2", CUstream)                                                       \
  X(cuEventCreate, "cuEventCreate", CUevent*, unsigned int)                                                \
  X(cuEventRecord, "cuEventRecord", CUevent, CUstream)                                                     \
  X(cuEventDestroy, "cuEventDestroy_v2", CUevent)                                                          \
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
  CUmemGenericAllocationHandle* pages;  // physical memory of each page
  uint8_t* used;     // page currently backs a segment of a pool
  uint8_t* skip;     // per block: declared empty for this loan (holds nothing, never backed up)
  uint8_t* touched;  // per block: a page of it was lent during the current loan
  // Lazy backup: before a non-empty block is first lent, it is copied to `backup` on `copy_stream`, and every stream
  // that asks for memory in it waits for that copy (its event).
  char* backup;
  CUevent* events;
  CUstream copy_stream;
  int dropped;  // the owner is gone; destroy once no page is used
} Region;

// A segment handed to a pool: `nregion` pages of region `r` (listed in `pages`), then `extra` new device memory.
typedef struct Segment {
  CUdeviceptr ptr;
  int64_t npages, nregion;
  int64_t* pages;
  Region* r;
  CUmemGenericAllocationHandle extra;
  struct Segment* next;
} Segment;

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static Segment* segments;              // all live segments: frees can arrive after the loan that made them ended
static Region* lender[MAX_DEVICES];    // region serving mallocs during the current loan on each device
static int64_t fallback_bytes[MAX_DEVICES], fallback_peak[MAX_DEVICES];

static CUmemAllocationProp device_memory(int device) {
  CUmemAllocationProp prop = {.type = 1, .location = {.type = 1, .id = device}};
  return prop;
}

static void allow_access(CUdeviceptr ptr, size_t size, int device) {
  CUmemAccessDesc access = {.location = {.type = 1, .id = device}, .flags = 3};
  cuMemSetAccess(ptr, size, &access, 1);
}

static void destroy(Region* r) {
  for (int64_t p = 0; p < r->npages && r->pages[p]; p++) {
    cuMemUnmap(r->base + p * PAGE, PAGE);
    cuMemRelease(r->pages[p]);
  }
  if (r->base) cuMemAddressFree(r->base, r->npages * PAGE);
  if (r->copy_stream) cuStreamDestroy(r->copy_stream);
  if (r->events)
    for (int64_t b = 0; b < r->npages / r->pages_per_block; b++) cuEventDestroy(r->events[b]);
  free(r->events), free(r->pages), free(r->used), free(r->skip), free(r->touched), free(r);
}

Region* memlend_create(int device, size_t nbytes, int64_t block_size) {
  // PyTorch's runtime calls use the device's primary context; the calling thread may not have it current yet.
  CUcontext ctx;
  if (cuDevicePrimaryCtxRetain(&ctx, device) != 0 || cuCtxSetCurrent(ctx) != 0) return NULL;
  Region* r = calloc(1, sizeof(Region));
  r->device = device;
  r->npages = nbytes / PAGE;
  r->pages_per_block = block_size / PAGE;
  int64_t nblocks = r->npages / r->pages_per_block;
  r->pages = calloc(r->npages, sizeof(CUmemGenericAllocationHandle));
  r->used = calloc(r->npages, 1);
  r->skip = calloc(nblocks, 1);
  r->touched = calloc(nblocks, 1);
  CUmemAllocationProp prop = device_memory(device);
  if (cuMemAddressReserve(&r->base, nbytes, 0, 0, 0) != 0) {
    r->base = 0;
    destroy(r);
    return NULL;
  }
  for (int64_t p = 0; p < r->npages; p++)
    if (cuMemCreate(&r->pages[p], PAGE, &prop, 0) != 0 || cuMemMap(r->base + p * PAGE, PAGE, 0, r->pages[p], 0) != 0) {
      if (r->pages[p]) cuMemRelease(r->pages[p]), r->pages[p] = 0;
      destroy(r);
      return NULL;
    }
  allow_access(r->base, nbytes, device);
  return r;
}

static int64_t used_pages(Region* r) {
  int64_t n = 0;
  for (int64_t i = 0; i < r->npages; i++) n += r->used[i];
  return n;
}

// The owner dropped the region. If leaked borrower segments still use its pages, free it when the last one goes.
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

// 1 if address p is in a page of r lent to a segment.
int memlend_lends(Region* r, CUdeviceptr p) {
  pthread_mutex_lock(&mu);
  Segment* s = segments;
  while (s && !(s->r == r && s->ptr <= p && p < s->ptr + s->nregion * PAGE)) s = s->next;
  pthread_mutex_unlock(&mu);
  return s != NULL;
}

// Up to n unlent pages of r, cheapest first: pages of blocks declared empty or already lent in this loan, then pages of
// blocks holding data, so that these are taken whole. Each group from the far end of the region.
static int64_t pick_pages(Region* r, int64_t n, int64_t* out) {
  int64_t k = 0;
  for (int costly = 0; costly < 2; costly++)
    for (int64_t p = r->npages - 1; p >= 0 && k < n; p--) {
      int64_t b = p / r->pages_per_block;
      int holds_data = !r->skip[b] && !r->touched[b];
      if (!r->used[p] && holds_data == costly) out[k++] = p;
    }
  return k;
}

void* memlend_malloc(ssize_t size, int device, void* stream) {
  pthread_mutex_lock(&mu);
  Region* r = lender[device];
  Segment* s = calloc(1, sizeof(Segment));
  s->r = r;
  s->npages = (size + PAGE - 1) / PAGE;
  s->pages = malloc(s->npages * sizeof(int64_t));
  s->nregion = r ? pick_pages(r, s->npages, s->pages) : 0;
  size_t extra = (s->npages - s->nregion) * PAGE;
  CUmemAllocationProp prop = device_memory(device);
  if ((extra && cuMemCreate(&s->extra, extra, &prop, 0) != 0) || cuMemAddressReserve(&s->ptr, s->npages * PAGE, 0, 0, 0) != 0) {
    if (extra && s->extra) cuMemRelease(s->extra);
    free(s->pages), free(s);
    pthread_mutex_unlock(&mu);
    return NULL;
  }
  int64_t last = -1;
  for (int64_t i = 0; i < s->nregion; i++) {
    int64_t p = s->pages[i], b = p / r->pages_per_block;
    r->used[p] = 1;
    cuMemMap(s->ptr + i * PAGE, PAGE, 0, r->pages[p], 0);
    if (b == last) continue;  // pages of a block are picked one after another
    last = b;
    if (r->backup && !r->skip[b]) {
      if (!r->touched[b]) {
        size_t bs = r->pages_per_block * PAGE;
        cuMemcpyDtoHAsync(r->backup + b * bs, r->base + b * bs, bs, r->copy_stream);
        cuEventRecord(r->events[b], r->copy_stream);
      }
      cuStreamWaitEvent((CUstream)stream, r->events[b], 0);  // the borrower's kernels on this stream follow the copy
    }
    r->touched[b] = 1;
  }
  if (extra) cuMemMap(s->ptr + s->nregion * PAGE, extra, 0, s->extra, 0);
  allow_access(s->ptr, s->npages * PAGE, device);
  fallback_bytes[device] += extra;
  if (fallback_bytes[device] > fallback_peak[device]) fallback_peak[device] = fallback_bytes[device];
  s->next = segments;
  segments = s;
  pthread_mutex_unlock(&mu);
  return (void*)s->ptr;
}

void memlend_free(void* ptr, size_t size, int device, void* stream) {
  pthread_mutex_lock(&mu);
  Segment** link = &segments;
  while ((*link)->ptr != (CUdeviceptr)ptr) link = &(*link)->next;
  Segment* s = *link;
  *link = s->next;
  cuCtxSynchronize();  // unmapping does not wait for kernels still using the memory
  for (int64_t i = 0; i < s->nregion; i++) {
    cuMemUnmap(s->ptr + i * PAGE, PAGE);
    s->r->used[s->pages[i]] = 0;
  }
  if (s->nregion < s->npages) {
    cuMemUnmap(s->ptr + s->nregion * PAGE, (s->npages - s->nregion) * PAGE);
    cuMemRelease(s->extra);
    fallback_bytes[device] -= (s->npages - s->nregion) * PAGE;
  }
  cuMemAddressFree(s->ptr, s->npages * PAGE);
  if (s->r && s->r->dropped && !used_pages(s->r)) destroy(s->r);
  free(s->pages), free(s);
  pthread_mutex_unlock(&mu);
}
