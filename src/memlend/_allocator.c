// Segment allocator behind the memory pool of a lendable tensor (see lend.py).
//
// The tensor's memory (the region) is one address range backed by one physical 2 MiB page per page of the range. The
// caching allocator asks for whole segments (multiples of 2 MiB); memlend_malloc gives each one a fresh address range
// and maps unlent pages of the region behind it, side by side, wherever they sit: first pages the owner does not keep
// or that were already lent in this loan, then kept pages, each from the far end of the region. Kept pages are copied to
// pinned CPU memory before they are first lent, and back when the loan ends. Once the region has no unlent page left,
// the rest of the segment is new device memory. Segments stay mapped until the caching allocator releases them, so later
// loans reuse them for free; a pool holding new device memory is dropped at the end of the loan (see lend.py).
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

#define DRIVER_FUNCTIONS(X)                                                                                         \
  X(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain", CUcontext*, int)                                         \
  X(cuCtxGetCurrent, "cuCtxGetCurrent", CUcontext*)                                                                \
  X(cuCtxSetCurrent, "cuCtxSetCurrent", CUcontext)                                                                 \
  X(cuCtxSynchronize, "cuCtxSynchronize")                                                                          \
  X(cuMemAddressReserve, "cuMemAddressReserve", CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long)     \
  X(cuMemAddressFree, "cuMemAddressFree", CUdeviceptr, size_t)                                                     \
  X(cuMemCreate, "cuMemCreate", CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long) \
  X(cuMemRelease, "cuMemRelease", CUmemGenericAllocationHandle)                                                    \
  X(cuMemMap, "cuMemMap", CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long)           \
  X(cuMemUnmap, "cuMemUnmap", CUdeviceptr, size_t)                                                                 \
  X(cuMemSetAccess, "cuMemSetAccess", CUdeviceptr, size_t, const CUmemAccessDesc*, size_t)                         \
  X(cuMemHostAlloc, "cuMemHostAlloc", void**, size_t, unsigned int)                                                \
  X(cuMemFreeHost, "cuMemFreeHost", void*)                                                                         \
  X(cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2", void*, CUdeviceptr, size_t, CUstream)                               \
  X(cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2", CUdeviceptr, const void*, size_t, CUstream)                         \
  X(cuStreamCreate, "cuStreamCreate", CUstream*, unsigned int)                                                     \
  X(cuStreamDestroy, "cuStreamDestroy_v2", CUstream)                                                               \
  X(cuStreamSynchronize, "cuStreamSynchronize", CUstream)


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
#define CHUNK 32  // pages per chunk of pinned CPU memory (64 MiB)
#define MAX_DEVICES 64

typedef struct Region {
  int device;
  CUcontext ctx;  // the device's primary context, which PyTorch uses
  CUdeviceptr base;
  int64_t npages;
  CUmemGenericAllocationHandle* pages;  // physical memory of each page
  uint8_t* used;                        // page backs a segment
  uint8_t* keep;                        // page holds data the owner gets back after the current loan
  uint8_t* touched;                     // page was lent during the current loan
  uint8_t* want;                        // scratch: pages to copy
  char** chunks;                        // pinned CPU copy of each chunk of pages, allocated when first needed
  CUstream copy_stream;
  int64_t extra_bytes, extra_peak;  // new device memory in segments; peak during the current loan
  int dropped;                      // the owner is gone; destroy once no page is used
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
static Segment* segments;            // all live segments
static Region* lender[MAX_DEVICES];  // region lent on each device, during a loan

static CUmemAllocationProp device_memory(int device) {
  CUmemAllocationProp prop = {.type = 1, .location = {.type = 1, .id = device}};
  return prop;
}

static void allow_access(CUdeviceptr ptr, size_t size, int device) {
  CUmemAccessDesc access = {.location = {.type = 1, .id = device}, .flags = 3};
  cuMemSetAccess(ptr, size, &access, 1);
}

// Makes r's device current on the calling thread, which may be using another one; returns the context to restore.
static CUcontext enter(Region* r) {
  CUcontext previous = NULL;
  cuCtxGetCurrent(&previous);
  cuCtxSetCurrent(r->ctx);
  return previous;
}

static void destroy(Region* r) {
  CUcontext previous = enter(r);
  for (int64_t p = 0; p < r->npages && r->pages[p]; p++) {
    cuMemUnmap(r->base + p * PAGE, PAGE);
    cuMemRelease(r->pages[p]);
  }
  if (r->base) cuMemAddressFree(r->base, r->npages * PAGE);
  for (int64_t c = 0; c * CHUNK < r->npages; c++)
    if (r->chunks[c]) cuMemFreeHost(r->chunks[c]);
  if (r->copy_stream) cuStreamDestroy(r->copy_stream);
  cuCtxSetCurrent(previous);
  free(r->pages), free(r->used), free(r->keep), free(r->touched), free(r->want), free(r->chunks), free(r);
}

Region* memlend_create(int device, size_t nbytes) {
  CUcontext ctx;
  if (cuDevicePrimaryCtxRetain(&ctx, device) != 0) return NULL;
  Region* r = calloc(1, sizeof(Region));
  r->device = device;
  r->ctx = ctx;
  CUcontext previous = enter(r);
  r->npages = nbytes / PAGE;
  r->pages = calloc(r->npages, sizeof(CUmemGenericAllocationHandle));
  r->used = calloc(r->npages, 1);
  r->keep = calloc(r->npages, 1);
  r->touched = calloc(r->npages, 1);
  r->want = calloc(r->npages, 1);
  r->chunks = calloc((r->npages + CHUNK - 1) / CHUNK, sizeof(char*));
  CUmemAllocationProp prop = device_memory(device);
  int failed = cuStreamCreate(&r->copy_stream, 1 /* CU_STREAM_NON_BLOCKING */) != 0 ||
               cuMemAddressReserve(&r->base, nbytes, 0, 0, 0) != 0;
  if (failed) r->base = 0;
  for (int64_t p = 0; p < r->npages && !failed; p++)
    if (cuMemCreate(&r->pages[p], PAGE, &prop, 0) != 0 || cuMemMap(r->base + p * PAGE, PAGE, 0, r->pages[p], 0) != 0) {
      if (r->pages[p]) cuMemRelease(r->pages[p]), r->pages[p] = 0;
      failed = 1;
    }
  if (!failed) allow_access(r->base, nbytes, device);
  cuCtxSetCurrent(previous);
  if (failed) destroy(r), r = NULL;
  return r;
}

static int64_t used_pages(Region* r) {
  int64_t n = 0;
  for (int64_t i = 0; i < r->npages; i++) n += r->used[i];
  return n;
}

// The owner dropped the region. If leaked segments still use its pages, free it when the last one goes.
void memlend_destroy(Region* r) {
  pthread_mutex_lock(&mu);
  if (used_pages(r))
    r->dropped = 1;
  else
    destroy(r);
  pthread_mutex_unlock(&mu);
}

CUdeviceptr memlend_base(Region* r) { return r->base; }
int64_t memlend_extra_bytes(Region* r) { return r->extra_bytes; }
int64_t memlend_extra_peak(Region* r) { return r->extra_peak; }

// Copies the pages in r->want to (to_host) or from their pinned CPU copy on stream, then clears r->want. Returns the
// number of pages, or -1 if pinned memory for one of them cannot be allocated.
static int64_t copy_wanted(Region* r, int to_host, CUstream stream) {
  int64_t n = 0;
  for (int64_t p = 0; p < r->npages;) {
    if (!r->want[p]) {
      p++;
      continue;
    }
    int64_t c = p / CHUNK, q = p;
    while (q < r->npages && r->want[q] && q / CHUNK == c) r->want[q++] = 0;
    if (!r->chunks[c] && cuMemHostAlloc((void**)&r->chunks[c], CHUNK * PAGE, 0) != 0) {
      r->chunks[c] = NULL;
      memset(r->want, 0, r->npages);
      return -1;
    }
    char* host = r->chunks[c] + (p % CHUNK) * PAGE;
    if (to_host)
      cuMemcpyDtoHAsync(host, r->base + p * PAGE, (q - p) * PAGE, stream);
    else
      cuMemcpyHtoDAsync(r->base + p * PAGE, host, (q - p) * PAGE, stream);
    n += q - p;
    p = q;
  }
  return n;
}

// Start a loan of r; keep has one byte per page. The pages segments already hold count as lent from the start: the kept
// ones are copied out now. Returns -1 if pinned CPU memory runs out.
int memlend_begin(Region* r, const uint8_t* keep) {
  pthread_mutex_lock(&mu);
  CUcontext previous = enter(r);
  memcpy(r->keep, keep, r->npages);
  memcpy(r->touched, r->used, r->npages);
  for (int64_t p = 0; p < r->npages; p++) r->want[p] = r->used[p] && r->keep[p];
  int64_t copied = copy_wanted(r, 1, r->copy_stream);
  cuStreamSynchronize(r->copy_stream);
  r->extra_peak = r->extra_bytes;
  if (copied >= 0) lender[r->device] = r;
  cuCtxSetCurrent(previous);
  pthread_mutex_unlock(&mu);
  return copied < 0 ? -1 : 0;
}

// End the loan of r: copies the kept pages it lent back on stream and waits for them. Returns the bytes copied back.
int64_t memlend_end(Region* r, void* stream) {
  pthread_mutex_lock(&mu);
  CUcontext previous = enter(r);
  lender[r->device] = NULL;
  for (int64_t p = 0; p < r->npages; p++) r->want[p] = r->touched[p] && r->keep[p];
  int64_t copied = copy_wanted(r, 0, (CUstream)stream);
  cuStreamSynchronize((CUstream)stream);
  cuCtxSetCurrent(previous);
  pthread_mutex_unlock(&mu);
  return copied * PAGE;
}

// Up to n unlent pages of r: first pages not kept or already lent in this loan, then kept pages; each from the far end.
static int64_t pick_pages(Region* r, int64_t n, int64_t* out) {
  int64_t k = 0;
  for (int kept = 0; kept < 2; kept++)
    for (int64_t p = r->npages - 1; p >= 0 && k < n; p--)
      if (!r->used[p] && (r->keep[p] && !r->touched[p]) == kept) out[k++] = p;
  return k;
}

void* memlend_malloc(ssize_t size, int device, void* stream) {
  (void)stream;
  pthread_mutex_lock(&mu);
  Region* r = lender[device];
  Segment* s = r ? calloc(1, sizeof(Segment)) : NULL;  // the pool is only used during loans
  if (!s) {
    pthread_mutex_unlock(&mu);
    return NULL;
  }
  s->r = r;
  s->npages = (size + PAGE - 1) / PAGE;
  s->pages = malloc(s->npages * sizeof(int64_t));
  s->nregion = pick_pages(r, s->npages, s->pages);
  for (int64_t i = 0; i < s->nregion; i++) r->want[s->pages[i]] = r->keep[s->pages[i]] && !r->touched[s->pages[i]];
  int64_t copied = copy_wanted(r, 1, r->copy_stream);
  size_t extra = (s->npages - s->nregion) * PAGE;
  CUmemAllocationProp prop = device_memory(device);
  if (copied < 0 || (extra && cuMemCreate(&s->extra, extra, &prop, 0) != 0) ||
      cuMemAddressReserve(&s->ptr, s->npages * PAGE, 0, 0, 0) != 0) {
    if (copied >= 0 && extra && s->extra) cuMemRelease(s->extra);
    free(s->pages), free(s);
    pthread_mutex_unlock(&mu);
    return NULL;
  }
  cuStreamSynchronize(r->copy_stream);  // the copies are done before the borrower gets the memory
  for (int64_t i = 0; i < s->nregion; i++) {
    cuMemMap(s->ptr + i * PAGE, PAGE, 0, r->pages[s->pages[i]], 0);
    r->used[s->pages[i]] = r->touched[s->pages[i]] = 1;
  }
  if (extra) cuMemMap(s->ptr + s->nregion * PAGE, extra, 0, s->extra, 0);
  allow_access(s->ptr, s->npages * PAGE, device);
  r->extra_bytes += extra;
  if (r->extra_bytes > r->extra_peak) r->extra_peak = r->extra_bytes;
  s->next = segments;
  segments = s;
  pthread_mutex_unlock(&mu);
  return (void*)s->ptr;
}

void memlend_free(void* ptr, size_t size, int device, void* stream) {
  (void)size, (void)device, (void)stream;
  pthread_mutex_lock(&mu);
  Segment** link = &segments;
  while ((*link)->ptr != (CUdeviceptr)ptr) link = &(*link)->next;
  Segment* s = *link;
  *link = s->next;
  CUcontext previous = enter(s->r);
  cuCtxSynchronize();  // unmapping does not wait for kernels still using the memory
  cuMemUnmap(s->ptr, s->npages * PAGE);
  for (int64_t i = 0; i < s->nregion; i++) s->r->used[s->pages[i]] = 0;
  if (s->nregion < s->npages) {
    cuMemRelease(s->extra);
    s->r->extra_bytes -= (s->npages - s->nregion) * PAGE;
  }
  cuMemAddressFree(s->ptr, s->npages * PAGE);
  cuCtxSetCurrent(previous);
  if (s->r->dropped && !used_pages(s->r)) destroy(s->r);
  free(s->pages), free(s);
  pthread_mutex_unlock(&mu);
}
