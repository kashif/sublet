// Segment allocator behind the memory pool of a lendable tensor (see lend.py).
//
// The tensor's memory (the region) is one address range backed by one physical 2 MiB page per page of the range. The
// caching allocator asks for whole segments (multiples of 2 MiB); sublet_malloc gives each one a fresh address range
// backed by the cheapest memory first: region pages that cost no copy (not kept, or already lent in this loan), then
// free device memory, then kept region pages, which are copied to pinned CPU memory first and back when the loan ends.
// Segments stay mapped from one loan to the next, so later loans reuse them for free. The free device memory they hold
// is given back when a loan ends and taken again when the next one starts; if it is gone by then, region pages fill in.
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
  X(cuMemGetInfo, "cuMemGetInfo_v2", size_t*, size_t*)                                                            \
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
int sublet_init(void) {
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
  int64_t nsegments;                // segments made for r and not freed yet
  int dropped;                      // the owner is gone; destroy once no segment is left
} Region;

// A segment handed to a pool: its first `nextra` pages are device memory of its own (`extra`, 0 between loans), the
// others are pages of region `r` (`slot[i]` is the region page behind page i).
typedef struct Segment {
  CUdeviceptr ptr;
  int64_t npages, nextra;
  int64_t* slot;
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

Region* sublet_create(int device, size_t nbytes) {
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

// The owner dropped the region. If leaked segments are still alive, free it when the last one goes.
void sublet_destroy(Region* r) {
  pthread_mutex_lock(&mu);
  if (r->nsegments)
    r->dropped = 1;
  else
    destroy(r);
  pthread_mutex_unlock(&mu);
}

CUdeviceptr sublet_base(Region* r) { return r->base; }
int64_t sublet_extra_peak(Region* r) { return r->extra_peak; }

// For inspection: one byte per page, set if the page was lent during the current loan (or the last one, after it).
void sublet_touched(Region* r, uint8_t* out) {
  pthread_mutex_lock(&mu);
  memcpy(out, r->touched, r->npages);
  pthread_mutex_unlock(&mu);
}

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

// Up to n unlent pages of r from the far end: pages that cost no copy (not kept, or already lent in this loan), then,
// with_kept, kept pages.
static int64_t pick_pages(Region* r, int64_t n, int64_t* out, int with_kept) {
  int64_t k = 0;
  for (int kept = 0; kept <= with_kept; kept++)
    for (int64_t p = r->npages - 1; p >= 0 && k < n; p--)
      if (!r->used[p] && (r->keep[p] && !r->touched[p]) == kept) out[k++] = p;
  return k;
}

// Up to n pages of free device memory as s->extra; returns how many.
static int64_t new_memory(Region* r, Segment* s, int64_t n) {
  size_t free_bytes = 0, total = 0;
  cuMemGetInfo(&free_bytes, &total);
  int64_t k = n < (int64_t)(free_bytes / PAGE) ? n : (int64_t)(free_bytes / PAGE);
  CUmemAllocationProp prop = device_memory(r->device);
  while (k > 0 && cuMemCreate(&s->extra, k * PAGE, &prop, 0) != 0) k = k * 3 / 4;  // part of the free memory is reserved
  if (k == 0) s->extra = 0;
  return k;
}

// Backs pages [0, n) of s with the cheapest memory: region pages that cost no copy, then free device memory (the first
// pages), then kept region pages, copied to the CPU first. Returns -1 if the GPU and the region run out.
static int back(Region* r, Segment* s, int64_t n) {
  int64_t* picked = malloc(n * sizeof(int64_t));
  int64_t cheap = pick_pages(r, n, picked, 0);
  for (int64_t i = 0; i < cheap; i++) r->used[picked[i]] = 1;  // so that the second pick skips them
  int64_t nextra = new_memory(r, s, n - cheap);
  int64_t kept = pick_pages(r, n - cheap - nextra, picked + cheap, 1);
  for (int64_t i = 0; i < cheap + kept; i++) r->want[picked[i]] = r->keep[picked[i]] && !r->touched[picked[i]];
  if (cheap + nextra + kept < n || copy_wanted(r, 1, r->copy_stream) < 0) {
    memset(r->want, 0, r->npages);
    for (int64_t i = 0; i < cheap; i++) r->used[picked[i]] = 0;
    if (nextra) cuMemRelease(s->extra), s->extra = 0;
    free(picked);
    return -1;
  }
  cuStreamSynchronize(r->copy_stream);  // the copies are done before the borrower gets the memory
  for (int64_t i = 0; i < n - nextra; i++) {
    s->slot[nextra + i] = picked[i];
    cuMemMap(s->ptr + (nextra + i) * PAGE, PAGE, 0, r->pages[picked[i]], 0);
    r->used[picked[i]] = r->touched[picked[i]] = 1;
  }
  if (nextra) cuMemMap(s->ptr, nextra * PAGE, 0, s->extra, 0);
  allow_access(s->ptr, n * PAGE, r->device);
  s->nextra = nextra;
  r->extra_bytes += nextra * PAGE;
  if (r->extra_bytes > r->extra_peak) r->extra_peak = r->extra_bytes;
  free(picked);
  return 0;
}

// Start a loan of r; keep has one byte per page. The pages segments already hold count as lent from the start: the kept
// ones are copied out now, and the device memory segments gave back is taken again. Returns -1 if that memory and the
// region both run out, or pinned CPU memory does.
int sublet_begin(Region* r, const uint8_t* keep) {
  pthread_mutex_lock(&mu);
  CUcontext previous = enter(r);
  memcpy(r->keep, keep, r->npages);
  memcpy(r->touched, r->used, r->npages);
  for (int64_t p = 0; p < r->npages; p++) r->want[p] = r->used[p] && r->keep[p];
  int ok = copy_wanted(r, 1, r->copy_stream) >= 0;
  cuStreamSynchronize(r->copy_stream);
  r->extra_peak = r->extra_bytes;
  for (Segment* s = segments; s && ok; s = s->next)
    if (s->r == r && s->nextra && !s->extra) ok = back(r, s, s->nextra) == 0;
  if (ok) lender[r->device] = r;
  cuCtxSetCurrent(previous);
  pthread_mutex_unlock(&mu);
  return ok ? 0 : -1;
}

// End the loan of r: copies the kept pages it lent back on stream and waits for them, then, with give_back, gives back
// the device memory its segments hold beyond the region. Returns the bytes copied back.
int64_t sublet_end(Region* r, void* stream, int give_back) {
  pthread_mutex_lock(&mu);
  CUcontext previous = enter(r);
  lender[r->device] = NULL;
  for (int64_t p = 0; p < r->npages; p++) r->want[p] = r->touched[p] && r->keep[p];
  int64_t copied = copy_wanted(r, 0, (CUstream)stream);
  cuStreamSynchronize((CUstream)stream);
  for (Segment* s = segments; s && give_back; s = s->next)
    if (s->r == r && s->extra) {
      cuMemUnmap(s->ptr, s->nextra * PAGE);
      cuMemRelease(s->extra);
      s->extra = 0;
      r->extra_bytes -= s->nextra * PAGE;
    }
  cuCtxSetCurrent(previous);
  pthread_mutex_unlock(&mu);
  return copied * PAGE;
}

void* sublet_malloc(ssize_t size, int device, void* stream) {
  (void)stream;
  pthread_mutex_lock(&mu);
  Region* r = lender[device];
  Segment* s = r ? calloc(1, sizeof(Segment)) : NULL;  // the pool is only used during loans
  if (s) {
    s->r = r;
    s->npages = (size + PAGE - 1) / PAGE;
    s->slot = malloc(s->npages * sizeof(int64_t));
    if (cuMemAddressReserve(&s->ptr, s->npages * PAGE, 0, 0, 0) != 0) {
      free(s->slot), free(s), s = NULL;
    } else if (back(r, s, s->npages) != 0) {
      cuMemAddressFree(s->ptr, s->npages * PAGE);
      free(s->slot), free(s), s = NULL;
    } else {
      s->next = segments;
      segments = s;
      r->nsegments++;
    }
  }
  pthread_mutex_unlock(&mu);
  return s ? (void*)s->ptr : NULL;
}

void sublet_free(void* ptr, size_t size, int device, void* stream) {
  (void)size, (void)device, (void)stream;
  pthread_mutex_lock(&mu);
  Segment** link = &segments;
  while ((*link)->ptr != (CUdeviceptr)ptr) link = &(*link)->next;
  Segment* s = *link;
  *link = s->next;
  CUcontext previous = enter(s->r);
  cuCtxSynchronize();  // unmapping does not wait for kernels still using the memory
  if (s->extra) {
    cuMemUnmap(s->ptr, s->nextra * PAGE);
    cuMemRelease(s->extra);
    s->r->extra_bytes -= s->nextra * PAGE;
  }
  if (s->nextra < s->npages) cuMemUnmap(s->ptr + s->nextra * PAGE, (s->npages - s->nextra) * PAGE);
  for (int64_t i = s->nextra; i < s->npages; i++) s->r->used[s->slot[i]] = 0;
  cuMemAddressFree(s->ptr, s->npages * PAGE);
  cuCtxSetCurrent(previous);
  if (--s->r->nsegments == 0 && s->r->dropped) destroy(s->r);
  free(s->slot), free(s);
  pthread_mutex_unlock(&mu);
}
