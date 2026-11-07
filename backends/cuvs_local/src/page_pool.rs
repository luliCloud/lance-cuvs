//! Pools of pinned host memory handed to Lance so that its large host buffers do not go through
//! the global allocator. A request takes one buffer from the pool and falls back to the heap when
//! the pool cannot serve it.
//!
//! * page pool: the ~32 MiB `O_DIRECT` page-read buffers of lance-io.
//!   `LANCE_CUVS_PAGE_POOL_GIB` (unset/0 = off), `LANCE_CUVS_PAGE_SLOT_MIB` (default 40).
//! * batch pool: the contiguous buffer a decoded batch is assembled into from its pages
//!   (lance-encoding), the buffer that is later copied to the device. Slots must hold a whole
//!   batch (`batch_size` rows x dim x 4 bytes, 1 GiB for 128K rows of 2048 floats), and the pool
//!   must cover every batch alive at once. `LANCE_CUVS_BATCH_POOL_GIB` (unset/0 = off),
//!   `LANCE_CUVS_BATCH_SLOT_MIB` (default 1024).
//!
//! Two backends serve either pool:
//!
//! * fixed slots (default): one `cudaMallocHost` region cut into fixed-size slots, one slot per
//!   request.
//! * RMM (`LANCE_CUVS_PAGE_POOL_RMM=1`): one process-wide cuVS RMM pinned host pool
//!   (`cuvsRMMPinnedHostPoolEnable`, a `pool_memory_resource` over `pinned_host_memory_resource`
//!   with initial size == maximum size, so no `cudaHostAlloc` happens during the run). Both pools
//!   draw from it, so its size is the sum of the two `*_POOL_GIB` values and the `*_SLOT_MIB`
//!   values are ignored: a request takes exactly the bytes it needs. The pool only guarantees
//!   256-byte alignment, so page-pool requests (`O_DIRECT` needs 4096) are over-allocated by one
//!   page and realigned; batch-pool requests are used as returned.
//!
//! Slices inside either pool are already page-locked, see [`contains`].
//!
//! Pinning takes about 0.3 s per GiB at shuffle start, so size the pools to what is in use at
//! once (`max_in_use_gib` in the report) rather than generously.

use crate::cuda::{PinnedHostBuffer, check_cuvs};
use lance_core::Result;
use lance_encoding::output_pool::{
    DecodeOutputAllocator, OutputBuf, set_decode_output_allocator,
};
use lance_io::buffer_pool::{DirectBufAllocator, PoolBuf, set_direct_buf_allocator};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const ALIGN: usize = 4096;

/// Address ranges (`[start, end)`) of the active pools, one entry per pool kind; empty when off.
static POOL_RANGES: [(AtomicUsize, AtomicUsize); 2] = [
    (AtomicUsize::new(0), AtomicUsize::new(0)),
    (AtomicUsize::new(0), AtomicUsize::new(0)),
];

/// True if `[ptr, ptr + len)` lies inside a pool's pinned region. Such memory is already
/// page-locked, so it must not be passed to `cudaHostRegister` (which fails with
/// `cudaErrorHostMemoryAlreadyRegistered`) and can be copied to the device directly.
pub(crate) fn contains(ptr: usize, len: usize) -> bool {
    POOL_RANGES.iter().any(|(start, end)| {
        let end = end.load(Ordering::Acquire);
        let start = start.load(Ordering::Acquire);
        ptr >= start && ptr.saturating_add(len) <= end
    })
}

fn set_range(range_index: usize, start: usize, end: usize) {
    let (range_start, range_end) = &POOL_RANGES[range_index];
    range_start.store(start, Ordering::Release);
    range_end.store(end, Ordering::Release);
}

/// The process-wide RMM pinned host pool, shared by every pool kind that uses the RMM backend.
/// Disabled when the last pool, and with it the last outstanding buffer, is gone.
struct RmmShared {
    total_bytes: usize,
    /// A range known to lie inside the pool's pinned memory, see [`probe_range`].
    range: Option<(usize, usize)>,
}

impl RmmShared {
    fn enable(bytes: usize) -> Result<Arc<Self>> {
        let start = std::time::Instant::now();
        check_cuvs(
            unsafe { cuvs_sys::cuvsRMMPinnedHostPoolEnable(bytes) },
            "enable the cuVS RMM pinned host pool",
        )?;
        let range = probe_range(bytes);
        eprintln!(
            "cuVS pinned pool (RMM): enabled, {:.1} GiB, initial == maximum (alloc {:.1}s), \
             registration skip range {}",
            bytes as f64 / (1u64 << 30) as f64,
            start.elapsed().as_secs_f64(),
            match range {
                Some((s, e)) => format!("{:.1} GiB", (e - s) as f64 / (1u64 << 30) as f64),
                None => "none (registration is not skipped for pool memory)".to_string(),
            }
        );
        Ok(Arc::new(Self {
            total_bytes: bytes,
            range,
        }))
    }
}

impl Drop for RmmShared {
    fn drop(&mut self) {
        // Runs after the last pool and every outstanding buffer is gone, so the pool is only
        // released after the last cuvsRMMHostFree.
        let _ = unsafe { cuvs_sys::cuvsRMMPinnedHostPoolDisable() };
    }
}

/// Finds an address range that lies inside the pool's pinned memory, so that [`contains`] works
/// for RMM-backed buffers. The pool is created with initial size == maximum size, so it owns a
/// single contiguous upstream block, and one successful allocation of (almost) the whole pool
/// can only have come from that block: the whole range `[ptr, ptr + len)` is pinned. The probe
/// is freed again right away. Backs off from the full size in case the pool cannot hand out
/// every last byte; the bytes outside the range are merely not skipped by [`contains`], which
/// costs a failed `cudaHostRegister` and a fallback, never correctness.
fn probe_range(bytes: usize) -> Option<(usize, usize)> {
    for back_off in [0usize, 1 << 20, 16 << 20, 256 << 20] {
        let len = bytes.checked_sub(back_off)? / ALIGN * ALIGN;
        if len < ALIGN {
            return None;
        }
        let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
        let status = unsafe { cuvs_sys::cuvsRMMHostAlloc(&mut raw, len) };
        if status == cuvs_sys::cuvsError_t::CUVS_SUCCESS {
            let start = raw as usize;
            let _ = unsafe { cuvs_sys::cuvsRMMHostFree(raw, len) };
            return Some((start, start + len));
        }
    }
    None
}

enum Backend {
    Slots {
        // Owns the pinned region; only touched on drop.
        _region: Mutex<PinnedHostBuffer<u8>>,
        base: usize,
        slot_bytes: usize,
        free: Mutex<Vec<usize>>,
        slots: usize,
    },
    Rmm(Arc<RmmShared>),
}

type Release = Box<dyn FnOnce() + Send + Sync>;

pub(crate) struct HostPool {
    label: &'static str,
    range_index: usize,
    /// RMM backend: hand out 4096-aligned buffers (for `O_DIRECT`).
    page_aligned: bool,
    backend: Backend,
    hits: AtomicU64,
    misses_empty: AtomicU64,
    misses_too_big: AtomicU64,
    realigned: AtomicU64,
    max_len: AtomicUsize,
    in_use: AtomicUsize,
    max_in_use: AtomicUsize,
    in_use_bytes: AtomicUsize,
    max_in_use_bytes: AtomicUsize,
}

impl HostPool {
    fn new(
        label: &'static str,
        range_index: usize,
        page_aligned: bool,
        backend: Backend,
    ) -> Arc<Self> {
        Arc::new(Self {
            label,
            range_index,
            page_aligned,
            backend,
            hits: AtomicU64::new(0),
            misses_empty: AtomicU64::new(0),
            misses_too_big: AtomicU64::new(0),
            realigned: AtomicU64::new(0),
            max_len: AtomicUsize::new(0),
            in_use: AtomicUsize::new(0),
            max_in_use: AtomicUsize::new(0),
            in_use_bytes: AtomicUsize::new(0),
            max_in_use_bytes: AtomicUsize::new(0),
        })
    }

    /// Takes a buffer for a request of `len` bytes, with the closure that returns it.
    fn take(self: &Arc<Self>, len: usize) -> Option<(NonNull<u8>, Release)> {
        self.max_len.fetch_max(len, Ordering::Relaxed);
        match &self.backend {
            Backend::Slots {
                base,
                slot_bytes,
                free,
                ..
            } => {
                if len > *slot_bytes {
                    self.misses_too_big.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                let idx = free.lock().expect("slot pool poisoned").pop();
                let Some(idx) = idx else {
                    self.misses_empty.fetch_add(1, Ordering::Relaxed);
                    return None;
                };
                let ptr = NonNull::new((base + idx * slot_bytes) as *mut u8)?;
                self.note_hit(*slot_bytes);
                let pool = self.clone();
                let bytes = *slot_bytes;
                Some((
                    ptr,
                    Box::new(move || {
                        pool.note_release(bytes);
                        if let Backend::Slots { free, .. } = &pool.backend {
                            free.lock().expect("slot pool poisoned").push(idx);
                        }
                    }),
                ))
            }
            Backend::Rmm(_) => {
                // O_DIRECT needs 4096-aligned buffers and the pool only guarantees 256 bytes:
                // ask for one extra page, hand out the first page-aligned address inside the
                // block, and free the original pointer. Batch buffers take the pool's alignment.
                let bytes = if self.page_aligned {
                    len.div_ceil(ALIGN) * ALIGN + ALIGN
                } else {
                    len
                };
                let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
                let status = unsafe { cuvs_sys::cuvsRMMHostAlloc(&mut raw, bytes) };
                if status != cuvs_sys::cuvsError_t::CUVS_SUCCESS {
                    self.misses_empty.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
                let addr = raw as usize;
                let aligned = if self.page_aligned {
                    addr.div_ceil(ALIGN) * ALIGN
                } else {
                    addr
                };
                let Some(ptr) = NonNull::new(aligned as *mut u8) else {
                    let _ = unsafe { cuvs_sys::cuvsRMMHostFree(raw, bytes) };
                    return None;
                };
                if aligned != addr {
                    self.realigned.fetch_add(1, Ordering::Relaxed);
                }
                self.note_hit(bytes);
                let pool = self.clone();
                Some((
                    ptr,
                    Box::new(move || {
                        pool.note_release(bytes);
                        let _ = unsafe { cuvs_sys::cuvsRMMHostFree(addr as *mut _, bytes) };
                    }),
                ))
            }
        }
    }

    fn note_hit(&self, bytes: usize) {
        self.hits.fetch_add(1, Ordering::Relaxed);
        let now = self.in_use.fetch_add(1, Ordering::Relaxed) + 1;
        self.max_in_use.fetch_max(now, Ordering::Relaxed);
        let now_bytes = self.in_use_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.max_in_use_bytes.fetch_max(now_bytes, Ordering::Relaxed);
    }

    fn note_release(&self, bytes: usize) {
        self.in_use.fetch_sub(1, Ordering::Relaxed);
        self.in_use_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    fn report(&self) {
        let config = match &self.backend {
            Backend::Slots {
                slot_bytes, slots, ..
            } => format!("slots={} slot_mib={}", slots, slot_bytes >> 20),
            Backend::Rmm(shared) => format!(
                "rmm_total_gib={:.1}",
                shared.total_bytes as f64 / (1u64 << 30) as f64
            ),
        };
        eprintln!(
            "cuVS {} pool: {} hits={} misses_empty={} misses_too_big={} realigned={} \
             max_in_use={} max_in_use_gib={:.2} max_request_mib={:.2}",
            self.label,
            config,
            self.hits.load(Ordering::Relaxed),
            self.misses_empty.load(Ordering::Relaxed),
            self.misses_too_big.load(Ordering::Relaxed),
            self.realigned.load(Ordering::Relaxed),
            self.max_in_use.load(Ordering::Relaxed),
            self.max_in_use_bytes.load(Ordering::Relaxed) as f64 / (1u64 << 30) as f64,
            self.max_len.load(Ordering::Relaxed) as f64 / (1u64 << 20) as f64,
        );
    }

    fn clear_range(&self) {
        let (start, end) = &POOL_RANGES[self.range_index];
        end.store(0, Ordering::Release);
        start.store(0, Ordering::Release);
    }
}

struct PageHandle(Arc<HostPool>);

impl DirectBufAllocator for PageHandle {
    fn alloc(&self, len: usize) -> Option<PoolBuf> {
        let (ptr, release) = self.0.take(len)?;
        Some(PoolBuf { ptr, release })
    }
}

struct BatchHandle(Arc<HostPool>);

impl DecodeOutputAllocator for BatchHandle {
    fn alloc(&self, len: usize) -> Option<OutputBuf> {
        let (ptr, release) = self.0.take(len)?;
        Some(OutputBuf { ptr, release })
    }
}

/// Registers the pools with lance-io / lance-encoding for their lifetime; unregisters on drop.
pub(crate) struct PagePoolGuard {
    page: Option<Arc<HostPool>>,
    batch: Option<Arc<HostPool>>,
}

impl Drop for PagePoolGuard {
    fn drop(&mut self) {
        if let Some(pool) = &self.page {
            set_direct_buf_allocator(None);
            pool.clear_range();
            pool.report();
        }
        if let Some(pool) = &self.batch {
            set_decode_output_allocator(None);
            pool.clear_range();
            pool.report();
        }
    }
}

fn env_f64(name: &str) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn create_slot_pool(
    label: &'static str,
    range_index: usize,
    gib: f64,
    slot_mib: usize,
) -> Result<Arc<HostPool>> {
    let slot_bytes = (slot_mib << 20).div_ceil(ALIGN) * ALIGN;
    let slots = ((gib * (1u64 << 30) as f64) as usize / slot_bytes).max(1);
    let start = std::time::Instant::now();
    // One extra page of slack so the first slot can be aligned to 4096.
    let mut region = PinnedHostBuffer::<u8>::try_new(slots * slot_bytes + ALIGN)?;
    let raw = region.as_mut_slice().as_mut_ptr() as usize;
    let base = raw.div_ceil(ALIGN) * ALIGN;
    let pool = HostPool::new(
        label,
        range_index,
        true,
        Backend::Slots {
            _region: Mutex::new(region),
            base,
            slot_bytes,
            free: Mutex::new((0..slots).rev().collect()),
            slots,
        },
    );
    set_range(range_index, base, base + slots * slot_bytes);
    eprintln!(
        "cuVS {label} pool: enabled, {} x {} MiB pinned = {:.1} GiB (alloc {:.1}s)",
        slots,
        slot_bytes >> 20,
        (slots * slot_bytes) as f64 / (1u64 << 30) as f64,
        start.elapsed().as_secs_f64()
    );
    Ok(pool)
}

fn create_rmm_pool(
    label: &'static str,
    range_index: usize,
    page_aligned: bool,
    shared: &Arc<RmmShared>,
) -> Arc<HostPool> {
    let pool = HostPool::new(
        label,
        range_index,
        page_aligned,
        Backend::Rmm(shared.clone()),
    );
    if let Some((start, end)) = shared.range {
        set_range(range_index, start, end);
    }
    eprintln!("cuVS {label} pool: enabled, backed by the RMM pinned host pool");
    pool
}

pub(crate) fn page_pool_from_env() -> Result<Option<PagePoolGuard>> {
    let page_gib = env_f64("LANCE_CUVS_PAGE_POOL_GIB");
    let batch_gib = env_f64("LANCE_CUVS_BATCH_POOL_GIB");
    if !(page_gib > 0.0) && !(batch_gib > 0.0) {
        return Ok(None);
    }
    let use_rmm = std::env::var("LANCE_CUVS_PAGE_POOL_RMM")
        .map(|v| v == "1")
        .unwrap_or(false);
    let shared = if use_rmm {
        // One pool for everything, rounded down to a page.
        let total = (page_gib.max(0.0) + batch_gib.max(0.0)) * (1u64 << 30) as f64;
        Some(RmmShared::enable(total as usize / ALIGN * ALIGN)?)
    } else {
        None
    };
    let mut guard = PagePoolGuard {
        page: None,
        batch: None,
    };
    if page_gib > 0.0 {
        let pool = match &shared {
            Some(shared) => create_rmm_pool("page", 0, true, shared),
            None => create_slot_pool(
                "page",
                0,
                page_gib,
                env_usize("LANCE_CUVS_PAGE_SLOT_MIB", 40),
            )?,
        };
        set_direct_buf_allocator(Some(Arc::new(PageHandle(pool.clone()))));
        guard.page = Some(pool);
    }
    if batch_gib > 0.0 {
        let pool = match &shared {
            Some(shared) => create_rmm_pool("batch", 1, false, shared),
            None => create_slot_pool(
                "batch",
                1,
                batch_gib,
                env_usize("LANCE_CUVS_BATCH_SLOT_MIB", 1024),
            )?,
        };
        set_decode_output_allocator(Some(Arc::new(BatchHandle(pool.clone()))));
        guard.batch = Some(pool);
    }
    Ok((guard.page.is_some() || guard.batch.is_some()).then_some(guard))
}
