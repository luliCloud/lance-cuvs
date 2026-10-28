//! EXPERIMENT (H2 prototype): a fixed pool of pinned host memory handed to
//! lance-io for its large `O_DIRECT` page-read buffers, so these ~32 MiB
//! allocations do not go through the global allocator. Falls back to the heap
//! when the pool is empty or a request does not fit a slot.
//!
//! `LANCE_CUVS_PAGE_POOL_GIB` (unset/0 = off), `LANCE_CUVS_PAGE_SLOT_MIB` (default 40).
//!
//! `LANCE_CUVS_PAGE_POOL_RMM=1` backs the pool with cuVS's RMM pinned host pool instead of the
//! fixed slots above: one `cuvsRMMPinnedHostPoolEnable(GIB)` (initial size == maximum size, so no
//! cudaHostAlloc during the run), buffers come from `cuvsRMMHostAlloc` / `cuvsRMMHostFree`, and
//! when the pool cannot serve a request the allocator returns `None` and lance-io falls back to
//! the heap, exactly as with the slots.

use crate::cuda::PinnedHostBuffer;
use lance_core::Result;
use lance_io::buffer_pool::{DirectBufAllocator, PoolBuf, set_direct_buf_allocator};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const ALIGN: usize = 4096;

pub(crate) struct PagePool {
    // Owns the pinned region; only touched on drop.
    _region: Mutex<PinnedHostBuffer<u8>>,
    base: usize,
    slot_bytes: usize,
    free: Mutex<Vec<usize>>,
    slots: usize,
    hits: AtomicU64,
    misses_empty: AtomicU64,
    misses_too_big: AtomicU64,
    max_len: AtomicUsize,
    in_use: AtomicUsize,
    max_in_use: AtomicUsize,
}

struct PoolHandle(Arc<PagePool>);

impl DirectBufAllocator for PoolHandle {
    fn alloc(&self, len: usize) -> Option<PoolBuf> {
        let pool = &self.0;
        pool.max_len.fetch_max(len, Ordering::Relaxed);
        if len > pool.slot_bytes {
            pool.misses_too_big.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let idx = pool.free.lock().expect("page pool poisoned").pop();
        let Some(idx) = idx else {
            pool.misses_empty.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        pool.hits.fetch_add(1, Ordering::Relaxed);
        let now = pool.in_use.fetch_add(1, Ordering::Relaxed) + 1;
        pool.max_in_use.fetch_max(now, Ordering::Relaxed);
        let ptr = NonNull::new((pool.base + idx * pool.slot_bytes) as *mut u8)?;
        let pool = pool.clone();
        Some(PoolBuf {
            ptr,
            release: Box::new(move || {
                pool.in_use.fetch_sub(1, Ordering::Relaxed);
                pool.free.lock().expect("page pool poisoned").push(idx);
            }),
        })
    }
}

/// Registers the pool with lance-io for its lifetime; unregisters on drop.
pub(crate) struct SlotsGuard {
    pool: Arc<PagePool>,
}

impl Drop for SlotsGuard {
    fn drop(&mut self) {
        set_direct_buf_allocator(None);
        let p = &self.pool;
        eprintln!(
            "cuVS page pool: slots={} slot_mib={} hits={} misses_empty={} misses_too_big={} max_in_use={} max_request_mib={:.2}",
            p.slots,
            p.slot_bytes >> 20,
            p.hits.load(Ordering::Relaxed),
            p.misses_empty.load(Ordering::Relaxed),
            p.misses_too_big.load(Ordering::Relaxed),
            p.max_in_use.load(Ordering::Relaxed),
            p.max_len.load(Ordering::Relaxed) as f64 / (1u64 << 20) as f64,
        );
    }
}

/// Keeps the pool registered with lance-io for as long as it lives.
pub(crate) enum PagePoolGuard {
    Slots(SlotsGuard),
    Rmm(RmmGuard),
}

// --- RMM-backed pool -------------------------------------------------------------------------

struct RmmPool {
    hits: AtomicU64,
    misses: AtomicU64,
    realigned: AtomicU64,
    max_len: AtomicUsize,
    in_use_bytes: AtomicUsize,
    max_in_use_bytes: AtomicUsize,
    in_use: AtomicUsize,
    max_in_use: AtomicUsize,
}

impl Drop for RmmPool {
    // Runs when the guard, the registered allocator and every outstanding buffer are gone, so the
    // pool is only released after the last cuvsRMMHostFree.
    fn drop(&mut self) {
        let _ = unsafe { cuvs_sys::cuvsRMMPinnedHostPoolDisable() };
        eprintln!(
            "cuVS page pool (RMM): hits={} misses={} realigned={} max_in_use={} max_in_use_gib={:.2} max_request_mib={:.2}",
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.realigned.load(Ordering::Relaxed),
            self.max_in_use.load(Ordering::Relaxed),
            self.max_in_use_bytes.load(Ordering::Relaxed) as f64 / (1u64 << 30) as f64,
            self.max_len.load(Ordering::Relaxed) as f64 / (1u64 << 20) as f64,
        );
    }
}

struct RmmHandle(Arc<RmmPool>);

impl DirectBufAllocator for RmmHandle {
    fn alloc(&self, len: usize) -> Option<PoolBuf> {
        let pool = &self.0;
        pool.max_len.fetch_max(len, Ordering::Relaxed);
        // O_DIRECT needs 4096-aligned buffers; the pool only guarantees 256 bytes. Ask for one
        // extra page, hand out the first page-aligned address inside the block, and free the
        // original pointer.
        let bytes = len.div_ceil(ALIGN) * ALIGN + ALIGN;
        let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
        let status = unsafe { cuvs_sys::cuvsRMMHostAlloc(&mut raw, bytes) };
        if status != cuvs_sys::cuvsError_t::CUVS_SUCCESS {
            pool.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let aligned = (raw as usize).div_ceil(ALIGN) * ALIGN;
        if aligned != raw as usize {
            pool.realigned.fetch_add(1, Ordering::Relaxed);
        }
        let ptr = NonNull::new(aligned as *mut u8)?;
        pool.hits.fetch_add(1, Ordering::Relaxed);
        let now = pool.in_use.fetch_add(1, Ordering::Relaxed) + 1;
        pool.max_in_use.fetch_max(now, Ordering::Relaxed);
        let now_bytes = pool.in_use_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        pool.max_in_use_bytes.fetch_max(now_bytes, Ordering::Relaxed);
        let addr = raw as usize;
        let pool = pool.clone();
        Some(PoolBuf {
            ptr,
            release: Box::new(move || {
                pool.in_use.fetch_sub(1, Ordering::Relaxed);
                pool.in_use_bytes.fetch_sub(bytes, Ordering::Relaxed);
                let _ = unsafe { cuvs_sys::cuvsRMMHostFree(addr as *mut std::ffi::c_void, bytes) };
            }),
        })
    }
}

pub(crate) struct RmmGuard {
    _pool: Arc<RmmPool>,
}

impl Drop for RmmGuard {
    fn drop(&mut self) {
        set_direct_buf_allocator(None);
    }
}

fn rmm_page_pool(gib: f64) -> Result<Option<PagePoolGuard>> {
    let bytes = (gib * (1u64 << 30) as f64) as usize;
    let start = std::time::Instant::now();
    crate::cuda::check_cuvs(
        unsafe { cuvs_sys::cuvsRMMPinnedHostPoolEnable(bytes) },
        "enable the cuVS RMM pinned host pool",
    )?;
    let pool = Arc::new(RmmPool {
        hits: AtomicU64::new(0),
        misses: AtomicU64::new(0),
        realigned: AtomicU64::new(0),
        max_len: AtomicUsize::new(0),
        in_use_bytes: AtomicUsize::new(0),
        max_in_use_bytes: AtomicUsize::new(0),
        in_use: AtomicUsize::new(0),
        max_in_use: AtomicUsize::new(0),
    });
    set_direct_buf_allocator(Some(Arc::new(RmmHandle(pool.clone()))));
    eprintln!(
        "cuVS page pool (RMM): enabled, {:.1} GiB pinned, initial == maximum (alloc {:.1}s)",
        bytes as f64 / (1u64 << 30) as f64,
        start.elapsed().as_secs_f64()
    );
    Ok(Some(PagePoolGuard::Rmm(RmmGuard { _pool: pool })))
}

pub(crate) fn page_pool_from_env() -> Result<Option<PagePoolGuard>> {
    let gib: f64 = std::env::var("LANCE_CUVS_PAGE_POOL_GIB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    if !(gib > 0.0) {
        return Ok(None);
    }
    if std::env::var("LANCE_CUVS_PAGE_POOL_RMM").map(|v| v == "1").unwrap_or(false) {
        return rmm_page_pool(gib);
    }
    let slot_mib: usize = std::env::var("LANCE_CUVS_PAGE_SLOT_MIB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    let slot_bytes = (slot_mib << 20).div_ceil(ALIGN) * ALIGN;
    let slots = ((gib * (1u64 << 30) as f64) as usize / slot_bytes).max(1);
    let start = std::time::Instant::now();
    // One extra page of slack so the first slot can be aligned to 4096.
    let mut region = PinnedHostBuffer::<u8>::try_new(slots * slot_bytes + ALIGN)?;
    let raw = region.as_mut_slice().as_mut_ptr() as usize;
    let base = raw.div_ceil(ALIGN) * ALIGN;
    let pool = Arc::new(PagePool {
        _region: Mutex::new(region),
        base,
        slot_bytes,
        free: Mutex::new((0..slots).rev().collect()),
        slots,
        hits: AtomicU64::new(0),
        misses_empty: AtomicU64::new(0),
        misses_too_big: AtomicU64::new(0),
        max_len: AtomicUsize::new(0),
        in_use: AtomicUsize::new(0),
        max_in_use: AtomicUsize::new(0),
    });
    set_direct_buf_allocator(Some(Arc::new(PoolHandle(pool.clone()))));
    eprintln!(
        "cuVS page pool: enabled, {} x {} MiB pinned = {:.1} GiB (alloc {:.1}s)",
        slots,
        slot_bytes >> 20,
        (slots * slot_bytes) as f64 / (1u64 << 30) as f64,
        start.elapsed().as_secs_f64()
    );
    Ok(Some(PagePoolGuard::Slots(SlotsGuard { pool })))
}
