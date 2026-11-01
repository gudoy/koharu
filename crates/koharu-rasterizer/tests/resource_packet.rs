use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    sync::Arc,
};

use koharu_rasterizer::{PreparedResource, PreparedResourcePacket};

/// Counts the bytes each thread requests, so tests running in parallel do not
/// see each other's allocations.
struct CountingAllocator;

thread_local! {
    static ALLOCATED: Cell<usize> = const { Cell::new(0) };
}

fn count(bytes: usize) {
    let _ = ALLOCATED.try_with(|allocated| allocated.set(allocated.get() + bytes));
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocated_by<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCATED.with(Cell::get);
    let value = operation();
    (value, ALLOCATED.with(Cell::get) - before)
}

#[test]
fn resource_packet_moves_payload_bytes_in_bulk() {
    const PAYLOAD: usize = 16 << 20;
    const OVERHEAD: usize = 4 << 10;
    let bytes: Arc<[u8]> = (0..PAYLOAD).map(|index| index as u8).collect();
    let packet = PreparedResourcePacket::from_resource(
        PreparedResource::encoded_raster(4_000, 6_000, "image/png", bytes).unwrap(),
    )
    .unwrap();

    let (encoded, allocated) = allocated_by(|| packet.encode().unwrap());
    assert!(
        allocated <= PAYLOAD + OVERHEAD,
        "encode allocated {allocated} bytes for a {PAYLOAD}-byte payload"
    );

    let (decoded, allocated) = allocated_by(|| PreparedResourcePacket::decode(&encoded).unwrap());
    assert!(
        allocated <= PAYLOAD + OVERHEAD,
        "decode allocated {allocated} bytes for a {PAYLOAD}-byte payload"
    );
    assert_eq!(decoded, packet);
}
