//! Count only allocations on this test thread; ONNX internals are deliberately excluded.
use beatnet_rs::{BeatNet, BeatNetConfig, Error, Inference, features::FEATURES, transport};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
struct Counting;
thread_local! {static ENABLED:Cell<bool>=const {Cell::new(false)};static COUNT:Cell<usize>=const {Cell::new(0)};}
fn count() {
    if ENABLED.try_with(Cell::get).unwrap_or(false) {
        let _ = COUNT.try_with(|c| c.set(c.get() + 1));
    }
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        count();
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;
struct Model;
impl Inference for Model {
    fn infer(&mut self, _: &[f32; FEATURES]) -> Result<[f32; 3], Error> {
        Ok([0.01, 0.98, 0.01])
    }
    fn reset(&mut self) {}
}
#[test]
fn callback_dsp_resampler_tracker_are_allocation_free() {
    for (rate, channels) in [22050, 22051, 44100, 48000]
        .into_iter()
        .flat_map(|rate| [1, 2, 64].map(|channels| (rate, channels)))
    {
        let mut net = BeatNet::with_model(
            BeatNetConfig {
                sample_rate: rate,
                channels,
                ..Default::default()
            },
            Model,
        )
        .unwrap();
        let (mut producer, mut consumer, stats) = transport::audio_queue(rate, channels).unwrap();
        let (mut outputs, mut output) = transport::output_queue(stats);
        let data = [0.1f32; 1024];
        net.process(&data, |_| {}).unwrap();
        COUNT.set(0);
        ENABLED.set(true);
        for _ in 0..100 {
            producer.push(&data, |x| x);
            while let Some(block) = consumer.pop() {
                net.process(block.samples(), |f| outputs.push(f)).unwrap();
            }
            while output.pop().is_some() {}
        }
        net.discontinuity(100000);
        net.process(&data[..usize::from(channels)], |_| {}).unwrap();
        net.finish(|_| {}).unwrap();
        ENABLED.set(false);
        assert_eq!(COUNT.get(), 0, "sample rate {rate}, channels {channels}");
    }
}
