# beatnet-rs

A synchronous Rust port of [BeatNet](https://github.com/mjhydri/BeatNet) for beat,
downbeat, tempo, and phase tracking. Includes all three pretrained models.
No Python or async runtime is needed to build or run the crate.

```text
PCM → mono → Rubato → Rust features → ONNX CRNN → particle filter → events/state
```

## Usage

```rust
use beatnet_rs::{BeatNet, BeatNetConfig};

fn main() -> Result<(), beatnet_rs::Error> {
    let mut tracker = BeatNet::new(BeatNetConfig {
        sample_rate: 48_000,
        channels: 2,
        ..Default::default()
    })?;

    // Run on a worker thread. Supply interleaved f32 PCM in any chunk size.
    let pcm = [0.0; 960 * 2];
    tracker.process(&pcm, |frame| {
        if let Some(beat) = frame.event {
            println!("{}s: {} BPM, downbeat={}", beat.time, beat.bpm, beat.downbeat);
        }
        // frame.state also exposes phase, meter, confidence, and probabilities.
    })?;
    tracker.finish(|_frame| {})?; // Drain the tail of a finite input.
    Ok(())
}
```

Live capture from the default device (last argument selects model 1, 2, or 3):

```sh
cargo run --release --features live --example live -- 1
```

Linux requires ALSA development headers (`libasound2-dev` on Debian/Ubuntu,
`alsa-lib-devel` on Fedora) and native TLS build dependencies. macOS uses
CoreAudio. Ctrl-C stops the example and joins its worker.

## Realtime contract

- Keep `BeatNet::process` on a synchronous worker. The optional CPAL adapter
  only converts/copies samples into preallocated `rtrb` blocks in its callback.
  Capture is optional; your application can own its device and threads.
- PCM transport holds about 250 ms. Overflow drops incoming blocks; backlog
  above 100 ms causes the consumer to discard stale audio. Compare each block's
  `source_frame()` with the expected position and call `discontinuity(frame)`
  after a gap. The live example does this. Use `samples()` for valid block data.
- The output queue holds 128 frame outputs. Overflow drops new outputs,
  potentially including beat events. Check `Diagnostics::snapshot()` for losses.
- Rust DSP, Rubato, the tracker, and transport have allocation regression tests.
  ONNX uses reusable tensors and one CPU inference thread, but its internals do
  not provide a hard realtime guarantee. Device lifecycle and logging stay
  outside the callback.

## Audio and output

Input is finite, interleaved `f32` PCM, normally in `[-1, 1]`, at 8–192 kHz with
1–64 channels. Chunk boundaries may split channel frames. Channels are averaged;
there is no automatic gain normalization. Rubato resamples to 22,050 Hz, with a
bypass at that rate. Common ratios use FFT resampling; others use synchronous
sinc processing to avoid large FFT blocks.

The model uses 272 features from 1,411-sample centered windows with 441-sample
hops. State updates every 20 ms. Timestamps refer to **source audio time**;
delivery adds roughly 32 ms of window lookahead plus resampler, device, and queue
latency. Resampler startup delay is removed from the audio timeline.

Spectrum magnitudes use `fearless_simd` with runtime CPU selection on x86-64
and NEON on ARM64. CPU detection happens at construction; processing needs no
nightly Rust, async runtime, or additional allocations. FFTs and resampling
retain RustFFT's and Rubato's SIMD implementations.

`BeatEvent` contains time, BPM, downbeat, and confidence. `TrackingState` adds
beat phase `[0, 1)`, meter, and `[beat, downbeat, non-beat]` probabilities. Tempo,
phase, and meter are absent until the first accepted beat. Confidence measures
particle agreement, not calibrated accuracy; phase may jump as estimates change.
Half/double-tempo ambiguity remains possible. Set plausible tempo bounds when known.

`reset()` starts at time zero; `discontinuity(source_frame)` resets all state at
a new source position. `finish()` drains exactly the finite input's tail and is
idempotent. Reset after finishing, a backend error, or a caught callback panic;
otherwise processing returns `Finished` or `NeedsReset`. Invalid PCM is rejected
before consuming that chunk. An incomplete channel frame can be completed before
retrying `finish()`.

## Models and compatibility

Default builds use `ort`'s native CPU runtime, downloaded at build time.
`Model::{One, Two, Three}` selects bundled weights; model 1 is the default.
These respectively exclude GTZAN, Ballroom, and Rock Corpus from training.
Models, frontend coefficients, and reference fixtures are checked in so normal
builds and tests require no Python. Provenance is in [assets/manifest.json](assets/manifest.json).

Use `OnnxModel::from_file` or `from_bytes` with `BeatNet::with_model` for an
external export, or implement the synchronous `Inference` trait. The ONNX
interface requires float32 tensors with these exact names and shapes:

| Inputs | Outputs |
| --- | --- |
| `features [1,1,272]` | `probabilities [3]` |
| `hidden [2,1,150]` | `hidden_out [2,1,150]` |
| `cell [2,1,150]` | `cell_out [2,1,150]` |

For an application-managed ONNX Runtime, disable default features, enable
`dynamic-runtime`, and set `ORT_DYLIB_PATH`. This crate targets API 27; the default
native distribution targets ONNX Runtime 1.28.

The particle filter retains BeatNet's state spaces, transitions, observation
gates, and stratified resampling. Repairs enforce fixed populations, use the
current observation for event decisions, advance meter only on accepted beats,
and handle zero weights/CDF boundaries. It uses a local ChaCha8 seed, not NumPy's
RNG trajectory. Defaults: 1,500 beat particles, 250 downbeat particles, 55–215 BPM,
and meters 2–4. Direct `ParticleFilter` callers must supply 20 ms frames and reset
after gaps. Training, offline DBN decoding, and future-beat scheduling are excluded.

## Development

```sh
cargo test --locked --features live
cargo test --locked --release --features live
cargo clippy --locked --features live --all-targets -- -D warnings
cargo run --locked --release --example benchmark -- 3000
```

Tests cover numerical parity, SIMD backends and numeric extremes, streaming/timing,
failure recovery, tracker behavior, overflow, and allocation guarantees.
Local before/after runs on an AMD Ryzen AI MAX+ 395 measured median processing:

| Input rate | Before | Optimized |
| --- | --- | --- |
| 22,050 Hz | 45 µs | 43 µs |
| 44,100 Hz | 49 µs | 47 µs |
| 48,000 Hz | 49 µs | 45 µs |

Each run processed 10,000 mono hops with model 1 on stable Rust, without
`target-cpu=native`. Optimized pipeline p99 was 54–57 µs per 20 ms hop, with no
drops in the 10-second paced queue test. These are local processing measurements,
not latency bounds; CPU frequency and scheduling affect results. ARM64 was
cross-checked with Clippy; native ARM64 performance, macOS, and physical capture
remain unverified locally.

To regenerate models and fixtures, use Python 3.11 with the CPU PyTorch wheel and
[tools/requirements.txt](tools/requirements.txt). Install NumPy, Cython,
setuptools, and wheel before building madmom with `--no-build-isolation`, then run:

```sh
python tools/export.py /path/to/BeatNet
```

This reads the upstream checkout, exports all three models, checks 2,000 recurrent
frames per model against PyTorch, and regenerates fixtures and provenance.

[LICENSE](LICENSE) contains BeatNet's CC BY 4.0 license, author attribution, and
madmom's BSD source notice. The bundled weights are converted, not retrained.
