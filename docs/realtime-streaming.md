# Realtime streaming wrapper for Apple

`RealtimeStreamingEnhancer` is an optional scheduling wrapper around
`StreamingEnhancer` in the Apple `ReuseStreaming` module. It tracks source sample
positions, bounds pending input, and discards output that can no longer reach its
playout deadline. The underlying incremental inference API is unchanged. Android
continues to use its separate Kotlin/ONNX Runtime streaming implementation.

Use the wrapper on one serial background worker. Submission, processing, reset,
and access to the wrapped enhancer must be serialized. It is not thread-safe and
must never run inside an audio render callback. Model loading and inference can
block while GPU work completes. The playback-clock closure must only read a
snapshot; it must not reenter the wrapper.

## Problem: completed inference can still miss playback

At 16 kHz, a 320-sample hop arrives every 20 ms. In a twelve-layer experiment on an
iPhone 13 Pro Max, sustained completed-frame times of roughly 25–34 ms exceeded
that cadence despite a 120 ms playout buffer. During one five-second interval,
the model returned nonzero output with RMS around −24 dBFS, while no enhanced
samples reached playback. This is stale-output starvation: inference can continue
while playback receives none of its results. It is not evidence of bad weights or
GPU corruption.

A larger buffer allows more latency and absorbs short stalls. It cannot sustain
an indefinitely growing queue when processing takes longer than incoming audio.
A host-side simulation with 34 ms processing still delivered only about 20% wet
samples after dropping old input. An eight-layer, 17 ms simulation restored
continuous enhanced playback. These are scheduling simulations, not GPU speedup
measurements or a guarantee for another device or stream.

This helper makes that scheduling behavior explicit and testable. It does not
make the model faster, guarantee continuous enhanced playback, improve model
quality, or automatically switch twelve layers to eight. Depth selection remains
a host decision.

## Construction and source timeline

```swift
import ReuseStreaming

// Create and warm the model on the same serial worker used below.
let enhancer = try StreamingEnhancer(
    modelURL: checkpointURL, layers: 12, optimization: .specializedMetal
)
try enhancer.warmUp()
let configuration = try RealtimeStreamingConfiguration(
    playoutDelaySamples: 1_920, // 120 ms at 16 kHz
    maximumPendingSamples: 3_200
)
let realtime = RealtimeStreamingEnhancer(
    enhancer: enhancer, configuration: configuration
)
```

The wrapper takes exclusive use of the supplied enhancer and resets its stream
before the first push. Warmed kernels and loaded weights remain available.

`RealtimeAudioBlock` carries finite 16 kHz mono `[Float]` samples, their original
`startSample: UInt64`, and a `generation: UInt64`. A block covers the half-open
source interval `[startSample, startSample + samples.count)`. Returned blocks keep
their actual source positions; dropping input never compacts the timeline.

`RealtimePlaybackPosition(nextSample:generation:)` must come from the host's
actual render clock, expressed in the same sample units and timeline. The helper
uses the configured playout delay when deciding which source samples are still
eligible for playback. For render position `n` and delay `d`, source sample `n - d`
is due when `n >= d`; before that, no source sample is due yet. The earliest
non-expired source position is `max(0, n - d)`. Returned audio can be queued ahead
of its due time; the host must schedule it against that source timeline. Input
arrival time, inference completion time, wall-clock time, and the number of
processed hops are not substitutes for the render clock.

Sample ranges and finite input are validated, including sample-index overflow.
Configuration requires a nonnegative delay and at least 320 pending samples;
the pending limit need not be a multiple of 320. Valid configuration alone does
not guarantee enough processing time to meet a deadline.

Overlapping input within one generation throws. Valid lower-generation input is
ignored; invalid blocks still fail input validation. A lower-generation playback
snapshot at entry prevents inference, and one observed after inference suppresses
its output. A newer input or playback generation invalidates pending and latent
audio, with the model reset before the next push. Use monotonically increasing
generations for a new stream, and never rewind the playback position within a
generation. The wrapper conservatively retains the greatest playback position
observed within that generation. `reset(generation:)` ignores generation rollback.
A same-generation reset clears stream state but preserves accepted input ordering:
it does not permit resubmitting an overlapping range or flushing discarded audio
to playback.

## Integration loop

The following is integration pseudocode. `ingress`, `renderClock`, and `outputQueue`
are host-owned, bounded components. The render callback only transfers samples and
updates its clock; it never calls this loop or waits for inference.

```swift
// Run on one serial worker. Lifecycle changes use this same worker.
while workerIsRunning {
    // Refresh input before EVERY processing attempt, so new arrivals can replace
    // expired work instead of sitting behind repeated processing attempts.
    for block in ingress.takeAvailableBounded() {
        try realtime.submit(block)
    }

    if let restored = try realtime.processNext(playbackPosition: {
        renderClock.snapshot() // Read the live clock each time this is called.
    }) {
        outputQueue.offerBounded(restored)
    }
    workerScheduler.waitOnlyWhenIdle() // No fixed sleep after inference.
}

// On the serial worker, when the host starts a new stream:
realtime.reset(generation: nextGeneration)
```

Each `processNext` processes at most one 320-sample input hop. Refresh ingress and
call it again as appropriate; do not drain a captured backlog with one old clock
snapshot or add a fixed 20 ms sleep after each completed push. The clock closure
is read around inference and again after a deferred model reset, so deadline or
generation changes during a GPU operation can invalidate the result.

The host must check generation and source deadlines again at actual playback.
The clock can advance after `processNext` returns. Retain the source tags in the
bounded output queue, discard expired prefixes, and render only samples belonging
to the current generation and scheduled source interval. The helper does not own
or advance the audio callback's clock.

## Lossy recovery and limits

Pending-input overflow retains only the newest complete 320-sample hop. Outside
the startup latch described below, a completely expired stream is reduced before
processing to its newest complete hop if that hop is still fresh; otherwise no
inference is submitted. Recovery resets model and overlap state before the next
push. Gaps in submitted source ranges also discard the old stream state.

Recovery discards pending and un-emitted audio; it does not call `finish()`.
`finish()` is an end-of-stream operation in the underlying API, not a way to
recover a live deadline miss.

Startup needs two input hops before output is available. A bounded recovery latch
allows the first empty push and the following cold push rather than restarting
before every attempt to produce audio. Completed output is still checked against
a fresh playback position: its expired prefix is trimmed, and stale-generation
results are suppressed. A running inference operation cannot be cancelled by this
wrapper. Inference or output-validation errors propagate after invalidating the
old stream; the next permitted push resets the model. Finite silence alone does
not trigger recovery.

These discontinuities can produce gaps. The host owns silence, concealment, or
dry-audio policy; callback ingress and output queues; routing and sample-rate
conversion; call lifecycle and foreground/background gating; and model-depth
selection. Do not modify the wrapped enhancer independently while it is in use.
The helper supplies neither an audio transport nor an end-to-end latency bound.
A bounded queue prevents unbounded retained work, not sustained throughput loss.

`pendingSampleCount` exposes retained pending input. `statistics` contains
lifetime, saturating counters for discarded input/output, recovery reasons,
processed hops, model resets, and inference failures. A processed hop is not
proof that audio was returned or played; measure actual rendered enhanced samples
in the host as well.
