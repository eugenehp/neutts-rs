# Release notes — neutts-rs 0.0.2

_2026-03-05_

---

## Highlights

### Significantly lower synthesis latency

The two most expensive overheads per synthesis call have been eliminated:

**1. GPU initialisation is now part of model loading, not synthesis.**
In 0.0.1, the Burn wgpu backend was initialised lazily on the first
`decode()` call — meaning the ~1.7 s GPU upload was silently counted inside
synthesis time.  In 0.0.2, `NeuCodecDecoder::from_file()` initialises the
backend immediately, so the cost appears in "loaded in X s" and never
inflates the synthesis RTF.

**2. Per-decode RoPE uploads eliminated.**
The NeuCodec transformer has 12 blocks, each running RoPE on both Q and K
projections — 24 CPU→GPU uploads per decode call in 0.0.1.  In 0.0.2,
sin/cos tables for up to 2048 positions are pre-computed once at load time
and stored as device tensors.  Each decode call now pays only a single O(1)
slice op.

Combined result on M2 MacBook Pro (Metal, neutts-nano-Q4_0, ~125 output tokens):

| | Synth time | RTF |
|---|---|---|
| 0.0.1 | 4.45 s | 1.79× |
| 0.0.2 | ~2.7 s | ~1.1× |

### New: streaming synthesis

The backbone now exposes a token-by-token streaming API:

```rust
backbone.generate_streaming(&prompt, 2048, |piece| {
    // called for every generated text piece — process immediately
    Ok(())
})?;
```

Combined with the codec's non-autoregressive decoder, this lets you start
playing audio before the backbone has finished generating.  The new
`stream_pcm` example demonstrates this end-to-end: models are preloaded once,
tokens are decoded in configurable chunks (default 25 tokens = 500 ms), and
raw PCM is streamed to stdout for piping to any audio player.

### New: `fast` / `precise` feature flags

Two mutually exclusive flags control how sin/cos values are computed for the
Rotary Positional Embedding tables:

- **`fast`** (default) — degree-7/6 Horner polynomial, no transcendental
  calls, ~1 × 10⁻⁴ max absolute error.
- **`precise`** — delegates to `f32::sin_cos()`, correctly rounded (ULP ≤ 1).

Both the GPU precompute path and the CPU runtime path use the same dispatch
function, so the algorithm is consistent across backends.

---

## What's new in detail

### `BackboneModel::generate_streaming` (new method)

```rust
pub fn generate_streaming<F>(
    &self,
    prompt:         &str,
    max_new_tokens: u32,
    mut on_piece:   F,
) -> Result<()>
where
    F: FnMut(&str) -> Result<()>,
```

Mirrors `generate()` but calls `on_piece` for each decoded text fragment as it
is produced.  Returning `Err(...)` from the callback aborts generation and
propagates the error.  The stop token is never forwarded; any text preceding it
within the same piece is correctly emitted first.

### `stream_pcm` example (new)

```sh
# Stream to speakers via aplay (Linux)
cargo run --example stream_pcm --features espeak -- \
  --codes samples/dave.npy --ref-text samples/dave.txt \
  --text  "Hello, streaming audio." | \
  aplay -f S16_LE -r 24000 -c 1

# Stream to speakers via sox (macOS)
... | sox -t raw -r 24000 -e signed -b 16 -c 1 - -d

# Save raw PCM and convert
... > out.pcm && sox -t raw -r 24000 -e signed -b 16 -c 1 out.pcm out.wav
```

PCM format: signed 16-bit little-endian, 24 000 Hz, mono.  
Diagnostics (TTFA, RTF) go to stderr so the byte stream is not corrupted.

| `--chunk` | Audio buffered | Quality |
|-----------|---------------|---------|
| 10 | ~200 ms | mild artefacts at boundaries |
| 25 | ~500 ms | good *(default)* |
| 50 | ~1 s | best |

---

## Upgrade notes

### No breaking changes to the public API

All existing call sites (`load_from_hub_cb`, `NeuTTS::infer`,
`NeuCodecDecoder::decode`, etc.) are unchanged.

### `fast` is now a default feature

If you previously built without `--features wgpu` and explicitly set no RoPE
feature, the new default `fast` is now enabled.  It is backward compatible —
the audio quality difference is imperceptible — but if you need bit-exact
reproducibility with the 0.0.1 output, add `--features precise`.

```sh
# 0.0.1 behaviour (precise sin/cos)
cargo build --features "backbone,espeak,precise"

# 0.0.2 default (fast polynomial — recommended)
cargo build --features "backbone,espeak"
```

### `wgpu` feature now initialises at load time

If you call `NeuCodecDecoder::from_file()` (or `NeuCodecDecoder::new()`) with
the `wgpu` feature, the GPU backend initialises immediately — you will see the
"GPU backend ready in X s" message at load time rather than at the first
`decode()` call.  Adjust any timeout logic if you are wrapping the load step.

### `backend_name()` return values

The string `"burn/wgpu (pending — lazy init)"` is no longer returned; the
backend is always `Ready` after construction.

---

## Full changelog

See [CHANGELOG.md](CHANGELOG.md) for the complete entry-by-entry record.
