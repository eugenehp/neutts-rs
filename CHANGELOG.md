# Changelog

All notable changes to **neutts-rs** are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.0.1] — 2026-03-04

### Added

#### NeuCodec decoder — pure-Rust CPU inference (`src/codec.rs`)
- Full XCodec2-based decoder implemented from scratch in safe Rust:
  - **FSQ decode** — integer codes (0–65 535) decomposed into 8 base-4 digits,
    scaled to {−1, −⅓, ⅓, 1}, projected via a learned linear layer.
    Supports both `generator.quantizer.fsqs.0.project_out.*` (older exports) and
    the flat `generator.quantizer.project_out.*` key layout automatically.
  - **VocosBackbone** — Conv1d(k=7) → 2 × ResnetBlock → 12 × TransformerBlock
    (RMSNorm + multi-head attention + RoPE + SiLU MLP) → 2 × ResnetBlock →
    LayerNorm.
  - **ISTFTHead** — Linear(hidden → n_fft+2) → split magnitude/phase → ISTFT
    with same-padding and Hann window.
  - **ISTFT** — overlap-add synthesis with window-envelope normalisation;
    produces 24 kHz mono `Vec<f32>`.
- Runtime weight loading from `models/neucodec_decoder.safetensors`; no
  recompilation needed when weights change.
- Supports both `F32` and `BF16` tensor dtypes in the safetensors file.
- Architecture hyper-parameters (`hidden_dim`, `depth`, `n_heads`,
  `hop_length`) auto-detected from weight shapes at load time.
- `n_heads` optionally overridden via `__metadata__` key in the safetensors
  header.
- Decoder output constants exported: `SAMPLE_RATE` (24 000 Hz),
  `ENCODER_SAMPLE_RATE` (16 000 Hz), `SAMPLES_PER_TOKEN` (480),
  `ENCODER_SAMPLES_PER_TOKEN` (320).
- Naive linear resampler (`resample()`) for 16 kHz → 24 kHz (or any ratio).
- `NeuCodecEncoder` stub with clear error message directing users to the Python
  `neucodec` package until a pure-Rust encoder is available.
- Full unit-test suite covering FSQ decode, `linear`, `conv1d`, `group_norm`,
  `layer_norm`, `rms_norm`, RoPE, Hann window, ISTFT length, and resampler.

#### Model registry (`src/download.rs`)
- `ModelInfo` struct capturing `repo`, `name`, `language`, `params`, and
  `is_gguf` for every known backbone.
- `BACKBONE_MODELS` static registry covering all 15 official repos:
  - English: Nano Q4/Q8/full, Air Q4/Q8/full
  - German: Nano Q4/Q8/full
  - French: Nano Q4/Q8/full
  - Spanish: Nano Q4/Q8/full
- `find_model(repo)` — O(n) registry lookup by repo ID.
- `backbone_language(repo)` — derives espeak-ng language code from the registry
  (falls back to `"en-us"` for unknown repos).
- `load_from_hub_cb` gains a `gguf_file: Option<&str>` parameter — pass `None`
  to auto-select the first `.gguf` found (previous behaviour) or
  `Some("filename.gguf")` to pin a specific quantisation variant.
- `load_from_hub` — convenience wrapper; unchanged call site, passes `None`
  internally.
- `list_gguf_files(repo)` — queries HuggingFace Hub and returns the names of
  all `.gguf` files in a repo; used by `--list-files`.
- `supported_gguf_repos()` — derived from `BACKBONE_MODELS`; replaces the
  hard-coded list.
- `print_model_table()` — prints a formatted ASCII table of all known models
  to stdout.

#### `speak` example (`examples/speak.rs`)
- New end-to-end example: WAV file in, synthesised WAV out, one command.
- **`--wav PATH`** — reference voice WAV file.  On first run the audio is
  encoded via the Python `neucodec` package (inline subprocess, no script
  file needed) and cached as `<stem>.npy` beside the WAV.  Subsequent runs
  load the cache and skip encoding entirely.
- **`--codes PATH`** — load a pre-encoded `.npy` directly, bypassing the
  encoder entirely.
- **`--ref-text TEXT|PATH`** — transcript of the reference recording; accepts
  a literal string or a file path.  Auto-detected from `<wav_stem>.txt` if
  the flag is omitted.
- **`--text TEXT`** — text to synthesise.
- **`--out PATH`** — output WAV path (default: `output.wav`).
- **`--backbone REPO`** — HuggingFace backbone repo ID.
- **`--gguf-file FILE`** — specific `.gguf` filename within the repo.
- **`--list-files`** — prints all `.gguf` files in `--backbone` (with registry
  metadata if the repo is known) and exits.
- **`--list-models`** — prints the full `print_model_table()` and exits.
- Python encoder subprocess: auto-detects `python3` / `python`; forwards
  stdout and stderr so HuggingFace download progress is visible; handles
  stereo-to-mono conversion and resampling to 16 kHz inside the inline script.
- Warning printed when reference audio is shorter than 3 seconds.

#### Weight conversion script (`scripts/convert_weights.py`)
- Downloads `neuphonic/neucodec` from HuggingFace Hub.
- Extracts only the decoder weights (generator, fc_post_a) from
  `pytorch_model.bin`.
- Saves `models/neucodec_decoder.safetensors` with `__metadata__` containing
  `n_heads` so the Rust loader can read it without probing.

#### Reference-code cache (`src/cache.rs`)
- `RefCodeCache` — SHA-256 keyed disk cache for pre-encoded reference codes.
  `try_load(wav)` returns codes + a `CacheOutcome` message on hit; `store()`
  writes codes on miss.

#### NPY I/O (`src/npy.rs`)
- `load_npy` / `load_npy_i32` — reads NPY v1.0 and v2.0 files; supports
  `float32` and `int32` dtypes (little-endian and big-endian); tolerates
  float arrays that are integer-valued (cast on load).
- `write_npy_i32` — writes a 1-D `int32` array as a valid NPY v1.0 file,
  padding the header to a multiple of 64 bytes per spec.
- `load_npz` — reads NPZ (ZIP of NPY) archives.

#### Bundled sample voices
- Five pre-encoded reference voices: Jo, Dave, Juliette, Greta, Mateo.
  Each has a `.wav`, `.npy`, and `.txt` transcript ready for use without
  any Python encoding step.

#### C FFI (`src/ffi.rs`, `include/neutts.h`)
- `neutts_model_load` / `neutts_model_free`
- `neutts_decode_tokens`
- `neutts_write_wav` / `neutts_free_audio`

#### Other
- `clone_voice` example — full voice cloning with SHA-256 reference-code cache,
  `--ref-audio` (WAV) or `--ref-codes` (NPY), `--no-cache`, `--cache-dir`.
- `encode_reference` example — helpful stub with Python fallback instructions.
- `download_models` example — stages safetensors weights.
- `test_pipeline` example — smoke-tests all components without model files.
- `src/preprocess.rs` — number, currency, and abbreviation normalisation.
- `src/phonemize.rs` — espeak-ng bindings; `is_espeak_available()` probe.
- `src/tokens.rs` — prompt builder and `<|speech_N|>` token extractor.
- `src/backbone.rs` — `BackboneModel` wrapping `llama-cpp-2`.

### Fixed

- **FSQ weight key** — `load_decoder_weights` now tries
  `generator.quantizer.fsqs.0.project_out.*` first and falls back to the flat
  `generator.quantizer.project_out.*` layout used by current NeuCodec exports,
  resolving a `TensorNotFound` error on all recent model checkpoints.

### Changed

- `load_from_hub_cb` signature gains a second parameter
  `gguf_file: Option<&str>` between `backbone_repo` and `on_progress`.
  Pass `None` to preserve the previous auto-select behaviour.
- `supported_backbone_repos()` is now derived from `BACKBONE_MODELS` rather
  than a hard-coded list.
- `backbone_language()` is now driven by the `BACKBONE_MODELS` registry
  instead of substring matching.
- README fully rewritten: quick-start uses `speak`, full model table, API
  examples updated for new `load_from_hub_cb` signature, citation block added.

---

## [Unreleased]

### Planned
- Pure-Rust NeuCodec encoder (removes Python dependency for reference encoding).
- `wgpu` feature for GPU-accelerated codec inference.
- Streaming / chunked synthesis API.
- iOS / Android build guides and pre-built XCFramework.
