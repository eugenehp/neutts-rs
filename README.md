# neutts-rs

Rust port of [NeuTTS](https://github.com/neuphonic/neutts) — on-device voice-cloning TTS
built on a GGUF LLM backbone and the [NeuCodec](https://huggingface.co/neuphonic/neucodec)
neural audio codec.

**Pure Rust — no ONNX Runtime, no native ML dependencies.**  
The codec runs as a self-contained CPU inference engine (`safetensors` + `ndarray` + `rustfft`).

---

## Quick start

### 1. Install system dependencies

```sh
# macOS
brew install espeak-ng

# Ubuntu / Debian
apt install espeak-ng

# Alpine
apk add espeak-ng
```

### 2. Convert codec weights (one-time, ~2 min)

```sh
pip install torch huggingface_hub safetensors
python scripts/convert_weights.py
```

Downloads `neuphonic/neucodec`, extracts the decoder weights, and saves them as
`models/neucodec_decoder.safetensors`.

### 3. Build

```sh
cargo build --features espeak
```

### 4. Clone a voice and synthesise

The simplest path — point at any WAV file and say what you want:

```sh
cargo run --example speak --features espeak -- \
  --wav      my_voice.wav \
  --ref-text "Exactly what I said in the recording." \
  --text     "Hello, this is my cloned voice."
```

On the first run the reference WAV is encoded via the Python `neucodec` package
and cached as `my_voice.npy` beside the WAV.  Every subsequent run loads the
cache and skips encoding entirely.

**One-time Python install for encoding:**

```sh
pip install neucodec huggingface_hub torchaudio
```

---

## Examples

| Example | What it does |
|---------|-------------|
| `speak` | **Recommended.** WAV in → synthesised audio out. Encodes on first run, caches `.npy` beside the WAV. Supports `--list-models`, `--list-files`, `--gguf-file`. |
| `basic` | Synthesise from a pre-encoded `.npy` reference |
| `clone_voice` | Full voice cloning — `.npy` or raw WAV + SHA-256 cache |
| `encode_reference` | Stub — returns a helpful error; use Python for now |
| `download_models` | Download / stage weights |
| `test_pipeline` | Smoke-test every component without model files |

### speak

```sh
# Minimal — encodes reference on first run, cached after that
cargo run --example speak --features espeak -- \
  --wav      my_voice.wav \
  --ref-text "What I said in the recording." \
  --text     "Hello, this is my cloned voice."

# Use a bundled sample voice (pre-encoded, no Python needed)
cargo run --example speak --features espeak -- \
  --wav      samples/jo.wav \
  --ref-text samples/jo.txt \
  --text     "Hello from Jo."

# Skip directly to a pre-encoded .npy
cargo run --example speak --features espeak -- \
  --codes    samples/dave.npy \
  --ref-text samples/dave.txt \
  --text     "Hello from Dave."

# List all known backbone models
cargo run --example speak --features espeak -- --list-models

# List GGUF files available in a specific repo
cargo run --example speak --features espeak -- \
  --backbone neuphonic/neutts-nano-q4-gguf --list-files

# Pick a specific GGUF quantisation
cargo run --example speak --features espeak -- \
  --wav       my_voice.wav \
  --ref-text  "What I said." \
  --text      "Hello." \
  --backbone  neuphonic/neutts-nano-q4-gguf \
  --gguf-file neutts-nano-Q4_K_M.gguf

# Different language backbone
cargo run --example speak --features espeak -- \
  --backbone neuphonic/neutts-nano-german-q4-gguf \
  --wav      samples/greta.wav \
  --ref-text samples/greta.txt \
  --text     "Hallo aus Rust."

# CPU-only (no wgpu)
cargo run --example speak --no-default-features --features espeak -- \
  --wav my_voice.wav --ref-text "..." --text "Hello."
```

**speak flags:**

| Flag | Short | Purpose |
|------|-------|---------|
| `--wav PATH` | `-w` | WAV file of the voice to clone |
| `--codes PATH` | `-c` | Pre-encoded `.npy` (skips encoding) |
| `--ref-text TEXT\|PATH` | `-r` | Transcript of the reference WAV (file or literal string). Auto-detected from `<stem>.txt` if omitted. |
| `--text TEXT` | `-t` | Text to synthesise |
| `--out PATH` | `-o` | Output WAV (default: `output.wav`) |
| `--backbone REPO` | `-b` | HuggingFace backbone repo (see `--list-models`) |
| `--gguf-file FILE` | `-g` | Specific `.gguf` filename within the repo |
| `--list-files` | | Print all `.gguf` files in `--backbone` and exit |
| `--list-models` | | Print table of all known backbone repos and exit |

### basic

```sh
# Default: NeuTTS-Nano Q4, bundled Jo voice
cargo run --example basic --features espeak

# Custom text and reference
cargo run --example basic --features espeak -- \
  --text      "The quick brown fox." \
  --ref-codes samples/jo.npy \
  --ref-text  samples/jo.txt

# Different backbone
cargo run --example basic --features espeak -- \
  --backbone neuphonic/neutts-air-q4-gguf
```

### clone_voice

```sh
# First run: encodes + SHA-256 caches
cargo run --example clone_voice --features espeak -- \
  --ref-audio samples/jo.wav \
  --text      "Hello."

# Second run: cache hit, encoder skipped
cargo run --example clone_voice --features espeak -- \
  --ref-audio samples/jo.wav \
  --text      "Different text."

# Pre-encoded .npy
cargo run --example clone_voice --features espeak -- \
  --ref-codes samples/jo.npy \
  --ref-text  samples/jo.txt \
  --text      "Hello."
```

---

## Available models

Run `--list-models` to see the full table at any time:

```sh
cargo run --example speak -- --list-models
```

| Repo | Name | Language | Params | GGUF |
|------|------|----------|--------|------|
| `neuphonic/neutts-nano-q4-gguf` | NeuTTS Nano Q4 | en-us | 0.2B | ✅ |
| `neuphonic/neutts-nano-q8-gguf` | NeuTTS Nano Q8 | en-us | 0.2B | ✅ |
| `neuphonic/neutts-nano` | NeuTTS Nano (full) | en-us | 0.2B | |
| `neuphonic/neutts-air-q4-gguf` | NeuTTS Air Q4 | en-us | 0.7B | ✅ |
| `neuphonic/neutts-air-q8-gguf` | NeuTTS Air Q8 | en-us | 0.7B | ✅ |
| `neuphonic/neutts-air` | NeuTTS Air (full) | en-us | 0.7B | |
| `neuphonic/neutts-nano-german-q4-gguf` | NeuTTS Nano German Q4 | de | 0.2B | ✅ |
| `neuphonic/neutts-nano-german-q8-gguf` | NeuTTS Nano German Q8 | de | 0.2B | ✅ |
| `neuphonic/neutts-nano-german` | NeuTTS Nano German (full) | de | 0.2B | |
| `neuphonic/neutts-nano-french-q4-gguf` | NeuTTS Nano French Q4 | fr-fr | 0.2B | ✅ |
| `neuphonic/neutts-nano-french-q8-gguf` | NeuTTS Nano French Q8 | fr-fr | 0.2B | ✅ |
| `neuphonic/neutts-nano-french` | NeuTTS Nano French (full) | fr-fr | 0.2B | |
| `neuphonic/neutts-nano-spanish-q4-gguf` | NeuTTS Nano Spanish Q4 | es | 0.2B | ✅ |
| `neuphonic/neutts-nano-spanish-q8-gguf` | NeuTTS Nano Spanish Q8 | es | 0.2B | ✅ |
| `neuphonic/neutts-nano-spanish` | NeuTTS Nano Spanish (full) | es | 0.2B | ✅ |

To discover which specific GGUF quantisation variants are in a repo:

```sh
cargo run --example speak -- \
  --backbone neuphonic/neutts-nano-q4-gguf --list-files
```

Then pick one with `--gguf-file`:

```sh
cargo run --example speak --features espeak -- \
  --backbone  neuphonic/neutts-nano-q4-gguf \
  --gguf-file neutts-nano-Q4_K_M.gguf \
  --wav my_voice.wav --ref-text "..." --text "Hello."
```

---

## Architecture

```
text ──► espeak-ng ──► IPA ──┐
                              ├──► prompt builder ──► GGUF backbone ──► speech tokens
ref_codes (.npy) ─────────────┘                                               │
                                                                               ▼
                                                                   NeuCodec decoder
                                                                               │
                                                                               ▼
                                                                   audio (Vec<f32>, 24 kHz)
```

### GGUF backbone

Small causal LM in GGUF format, run via `llama-cpp-2`.  Takes a phonemized text
prompt and pre-encoded reference speaker codes, generates `<|speech_N|>` tokens.

### NeuCodec decoder (pure Rust)

XCodec2-based architecture loaded at runtime from `models/neucodec_decoder.safetensors`:

```
codes [T]
   └─► FSQ decode  (integer → 8 scaled digits → project_out Linear 8→2048)
         │
    fc_post_a  (Linear 2048→1024)
         │
   VocosBackbone
    ├─ Conv1d(k=7)
    ├─ 2 × ResnetBlock  (GroupNorm → SiLU → Conv1d)
    ├─ 12 × TransformerBlock  (RMSNorm → MHA + RoPE → SiLU MLP)
    └─ 2 × ResnetBlock + LayerNorm
         │
   ISTFTHead
    ├─ Linear(1024 → n_fft+2)
    └─ ISTFT (same padding, Hann window)
         │
   audio [T × hop_length]  (24 kHz)
```

| Property | Value |
|----------|-------|
| Output sample rate | 24 000 Hz |
| Tokens / second | 50 |
| Samples / token | 480 (hop_length) |
| FSQ codebook size | 4⁸ = 65 536 codes |
| Encoder input | 16 000 Hz mono WAV |

---

## Bundled reference voices

Five pre-encoded voices are included and work without any Python encoding step:

| Files | Voice | Language |
|-------|-------|----------|
| `samples/jo.*` | Jo | English |
| `samples/dave.*` | Dave | English |
| `samples/juliette.*` | Juliette | French |
| `samples/greta.*` | Greta | German |
| `samples/mateo.*` | Mateo | Spanish |

Each has a `.wav` (original audio), `.npy` (pre-encoded tokens), and `.txt` (transcript).

---

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `backbone` | ✓ | GGUF backbone via `llama-cpp-2` (requires cmake + C++) |
| `espeak` | | Raw-text input via `libespeak-ng` |
| `wgpu` | | Reserved for future GPU codec acceleration (currently no-op) |
| `metal` | | macOS Metal GPU for the backbone |
| `cuda` | | NVIDIA CUDA for the backbone |

**Without `backbone`** — codec-only mode; use `NeuCodecDecoder::decode()` directly.

**Without `espeak`** — pass pre-phonemized IPA via `tts.infer_from_ipa()`.

---

## Build requirements

| Platform | Backbone | Codec | Phonemizer |
|----------|----------|-------|------------|
| Linux / macOS | cmake + C++ (auto) | pure Rust | `libespeak-ng-dev` / `brew install espeak-ng` |
| iOS / Android | cross-compile llama.cpp | pure Rust | cross-compile espeak-ng; set `ESPEAK_LIB_DIR` |

---

## Using the library

### Full pipeline

```rust
use neutts::{NeuTTS, download};
use std::path::Path;

// Download backbone from HuggingFace (cached after first run).
// Pass None to auto-select the first GGUF in the repo,
// or Some("filename.gguf") to pick a specific quantisation.
let tts = download::load_from_hub_cb(
    "neuphonic/neutts-nano-q4-gguf",
    None,           // or Some("neutts-nano-Q4_K_M.gguf")
    |_| {},
).unwrap();

// Load pre-encoded reference codes
let ref_codes = tts.load_ref_codes(Path::new("samples/jo.npy")).unwrap();

// Synthesise — returns Vec<f32> at 24 kHz mono
let audio = tts.infer(
    "Hello from Rust!",
    &ref_codes,
    "Transcript of the reference recording.",
).unwrap();

// Save to WAV
tts.write_wav(&audio, Path::new("output.wav")).unwrap();
```

### Discover models programmatically

```rust
use neutts::download::{BACKBONE_MODELS, list_gguf_files, find_model};

// Iterate the registry
for m in BACKBONE_MODELS {
    println!("{} ({}) — GGUF: {}", m.repo, m.language, m.is_gguf);
}

// Find a specific repo
if let Some(info) = find_model("neuphonic/neutts-nano-q4-gguf") {
    println!("language: {}", info.language); // "en-us"
}

// List GGUF files available in a repo (network call)
let files = list_gguf_files("neuphonic/neutts-nano-q4-gguf").unwrap();
for f in &files { println!("{f}"); }
```

### IPA passthrough (without espeak)

```rust
let audio = tts.infer_from_ipa(
    "hɛloʊ fɹʌm ɹʌst",    // input IPA
    &ref_codes,
    "wɪ ɑːɹ tɛstɪŋ ðɪs",  // reference IPA
).unwrap();
```

### Decoder only

```rust
use neutts::NeuCodecDecoder;

// Loads models/neucodec_decoder.safetensors at runtime
let dec = NeuCodecDecoder::new().unwrap();
println!("backend: {}", dec.backend_name()); // "cpu (ndarray)"
println!("{} samples/token", dec.hop_length());

let codes: Vec<i32> = vec![/* speech token IDs */];
let audio: Vec<f32> = dec.decode(&codes).unwrap();
```

### Reference-code cache

```rust
use neutts::RefCodeCache;
use std::path::Path;

let cache = RefCodeCache::new()?;
if let Some((codes, outcome)) = cache.try_load(Path::new("reference.wav"))? {
    println!("{outcome}"); // "Cache hit (SHA-256: …)"
}
```

---

## Mobile / C FFI

A practical mobile architecture runs the backbone server-side and only the
NeuCodec decoder on-device:

```c
NeuTtsHandle *codec = neutts_model_load("/path/to/neucodec_decoder.safetensors");

float *audio = neutts_decode_tokens(codec, codes, num_codes, &n_samples);
neutts_write_wav(audio, n_samples, "/path/to/output.wav");
neutts_free_audio(audio, n_samples);
neutts_model_free(codec);
```

See [`include/neutts.h`](include/neutts.h) for the full C header.

---

## Pipeline stages

1. **Text preprocessing** — numbers, currencies, abbreviations → spoken words
2. **Phonemisation** — espeak-ng converts text to IPA phonemes
3. **Prompt construction** — reference codes + IPA → GGUF prompt
4. **Backbone inference** — GGUF LLM generates `<|speech_N|>` tokens
5. **Token extraction** — regex extracts integer IDs from generated text
6. **Codec decode** — NeuCodec decoder converts IDs to 24 kHz audio

---

## Status

| Component | Status |
|-----------|--------|
| GGUF backbone inference | ✅ |
| NeuCodec decoder (pure Rust, safetensors) | ✅ |
| NeuCodec encoder (pure Rust) | ⏳ not yet — `speak` example falls back to Python `neucodec` |
| English backbones (Nano / Air, Q4 / Q8) | ✅ |
| German / French / Spanish backbones | ✅ |
| Full (non-GGUF) model repos | ✅ in registry; GGUF files detected automatically |
| GPU acceleration (codec) | ⏳ planned via `wgpu` feature |
| iOS / Android build | ✅ codec is pure Rust; backbone needs cross-compile |

---

## Citation

If you use this software in your research or project, please cite it as:

```bibtex
@software{hauptmann2026neuttsrs,
  author       = {Hauptmann, Eugene},
  title        = {{neutts}: Rust port of {NeuTTS} — on-device voice-cloning {TTS}
                  with {GGUF} backbone and {NeuCodec} decoder},
  year         = {2026},
  version      = {0.0.1},
  license      = {MIT},
  url          = {https://github.com/eugenehp/neutts-rs}
}
```

If you also use the underlying NeuTTS model or NeuCodec, please cite those works directly via their respective HuggingFace repositories at [huggingface.co/neuphonic](https://huggingface.co/neuphonic).

---

## License

MIT
