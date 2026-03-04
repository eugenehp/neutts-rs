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

Downloads `neuphonic/neucodec/pytorch_model.bin`, extracts the decoder weights,
and saves them as `models/neucodec_decoder.safetensors`.

### 3. Build

```sh
cargo build --features espeak
```

### 4. Encode a reference voice

Encoding is not yet implemented in pure Rust (it requires the heavy Wav2Vec2Bert
semantic model).  Use Python once per reference speaker:

```sh
pip install neucodec torchaudio numpy
python -c "
from neucodec import NeuCodec
import numpy as np, torchaudio
model = NeuCodec.from_pretrained('neuphonic/neucodec')
y, sr = torchaudio.load('reference.wav')
codes = model.encode_code(y)
np.save('ref.npy', codes.numpy().astype('int32'))
print(f'{len(codes[0])} tokens saved')
"
```

### 5. Synthesise

```sh
cargo run --example basic --features espeak -- \
  --ref-codes ref.npy \
  --ref-text  "Transcript of your reference recording." \
  --text      "Hello, this is your cloned voice speaking."
```

---

## Examples

| Example | What it does |
|---------|-------------|
| `test_pipeline` | Smoke-test every pipeline component that works without model files |
| `basic` | Synthesise from a pre-encoded `.npy` reference |
| `clone_voice` | Full voice cloning — pre-encoded `.npy` or raw WAV + auto SHA-256 cache |
| `encode_reference` | Stub — returns a helpful error; use Python for now |
| `download_models` | Download / stage weights (updated for safetensors workflow) |

```sh
# No models needed
cargo run --example test_pipeline --no-default-features

# Synthesis
cargo run --example basic --features espeak

# Voice cloning with cache
cargo run --example clone_voice --features espeak -- \
  --ref-codes samples/jo.npy \
  --ref-text  samples/jo.txt \
  --text      "Hello from Rust."
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

XCodec2-based architecture loaded from `models/neucodec_decoder.safetensors`:

```
codes [T]                      FSQ lookup                [T, 2048]
   └─► decode integer indices ─────────────────────────► project_out (Linear 8→2048)
                                                               │
                                                          fc_post_a (Linear 2048→1024)
                                                               │
                                                         VocosBackbone
                                                          ├─ Conv1d(k=7)
                                                          ├─ 2 × ResnetBlock
                                                          ├─ 12 × TransformerBlock (RoPE)
                                                          └─ 2 × ResnetBlock + LayerNorm
                                                               │
                                                         ISTFTHead
                                                          ├─ Linear(1024 → n_fft+2)
                                                          └─ ISTFT (same padding)
                                                               │
                                                        audio [T × hop_length]
```

| Property | Value |
|----------|-------|
| Output sample rate | 24 000 Hz |
| Tokens / second | 50 (hop_length = 480) |
| Samples / token | 480 |
| FSQ codebook | 4⁸ = 65 536 codes |
| Input to encoder | 16 000 Hz mono WAV |

---

## Models

### Backbones (GGUF)

| HuggingFace repo | Language | Size |
|---|---|---|
| `neuphonic/neutts-nano-q4-gguf` | English | ~75 MB |
| `neuphonic/neutts-nano-q8-gguf` | English | ~140 MB |
| `neuphonic/neutts-air-q4-gguf` | English | ~200 MB |
| `neuphonic/neutts-air-q8-gguf` | English | ~380 MB |
| `neuphonic/neutts-nano-german-q4-gguf` | German | ~75 MB |
| `neuphonic/neutts-nano-french-q4-gguf` | French | ~75 MB |
| `neuphonic/neutts-nano-spanish-q4-gguf` | Spanish | ~75 MB |

### Codec

| Source | Format | Used for |
|---|---|---|
| `neuphonic/neucodec` (`pytorch_model.bin`) | PyTorch → safetensors | Decoder (Rust) + Encoder (Python) |

The `scripts/convert_weights.py` helper extracts and saves only the decoder
weights (~300–700 MB depending on model config).

---

## Bundled reference voices

Five pre-encoded speaker voices are included:

| File | Voice |
|---|---|
| `samples/dave.npy` | Dave |
| `samples/greta.npy` | Greta |
| `samples/jo.npy` | Jo |
| `samples/juliette.npy` | Juliette |
| `samples/mateo.npy` | Mateo |

Each `.npy` has a matching `.wav` (original audio) and `.txt` (transcript).

---

## Feature flags

| Feature | Default | Description |
|---|---|---|
| `backbone` | ✓ | GGUF backbone via `llama-cpp-2` (requires cmake + C++ compiler) |
| `espeak` | | Raw-text input via `libespeak-ng` |
| `wgpu` | | Reserved for future GPU codec acceleration (currently no-op) |
| `metal` | | macOS Metal GPU for the backbone |
| `cuda` | | NVIDIA CUDA for the backbone |

**Without `backbone`** — codec-only mode (mobile path); use `NeuCodecDecoder::decode()` directly.

**Without `espeak`** — pass pre-phonemized IPA via `tts.infer_from_ipa()`.

---

## Build requirements

| Platform | Backbone | Codec | Phonemizer |
|---|---|---|---|
| Linux / macOS | cmake + C++ (auto) | pure Rust | `libespeak-ng-dev` / `brew install espeak-ng` |
| iOS / Android | cross-compile llama.cpp | pure Rust | cross-compile espeak-ng; set `ESPEAK_LIB_DIR` |

---

## Using the library

```rust
use neutts::{NeuTTS, download};
use std::path::Path;

// Download backbone from HuggingFace (cached after first run).
// Codec weights are loaded from models/neucodec_decoder.safetensors.
let tts = download::load_from_hub("neuphonic/neutts-nano-q4-gguf").unwrap();

// Load pre-encoded reference codes
let ref_codes = tts.load_ref_codes(Path::new("ref.npy")).unwrap();

// Synthesise — returns Vec<f32> at 24 kHz mono
let audio = tts.infer(
    "Hello from Rust!",
    &ref_codes,
    "Transcript of the reference recording.",
).unwrap();

// Save to WAV
tts.write_wav(&audio, Path::new("output.wav")).unwrap();
```

### IPA passthrough (without espeak)

```rust
let audio = tts.infer_from_ipa(
    "hɛloʊ fɹʌm ɹʌst",   // input IPA
    &ref_codes,
    "wɪ ɑːɹ tɛstɪŋ ðɪs", // reference IPA
).unwrap();
```

### Decoder only

```rust
use neutts::NeuCodecDecoder;

let dec = NeuCodecDecoder::new().unwrap(); // loads models/neucodec_decoder.safetensors
println!("backend: {}", dec.backend_name()); // "cpu (ndarray)"
println!("{} samples/token", dec.hop_length());

let codes: Vec<i32> = vec![/* speech token IDs */];
let audio: Vec<f32> = dec.decode(&codes).unwrap();
```

### Reference-code cache

```rust
use neutts::RefCodeCache;

let cache = RefCodeCache::new()?;
// Returns cached codes or error if not cached — encode with Python then store
if let Some((codes, outcome)) = cache.try_load(Path::new("reference.wav"))? {
    println!("{outcome}");
}
```

---

## Mobile / C FFI

The GGUF backbone is heavy for on-device deployment.  A practical mobile
architecture runs the backbone server-side and only the NeuCodec decoder
on-device via the C FFI:

```c
// Load decoder (safetensors weights must be bundled with the app)
NeuTtsHandle *codec = neutts_model_load("/path/to/neucodec_decoder.safetensors");

// Decode tokens from server
size_t n;
float *audio = neutts_decode_tokens(codec, codes, num_codes, &n);
neutts_write_wav(audio, n, "/path/to/output.wav");
neutts_free_audio(audio, n);
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
|---|---|
| GGUF backbone inference | ✅ |
| NeuCodec decoder (pure Rust) | ✅ weights loaded from safetensors |
| NeuCodec encoder (pure Rust) | ⏳ not yet — use Python `neucodec` package |
| Multi-language backbones | ✅ German, French, Spanish |
| GPU acceleration (codec) | ⏳ planned via `wgpu` feature |
| iOS / Android build | ✅ codec is pure Rust; backbone needs cross-compile |

---

## License

MIT
