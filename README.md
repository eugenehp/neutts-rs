# neutts-rs

Rust port of [NeuTTS](https://github.com/neuphonic/neutts) — on-device voice-cloning TTS
built on a GGUF LLM backbone and the NeuCodec neural audio codec.

_Created by [Neuphonic](https://neuphonic.com/) — building faster, smaller, on-device voice AI._

## Quick start

```rust
use neutts::{NeuTTS, download};
use std::path::Path;

// Download backbone + codec from HuggingFace (cached after first run)
let tts = download::load_from_hub(
    "neuphonic/neutts-nano-q4-gguf",
    "neuphonic/neucodec-onnx-decoder",
).unwrap();

// Load pre-encoded reference codes (see "Pre-encoding reference audio" below)
let ref_codes = tts.load_ref_codes(Path::new("samples/jo.npy")).unwrap();
let ref_text  = "We are testing this model today.";

// Synthesise — returns Vec<f32> at 24 kHz mono
let audio = tts.infer("Hello from Rust!", &ref_codes, ref_text).unwrap();

// Save to WAV
tts.write_wav(&audio, Path::new("output.wav")).unwrap();
```

## Models

### Backbones (GGUF)

| HuggingFace repo                              | Language | Size  |
|-----------------------------------------------|----------|-------|
| `neuphonic/neutts-air-q4-gguf`                | English  | ~200 MB |
| `neuphonic/neutts-air-q8-gguf`                | English  | ~380 MB |
| `neuphonic/neutts-nano-q4-gguf`               | English  | ~75 MB  |
| `neuphonic/neutts-nano-q8-gguf`               | English  | ~140 MB |
| `neuphonic/neutts-nano-german-q4-gguf`        | German   | ~75 MB  |
| `neuphonic/neutts-nano-french-q4-gguf`        | French   | ~75 MB  |
| `neuphonic/neutts-nano-spanish-q4-gguf`       | Spanish  | ~75 MB  |

### Codecs (ONNX)

| HuggingFace repo                              | Notes              |
|-----------------------------------------------|--------------------|
| `neuphonic/neucodec-onnx-decoder`             | Full precision     |
| `neuphonic/neucodec-onnx-decoder-int8`        | Faster, smaller    |

## Architecture

```
text ──► phonemize (espeak-ng) ──► prompt builder ──► GGUF backbone ──► speech tokens
                                         ▲                                     │
                                  ref_codes + ref_text                         ▼
                                                                    NeuCodec ONNX decoder
                                                                               │
                                                                               ▼
                                                                    audio (Vec<f32>, 24 kHz)
```

The synthesis pipeline has two models:

1. **GGUF backbone** — a small causal LM (NeuTTS-Nano or NeuTTS-Air) that takes a phonemized
   text prompt and pre-encoded reference speaker codes, then generates speech token IDs.

2. **NeuCodec ONNX decoder** — a 50 Hz neural audio codec that converts speech token IDs
   back to a 24 kHz waveform (480 samples per token).

## Pre-encoding reference audio

The reference codes (speaker embedding for voice cloning) must be computed with the NeuCodec
encoder, which is currently only available in Python:

```python
from neutts import NeuTTS
import numpy as np

tts = NeuTTS(codec_repo="neuphonic/neucodec")
codes = tts.encode_reference("reference.wav")           # -> int tensor, ~50 tokens/sec
np.save("ref_codes.npy", codes.numpy().astype("int32"))
```

Pass the saved `.npy` path to `NeuTTS::load_ref_codes()`, or supply it on the
command line via `--ref-codes` (or `--ref-audio` with the matching `.wav`).

### Bundled samples

Five pre-encoded speaker voices ship with the repo and are ready to use immediately:

| File                   | Voice    |
|------------------------|----------|
| `samples/dave.npy`     | Dave     |
| `samples/greta.npy`    | Greta    |
| `samples/jo.npy`       | Jo       |
| `samples/juliette.npy` | Juliette |
| `samples/mateo.npy`    | Mateo    |

Each `.npy` file has a matching `.wav` (the original reference audio) and `.txt`
(its transcript).  If you pass a path that doesn't exist, the CLI will print the
full list of available samples automatically.

## Running the `basic` example

```sh
# Minimal — uses the Jo voice by default
cargo run --example basic --features espeak -- \
  --text "Hello from Rust."

# Choose a different bundled voice
cargo run --example basic --features espeak -- \
  --text "Hello from Rust." \
  --ref-audio samples/dave.wav \
  --out output.wav

# Your own reference audio (needs a pre-encoded .npy — see above)
cargo run --example basic --features espeak -- \
  --text "Hello from Rust." \
  --ref-codes path/to/my_voice.npy \
  --ref-text  "Transcript of the reference recording." \
  --out output.wav
```

### CLI flags

| Flag | Alias | Default | Description |
|------|-------|---------|-------------|
| `--backbone` | | `neuphonic/neutts-nano-q4-gguf` | HuggingFace backbone repo |
| `--codec` | | `neuphonic/neucodec-onnx-decoder` | HuggingFace codec repo |
| `--text` | | built-in demo sentence | Text to synthesise |
| `--ref-codes` | | `samples/jo.npy` | Pre-encoded reference codes (`.npy`) |
| `--ref-audio` | | | Reference `.wav` — derives the `.npy` path from the same stem |
| `--ref-text` | | `samples/jo.txt` | Transcript of the reference audio (file path or literal string) |
| `--output` | `--out` | `output.wav` | Output WAV file |

## Build requirements

| Platform      | Backbone (llama.cpp)          | Codec (ort)      | Phonemizer (espeak-ng) |
|---------------|-------------------------------|------------------|------------------------|
| Linux         | cmake + GCC/Clang (auto-build)| auto-downloaded  | `apt install libespeak-ng-dev` |
| macOS         | Xcode CLI tools (auto-build)  | auto-downloaded  | `brew install espeak-ng` |
| iOS           | cross-compile llama.cpp       | bundled `.a`     | cross-compile espeak-ng; set `ESPEAK_LIB_DIR` |
| Android       | cross-compile llama.cpp       | bundled `.so`    | cross-compile espeak-ng; set `ESPEAK_LIB_DIR` |

## Feature flags

| Feature    | Default | Description                                                      |
|------------|---------|------------------------------------------------------------------|
| `backbone` | ✓       | GGUF backbone via llama-cpp-2 (requires cmake + C++ compiler)    |
| `espeak`   |         | Raw-text input via libespeak-ng                                  |
| `metal`    |         | macOS Metal GPU acceleration for the backbone                    |
| `cuda`     |         | NVIDIA CUDA acceleration for the backbone                        |

**Without `espeak`** — use `NeuTTS::infer_from_ipa()` to pass pre-phonemized IPA strings.

**Without `backbone`** — use `NeuTTS::decode_tokens()` for codec-only decoding (mobile path).

## Skip phonemisation

When the `espeak` feature is disabled, pass IPA directly:

```rust
// IPA from a server, pre-built table, or a different G2P library.
let ref_ipa   = "wɪ ɑːɹ tɛstɪŋ ðɪs mɑːdl̩ tədeɪ";
let input_ipa = "hɛloʊ fɹʌm ɹʌst";
let audio = tts.infer_from_ipa(input_ipa, &ref_codes, ref_ipa).unwrap();
```

## Mobile (iOS / Android)

The GGUF backbone can be heavy for on-device deployment.  A practical mobile
architecture has the backbone run server-side and only the NeuCodec ONNX decoder
run on-device via the C FFI:

```c
// App startup
neutts_set_espeak_data_path("/path/to/espeak-ng-data");
NeuTtsHandle *codec = neutts_model_load("/path/to/neucodec_decoder.onnx");

// Each utterance (codes arrive from server)
size_t n;
float *audio = neutts_decode_tokens(codec, codes, num_codes, &n);
neutts_write_wav(audio, n, "/path/to/output.wav");
neutts_free_audio(audio, n);

// Shutdown
neutts_model_free(codec);
```

See [`include/neutts.h`](include/neutts.h) for the full C header.

## Pipeline (matching Python neutts)

1. **Text preprocessing** — numbers, currencies, abbreviations → spoken words.
2. **Phonemisation** — espeak-ng converts text to IPA phonemes.
3. **Prompt construction** — reference codes + IPA phonemes → GGUF prompt.
4. **Backbone inference** — GGUF LLM generates `<|speech_N|>` tokens.
5. **Token extraction** — regex extracts integer IDs from generated text.
6. **Codec decode** — NeuCodec ONNX decoder converts IDs to 24 kHz audio.

## License

Apache-2.0
