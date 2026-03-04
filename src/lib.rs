//! # neutts
//!
//! Rust port of [NeuTTS](https://github.com/neuphonic/neutts) —
//! an on-device voice-cloning TTS system based on a GGUF LLM backbone
//! and the NeuCodec neural audio codec.
//!
//! ## Architecture
//!
//! The synthesis pipeline has two models:
//!
//! 1. **GGUF backbone** — a small causal LM (NeuTTS-Nano or NeuTTS-Air in GGUF
//!    format) that takes a text prompt and pre-encoded reference codes and
//!    generates speech token IDs.
//! 2. **NeuCodec ONNX decoder** — decodes speech token IDs back to a 24 kHz
//!    audio waveform.
//!
//! ## Quick start
//!
//! ```ignore
//! // Requires features: backbone, espeak
//! use neutts::{NeuTTS, download};
//! use std::path::Path;
//!
//! // Download backbone + codec from HuggingFace (cached after first run)
//! let tts = download::load_from_hub(
//!     "neuphonic/neutts-nano-q4-gguf",
//!     "neuphonic/neucodec-onnx-decoder",
//! ).unwrap();
//!
//! // Load pre-encoded reference codes (Vec<i32>, obtained offline with Python)
//! let ref_codes = tts.load_ref_codes(Path::new("samples/jo.npy")).unwrap();
//! let ref_text  = "We are testing this model today.";
//!
//! // Generate audio (Vec<f32>, 24 kHz mono)
//! let audio = tts.infer("Hello from Rust!", &ref_codes, ref_text).unwrap();
//!
//! // Save to WAV
//! tts.write_wav(&audio, Path::new("output.wav")).unwrap();
//! ```
//!
//! ## Pre-encoding reference audio
//!
//! The reference codes must be encoded with the NeuCodec encoder, which is
//! currently only available via the Python package:
//!
//! ```python
//! from neutts import NeuTTS
//! import numpy as np
//!
//! tts = NeuTTS(codec_repo="neuphonic/neucodec")
//! codes = tts.encode_reference("reference.wav")   # tensor of i32 values
//! np.save("ref_codes.npy", codes.numpy().astype("int32"))
//! ```
//!
//! Pass the saved `.npy` path to [`NeuTTS::load_ref_codes`].
//!
//! ## Text input without espeak
//!
//! When the `espeak` feature is disabled, pass pre-phonemized IPA directly:
//!
//! ```ignore
//! // Requires feature: backbone
//! # use neutts::{NeuTTS, download};
//! # use std::path::Path;
//! # let tts = download::load_from_hub("neuphonic/neutts-nano-q4-gguf", "neuphonic/neucodec-onnx-decoder").unwrap();
//! # let ref_codes = tts.load_ref_codes(Path::new("samples/jo.npy")).unwrap();
//! // IPA obtained from an external source (e.g. a server, pre-built table, etc.)
//! let ref_ipa   = "wɪ ɑːɹ tɛstɪŋ ðɪs mɑːdl̩ tədeɪ";
//! let input_ipa = "hɛloʊ fɹʌm ɹʌst";
//! let audio = tts.infer_from_ipa(input_ipa, &ref_codes, ref_ipa).unwrap();
//! ```
//!
//! ## Build requirements
//!
//! | Platform       | Backbone            | Codec (ort)     | Phonemizer (espeak)        |
//! |----------------|---------------------|-----------------|----------------------------|
//! | Linux / macOS  | cmake + C++ (auto)  | auto-downloaded | `apt install libespeak-ng-dev` / `brew install espeak-ng` |
//! | iOS / Android  | cross-compile llama.cpp | bundled .a  | cross-compile espeak-ng; set `ESPEAK_LIB_DIR` |
//!
//! ## Features
//!
//! | Feature    | Default | Effect                                                      |
//! |------------|---------|-------------------------------------------------------------|
//! | `backbone` | ✓       | Enables GGUF backbone via llama-cpp-2 (requires cmake + C++) |
//! | `espeak`   |         | Enables raw-text input via libespeak-ng                     |
//! | `metal`    |         | macOS Metal GPU acceleration (backbone)                     |
//! | `cuda`     |         | NVIDIA CUDA acceleration (backbone)                         |

// HuggingFace Hub download — desktop only (hf-hub needs OpenSSL).
#[cfg(not(any(target_os = "ios", target_os = "android")))]
pub mod download;

// C FFI for iOS / Android.
pub mod ffi;

#[cfg(feature = "backbone")]
pub mod backbone;

pub mod codec;
pub mod model;
pub mod npy;
pub mod phonemize;
pub mod preprocess;
pub mod tokens;

// ─── Re-exports ───────────────────────────────────────────────────────────────

/// The main TTS handle — see [`download::load_from_hub`] to create one.
pub use model::NeuTTS;

/// Audio sample rate produced by NeuCodec.
pub use codec::SAMPLE_RATE;
