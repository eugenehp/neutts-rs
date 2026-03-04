//! NeuTTS model — ties the GGUF backbone and NeuCodec ONNX decoder together.
//!
//! ## Pipeline
//!
//! ```text
//!            ┌──────────────┐
//!  text ─────►  phonemize   │  (espeak feature)
//!            └──────┬───────┘
//!                   │ IPA
//!            ┌──────▼───────────────────────────────────┐
//!  ref_codes ►   build_prompt()  →  GGUF backbone       │  (backbone feature)
//!            └──────────────────────┬───────────────────┘
//!                                   │ generated text
//!            ┌──────────────────────▼────────────────┐
//!            │  tokens::extract_ids()                │
//!            └──────────────────────┬────────────────┘
//!                                   │ speech token IDs
//!            ┌──────────────────────▼────────────────┐
//!            │  NeuCodec ONNX decoder                │
//!            └──────────────────────┬────────────────┘
//!                                   │ audio (Vec<f32>, 24 kHz)
//! ```

use std::path::Path;

use anyhow::{Context, Result};

use crate::codec::{NeuCodecDecoder, SAMPLE_RATE};
use crate::npy;

#[cfg(feature = "backbone")]
use crate::tokens;

#[cfg(all(feature = "backbone", feature = "espeak"))]
use crate::phonemize;

#[cfg(feature = "backbone")]
use crate::backbone::{BackboneModel, DEFAULT_N_CTX};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Generation hyper-parameters passed to the backbone.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of *new* tokens the backbone may generate.
    /// Maps to Python `max_context = 2048`.
    pub max_new_tokens: u32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self { max_new_tokens: 2048 }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NeuTTS
// ─────────────────────────────────────────────────────────────────────────────

/// The main NeuTTS handle.
///
/// Obtained via [`crate::download::load_from_hub`] (desktop) or constructed
/// directly with [`NeuTTS::load`] (mobile / offline).
pub struct NeuTTS {
    /// GGUF LLM backbone — generates speech tokens from a text prompt.
    #[cfg(feature = "backbone")]
    pub backbone: BackboneModel,

    /// NeuCodec ONNX decoder — converts speech token IDs to audio.
    pub codec: NeuCodecDecoder,

    /// espeak-ng language code (e.g. `"en-us"`, `"de"`, `"fr-fr"`, `"es"`).
    pub language: String,

    /// Generation hyper-parameters.
    pub config: GenerationConfig,
}

impl NeuTTS {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Load NeuTTS from files on disk.
    ///
    /// # Arguments
    ///
    /// * `backbone_path` — Path to a NeuTTS GGUF model file.
    ///   **Requires the `backbone` Cargo feature.**
    /// * `codec_path`    — Path to the NeuCodec ONNX decoder file.
    /// * `language`      — espeak-ng language code (e.g. `"en-us"`).
    ///   Used only when the `espeak` feature is enabled.
    #[cfg(feature = "backbone")]
    pub fn load(backbone_path: &Path, codec_path: &Path, language: &str) -> Result<Self> {
        println!("Loading backbone: {}", backbone_path.display());
        let backbone = BackboneModel::load(backbone_path, DEFAULT_N_CTX)
            .context("Failed to load backbone")?;

        println!("Loading codec: {}", codec_path.display());
        let codec = NeuCodecDecoder::load(codec_path)
            .context("Failed to load NeuCodec ONNX decoder")?;

        Ok(Self {
            backbone,
            codec,
            language: language.to_string(),
            config: GenerationConfig::default(),
        })
    }

    /// Load NeuTTS with only the codec (no backbone).
    ///
    /// Use this when you already have generated speech token IDs and only need
    /// the ONNX decoder step (e.g. on mobile where the backbone is called
    /// server-side).
    ///
    /// When the `backbone` feature is enabled, use [`NeuTTS::load`] instead.
    #[cfg(not(feature = "backbone"))]
    pub fn load_codec_only(codec_path: &Path) -> Result<Self> {
        let codec = NeuCodecDecoder::load(codec_path)
            .context("Failed to load NeuCodec ONNX decoder")?;
        Ok(Self {
            codec,
            language: "en-us".to_string(),
            config: GenerationConfig::default(),
        })
    }

    // ── Reference code loading ────────────────────────────────────────────────

    /// Load pre-encoded NeuCodec reference codes from a `.npy` file.
    ///
    /// The NPY file should contain a 1-D int32 array of codec token IDs (values
    /// 0–1023), saved with:
    ///
    /// ```python
    /// import numpy as np
    /// np.save("ref_codes.npy", tts.encode_reference("ref.wav").numpy().astype("int32"))
    /// ```
    pub fn load_ref_codes(&self, path: &Path) -> Result<Vec<i32>> {
        npy::load_npy_i32(path)
            .with_context(|| format!("Failed to load reference codes: {}", path.display()))
    }

    // ── Synthesis ─────────────────────────────────────────────────────────────

    /// Generate audio from `text` using `ref_codes` and `ref_text` for voice
    /// cloning.
    ///
    /// Phonemises both `text` and `ref_text` with espeak-ng, builds the prompt,
    /// runs the GGUF backbone, decodes speech tokens with NeuCodec.
    ///
    /// **Requires both the `backbone` and `espeak` Cargo features.**
    ///
    /// Returns a flat `Vec<f32>` at [`SAMPLE_RATE`] Hz (24 kHz, mono).
    ///
    /// # Arguments
    ///
    /// * `text`      — The utterance to synthesise.
    /// * `ref_codes` — Pre-encoded NeuCodec codes of the reference audio
    ///                 (see [`load_ref_codes`](Self::load_ref_codes)).
    /// * `ref_text`  — Transcript of the reference audio (used for voice cloning).
    #[cfg(all(feature = "backbone", feature = "espeak"))]
    pub fn infer(&self, text: &str, ref_codes: &[i32], ref_text: &str) -> Result<Vec<f32>> {
        let ref_phones   = phonemize::phonemize(ref_text, &self.language)
            .context("Phonemisation of ref_text failed")?;
        let input_phones = phonemize::phonemize(text, &self.language)
            .context("Phonemisation of input text failed")?;

        self.infer_from_ipa(&input_phones, ref_codes, &ref_phones)
    }

    /// Generate audio from pre-phonemized IPA strings.
    ///
    /// Use this when espeak-ng is unavailable or when you want to supply IPA
    /// from a different G2P pipeline.
    ///
    /// **Requires the `backbone` Cargo feature.**
    ///
    /// # Arguments
    ///
    /// * `input_ipa` — IPA phonemes for the text to synthesise.
    /// * `ref_codes` — Pre-encoded NeuCodec codes for the reference speaker.
    /// * `ref_ipa`   — IPA phonemes for the reference audio transcript.
    #[cfg(feature = "backbone")]
    pub fn infer_from_ipa(
        &self,
        input_ipa: &str,
        ref_codes: &[i32],
        ref_ipa: &str,
    ) -> Result<Vec<f32>> {
        // Build prompt.
        let prompt = tokens::build_prompt(ref_ipa, input_ipa, ref_codes);

        // Run backbone to generate speech token string.
        let generated = self.backbone
            .generate(&prompt, self.config.max_new_tokens)
            .context("Backbone generation failed")?;

        // Extract speech token IDs.
        let speech_ids = tokens::extract_ids(&generated);
        if speech_ids.is_empty() {
            anyhow::bail!(
                "No speech tokens found in backbone output.\n\
                 Prompt may have exceeded the context window, or the model produced no output.\n\
                 Generated text snippet: {:?}",
                &generated[..generated.len().min(200)]
            );
        }

        // Decode with NeuCodec.
        self.codec.decode(&speech_ids)
            .context("NeuCodec decode failed")
    }

    /// Decode a pre-generated sequence of speech token IDs directly to audio.
    ///
    /// This is the mobile / streaming path: the backbone runs server-side and
    /// the client only calls the codec.
    pub fn decode_tokens(&self, speech_ids: &[i32]) -> Result<Vec<f32>> {
        self.codec.decode(speech_ids).context("NeuCodec decode failed")
    }

    // ── WAV output ────────────────────────────────────────────────────────────

    /// Write `audio` samples to a 16-bit PCM WAV file at [`SAMPLE_RATE`] Hz.
    ///
    /// 16-bit PCM is used for maximum compatibility (Android `MediaPlayer`
    /// does not reliably handle IEEE-float WAV at runtime).
    pub fn write_wav(&self, audio: &[f32], output_path: &Path) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(output_path, spec)
            .with_context(|| format!("Cannot create WAV: {}", output_path.display()))?;
        for &s in audio {
            let s16 = (s * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            writer.write_sample(s16).context("WAV write error")?;
        }
        writer.finalize().context("WAV finalise error")?;
        println!(
            "Saved {} samples ({:.2} s) to {}",
            audio.len(),
            audio.len() as f32 / SAMPLE_RATE as f32,
            output_path.display()
        );
        Ok(())
    }

    // ── High-level convenience wrappers ───────────────────────────────────────

    /// Generate audio and save to a WAV file in one call.
    ///
    /// **Requires `backbone` + `espeak` features.**
    #[cfg(all(feature = "backbone", feature = "espeak"))]
    pub fn infer_to_file(
        &self,
        text: &str,
        ref_codes: &[i32],
        ref_text: &str,
        output_path: &Path,
    ) -> Result<()> {
        let audio = self.infer(text, ref_codes, ref_text)?;
        self.write_wav(&audio, output_path)
    }
}
