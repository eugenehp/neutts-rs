//! NeuCodec ONNX decoder — converts speech token IDs to a 24 kHz audio waveform.
//!
//! Wraps the `neuphonic/neucodec-onnx-decoder` (or its int8 variant) ONNX model
//! using [`ort`] (ONNX Runtime Rust bindings).
//!
//! ## Model interface
//!
//! | Input  | Name     | Shape       | dtype |
//! |--------|----------|-------------|-------|
//! | 0      | `codes`  | `[1, 1, T]` | int32 |
//!
//! | Output | Name    | Shape            | dtype   |
//! |--------|---------|------------------|---------|
//! | 0      | `audio` | `[1, 1, T_audio]`| float32 |
//!
//! The codec runs at 50 Hz: each input token corresponds to
//! `24 000 / 50 = 480` output audio samples.

use std::{path::Path, sync::Mutex};

use anyhow::{Context, Result};
use ort::{session::Session, value::Tensor};

/// Audio sample rate produced by NeuCodec.
pub const SAMPLE_RATE: u32 = 24_000;

/// NeuCodec ONNX decoder handle.
pub struct NeuCodecDecoder {
    session: Mutex<Session>,
}

impl NeuCodecDecoder {
    /// Load the decoder from an ONNX file on disk.
    pub fn load(model_path: &Path) -> Result<Self> {
        let session = Session::builder()
            .context("Failed to create ORT session builder")?
            .commit_from_file(model_path)
            .with_context(|| format!("Cannot load ONNX codec: {}", model_path.display()))?;
        Ok(Self { session: Mutex::new(session) })
    }

    /// Decode a sequence of NeuCodec token IDs to a mono audio waveform.
    ///
    /// `codes` — integer token IDs in the range `[0, 1023]`.
    ///
    /// Returns a flat `Vec<f32>` at [`SAMPLE_RATE`] Hz (24 kHz).
    pub fn decode(&self, codes: &[i32]) -> Result<Vec<f32>> {
        if codes.is_empty() {
            return Ok(Vec::new());
        }
        let seq_len = codes.len();

        // Build input tensor: shape [1, 1, T], dtype int32.
        let t_codes = Tensor::<i32>::from_array((
            [1usize, 1usize, seq_len],
            codes.to_vec(),
        ))
        .context("Failed to build codes tensor")?;

        // Run inference.
        let mut session = self.session.lock().expect("ORT session mutex poisoned");
        let outputs = session
            .run(ort::inputs![t_codes])
            .context("NeuCodec ONNX inference failed")?;

        // Output 0: audio waveform (shape [1, 1, T_audio] or [1, T_audio] or [T_audio]).
        let (_shape, audio_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract audio tensor from codec output")?;

        Ok(audio_data.to_vec())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_rate() {
        assert_eq!(SAMPLE_RATE, 24_000);
    }
}
