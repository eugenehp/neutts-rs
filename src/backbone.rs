//! GGUF backbone — runs the NeuTTS LLM that generates speech token IDs.
//!
//! Wraps [`llama_cpp_2`] (Rust bindings to llama.cpp) to load a GGUF model
//! and run token generation with temperature + top-k sampling.
//!
//! ## Pipeline
//!
//! 1. Prompt (text + reference codes) is tokenised by the GGUF model's
//!    built-in tokeniser (which includes the special `<|speech_N|>` tokens).
//! 2. Prompt tokens are fed into the KV cache via `ctx.decode()`.
//! 3. New tokens are sampled (temperature=1.0, top-k=50) until the model
//!    emits `<|SPEECH_GENERATION_END|>` or the context limit is reached.
//! 4. The generated text is returned; the caller extracts speech token IDs
//!    with [`crate::tokens::extract_ids`].

use std::num::NonZeroU32;
use std::path::Path;

use anyhow::{Context, Result};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    sampling::LlamaSampler,
};

use crate::tokens::STOP_TOKEN;

/// Default context window (must match Python's `max_context = 2048`).
pub const DEFAULT_N_CTX: u32 = 2048;

/// NeuTTS GGUF backbone model.
///
/// Holds the loaded [`LlamaModel`] and configuration.  A new [`LlamaContext`]
/// is created for each [`generate`](BackboneModel::generate) call so there is
/// no cross-inference state leakage.
pub struct BackboneModel {
    /// llama.cpp backend handle — must outlive the model.
    _backend: LlamaBackend,
    /// Loaded GGUF model.
    model: LlamaModel,
    /// Context window size (tokens).
    n_ctx: u32,
    /// Random seed for the sampler.  `None` → a fresh random seed per call.
    pub seed: Option<u32>,
}

impl BackboneModel {
    /// Load a GGUF model from `path`.
    ///
    /// `n_ctx` — context window size.  Pass [`DEFAULT_N_CTX`] for the default.
    ///
    /// The backbone uses all available CPU threads by default.  Enable the
    /// `metal` or `cuda` Cargo features for GPU acceleration.
    pub fn load(path: &Path, n_ctx: u32) -> Result<Self> {
        let mut backend = LlamaBackend::init()
            .context("Failed to initialise llama.cpp backend")?;

        // Silence llama.cpp / ggml stderr spam unless the `verbose` feature is on.
        #[cfg(not(feature = "verbose"))]
        backend.void_logs();
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, path, &model_params)
            .with_context(|| format!("Cannot load GGUF model: {}", path.display()))?;
        Ok(Self { _backend: backend, model, n_ctx, seed: None })
    }

    /// Run the backbone on `prompt` and return the generated token string.
    ///
    /// Stops when the model produces `<|SPEECH_GENERATION_END|>` or when
    /// `max_new_tokens` tokens have been generated (whichever comes first).
    /// The stop token itself is **not** included in the returned string.
    ///
    /// Use [`crate::tokens::extract_ids`] on the returned string to get the
    /// integer speech token IDs.
    pub fn generate(&self, prompt: &str, max_new_tokens: u32) -> Result<String> {
        // ── Create a fresh context for this inference ─────────────────────────
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx));
        let mut ctx = self.model
            .new_context(&self._backend, ctx_params)
            .context("Failed to create llama.cpp context")?;

        // ── Tokenise prompt ───────────────────────────────────────────────────
        let tokens = self.model
            .str_to_token(prompt, AddBos::Always)
            .context("Tokenisation failed")?;

        if tokens.is_empty() {
            return Ok(String::new());
        }

        // ── Fill the KV cache with the prompt ─────────────────────────────────
        let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
        let last_idx = tokens.len() - 1;
        for (i, &tok) in tokens.iter().enumerate() {
            batch
                .add(tok, i as i32, &[0], i == last_idx)
                .context("Failed to add token to batch")?;
        }
        ctx.decode(&mut batch).context("Prompt decode failed")?;

        // ── Sampler: top-k(50) → temperature(1.0) → random distribution ───────
        // llama-cpp-2 v0.1+ API: static constructors + chain_simple([...])
        let seed = self.seed.unwrap_or_else(rand::random::<u32>);
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(50),
            LlamaSampler::temp(1.0),
            LlamaSampler::dist(seed),
        ]);

        // ── Generation loop ───────────────────────────────────────────────────
        let mut n_cur = tokens.len() as i32;
        let max_tokens = n_cur + max_new_tokens as i32;
        let mut output = String::new();

        loop {
            // Sample the next token.
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            // End-of-generation token (EOS / EOT)?
            if self.model.is_eog_token(token) {
                break;
            }

            // Decode token bytes → UTF-8 string.
            // token_to_piece_bytes(token, buf_size, special=true, lstrip=None)
            let piece = token_to_piece(&self.model, token)?;
            output.push_str(&piece);

            // Stop at the explicit NeuTTS stop token.
            if let Some(pos) = output.find(STOP_TOKEN) {
                output.truncate(pos);
                break;
            }

            // Context limit reached?
            if n_cur >= max_tokens {
                break;
            }

            // Feed the new token back for the next step.
            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .context("Failed to add generated token to batch")?;
            ctx.decode(&mut batch).context("Decode step failed")?;
            n_cur += 1;
        }

        Ok(output)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Decode a single token to a UTF-8 string using the non-deprecated
/// `token_to_piece_bytes` API (llama-cpp-2 ≥ 0.1.x).
///
/// `special = true` ensures that special tokens like `<|speech_N|>` are
/// rendered as their text representation rather than as a placeholder byte.
fn token_to_piece(model: &LlamaModel, token: llama_cpp_2::token::LlamaToken) -> Result<String> {
    use llama_cpp_2::TokenToStringError;

    // Start with a 64-byte buffer; retry with the exact size if too small.
    let bytes = match model.token_to_piece_bytes(token, 64, true, None) {
        Ok(b) => b,
        Err(TokenToStringError::InsufficientBufferSpace(needed)) => {
            let size = needed.unsigned_abs() as usize + 1;
            model.token_to_piece_bytes(token, size, true, None)
                .context("token_to_piece_bytes retry failed")?
        }
        Err(e) => return Err(anyhow::anyhow!("token decode error: {e}")),
    };

    // The bytes are typically valid UTF-8 but may contain lone bytes for
    // partial multi-byte sequences mid-stream; use lossy conversion.
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}
