//! HuggingFace Hub model downloader.
//!
//! Downloads (or reuses cached copies of) the GGUF backbone from HuggingFace,
//! then constructs and returns a [`NeuTTS`].
//!
//! The **NeuCodec codec** is compiled into the binary at build time (via
//! `burn-import` in `build.rs`) — no runtime ONNX download is needed for the
//! decoder.  The encoder ONNX file can be downloaded with [`load_encoder`] for
//! reference audio encoding.
//!
//! Files are cached under `~/.cache/huggingface/hub`; subsequent calls return
//! immediately from cache without a network request.
//!
//! ## Default models
//!
//! | Name                | HuggingFace repo                        |
//! |---------------------|-----------------------------------------|
//! | NeuTTS-Nano Q4      | `neuphonic/neutts-nano-q4-gguf`         |
//! | NeuTTS-Nano Q8      | `neuphonic/neutts-nano-q8-gguf`         |
//! | NeuTTS-Air Q4       | `neuphonic/neutts-air-q4-gguf`          |
//! | NeuTTS-Air Q8       | `neuphonic/neutts-air-q8-gguf`          |
//! | NeuCodec Decoder    | `neuphonic/neucodec-onnx-decoder`       |
//! | NeuCodec Encoder    | `neuphonic/neucodec-onnx-encoder`       |

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use hf_hub::api::sync::Api;

#[cfg(feature = "backbone")]
use crate::model::NeuTTS;

// ─────────────────────────────────────────────────────────────────────────────
// Model registry
// ─────────────────────────────────────────────────────────────────────────────

/// Metadata for a single backbone repository.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// HuggingFace repo ID, e.g. `"neuphonic/neutts-nano-q4-gguf"`.
    pub repo: &'static str,
    /// Human-readable model name.
    pub name: &'static str,
    /// espeak-ng language code for phonemisation.
    pub language: &'static str,
    /// Approximate parameter count.
    pub params: &'static str,
    /// Whether the repo contains pre-quantised GGUF files.
    pub is_gguf: bool,
}

/// All known NeuTTS backbone repositories, ordered by language then size.
pub const BACKBONE_MODELS: &[ModelInfo] = &[
    // ── English ───────────────────────────────────────────────────────────────
    ModelInfo { repo: "neuphonic/neutts-nano-q4-gguf", name: "NeuTTS Nano Q4",      language: "en-us", params: "0.2B", is_gguf: true  },
    ModelInfo { repo: "neuphonic/neutts-nano-q8-gguf", name: "NeuTTS Nano Q8",      language: "en-us", params: "0.2B", is_gguf: true  },
    ModelInfo { repo: "neuphonic/neutts-nano",         name: "NeuTTS Nano (full)",  language: "en-us", params: "0.2B", is_gguf: false },
    ModelInfo { repo: "neuphonic/neutts-air-q4-gguf",  name: "NeuTTS Air Q4",       language: "en-us", params: "0.7B", is_gguf: true  },
    ModelInfo { repo: "neuphonic/neutts-air-q8-gguf",  name: "NeuTTS Air Q8",       language: "en-us", params: "0.7B", is_gguf: true  },
    ModelInfo { repo: "neuphonic/neutts-air",          name: "NeuTTS Air (full)",   language: "en-us", params: "0.7B", is_gguf: false },
    // ── German ────────────────────────────────────────────────────────────────
    ModelInfo { repo: "neuphonic/neutts-nano-german-q4-gguf", name: "NeuTTS Nano German Q4",     language: "de", params: "0.2B", is_gguf: true  },
    ModelInfo { repo: "neuphonic/neutts-nano-german-q8-gguf", name: "NeuTTS Nano German Q8",     language: "de", params: "0.2B", is_gguf: true  },
    ModelInfo { repo: "neuphonic/neutts-nano-german",         name: "NeuTTS Nano German (full)", language: "de", params: "0.2B", is_gguf: false },
    // ── French ────────────────────────────────────────────────────────────────
    ModelInfo { repo: "neuphonic/neutts-nano-french-q4-gguf", name: "NeuTTS Nano French Q4",     language: "fr-fr", params: "0.2B", is_gguf: true  },
    ModelInfo { repo: "neuphonic/neutts-nano-french-q8-gguf", name: "NeuTTS Nano French Q8",     language: "fr-fr", params: "0.2B", is_gguf: true  },
    ModelInfo { repo: "neuphonic/neutts-nano-french",         name: "NeuTTS Nano French (full)", language: "fr-fr", params: "0.2B", is_gguf: false },
    // ── Spanish ───────────────────────────────────────────────────────────────
    ModelInfo { repo: "neuphonic/neutts-nano-spanish-q4-gguf", name: "NeuTTS Nano Spanish Q4",     language: "es", params: "0.2B", is_gguf: true  },
    ModelInfo { repo: "neuphonic/neutts-nano-spanish-q8-gguf", name: "NeuTTS Nano Spanish Q8",     language: "es", params: "0.2B", is_gguf: true  },
    ModelInfo { repo: "neuphonic/neutts-nano-spanish",          name: "NeuTTS Nano Spanish (full)", language: "es", params: "0.2B", is_gguf: false },
];

/// Look up a [`ModelInfo`] by repo ID.  Returns `None` for unknown repos.
pub fn find_model(repo: &str) -> Option<&'static ModelInfo> {
    BACKBONE_MODELS.iter().find(|m| m.repo == repo)
}

/// espeak-ng language code for a backbone repo.
/// Falls back to `"en-us"` for unknown repos.
fn backbone_language(repo: &str) -> &'static str {
    find_model(repo).map(|m| m.language).unwrap_or("en-us")
}

// ─────────────────────────────────────────────────────────────────────────────
// Progress reporting
// ─────────────────────────────────────────────────────────────────────────────

/// Progress event emitted during model loading.
///
/// The total step count is **2** (backbone fetch + backbone load):
///
/// | Step | Event                                         |
/// |------|-----------------------------------------------|
/// | 1/2  | `Fetching` backbone GGUF                      |
/// | 2/2  | `Loading` backbone into llama.cpp             |
///
/// The Burn codec decoder is already compiled in — no download or load step.
#[derive(Debug, Clone)]
pub enum LoadProgress {
    /// About to fetch (or retrieve from cache) a model file.
    Fetching { step: u32, total: u32, file: String, repo: String },
    /// Building an inference session for a component.
    Loading { step: u32, total: u32, component: String },
}

// ─────────────────────────────────────────────────────────────────────────────
// Download helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Download a single file from a HuggingFace repository.
fn hf_download(api: &Api, repo_id: &str, filename: &str) -> Result<PathBuf> {
    let repo = api.model(repo_id.to_string());
    repo.get(filename)
        .with_context(|| format!("Failed to download '{filename}' from '{repo_id}'"))
}

/// List all files in a HuggingFace repository.
fn hf_list_files(api: &Api, repo_id: &str) -> Result<Vec<String>> {
    let repo = api.model(repo_id.to_string());
    let info = repo.info().with_context(|| format!("Failed to fetch repo info for '{repo_id}'"))?;
    Ok(info.siblings.into_iter().map(|s| s.rfilename).collect())
}

/// Find and download the first file with one of the given extensions.
fn hf_download_by_extension(
    api: &Api,
    repo_id: &str,
    extensions: &[&str],
) -> Result<PathBuf> {
    let files = hf_list_files(api, repo_id)?;
    for ext in extensions {
        if let Some(fname) = files.iter().find(|f| f.ends_with(ext)) {
            return hf_download(api, repo_id, fname);
        }
    }
    bail!(
        "No file with extension {:?} found in '{}'.\n\
         Available files: {:?}",
        extensions, repo_id, files
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Download and load a [`NeuTTS`] model from HuggingFace Hub, calling
/// `on_progress` before each step for progress reporting.
///
/// Downloads the GGUF backbone only (2 steps total).  The NeuCodec Burn
/// decoder is compiled into the binary — no runtime codec download needed.
///
/// # Arguments
///
/// * `backbone_repo` — HuggingFace repo for the GGUF backbone, e.g.
///   `"neuphonic/neutts-nano-q4-gguf"`.
/// * `gguf_file`     — Specific filename within the repo to download, e.g.
///   `Some("neutts-nano-Q4_K_M.gguf")`.  `None` picks the first `.gguf`
///   found in the repo (original behaviour).
/// * `on_progress`   — Progress callback; see [`LoadProgress`].
#[cfg(feature = "backbone")]
pub fn load_from_hub_cb<F>(
    backbone_repo: &str,
    gguf_file: Option<&str>,
    mut on_progress: F,
) -> Result<NeuTTS>
where
    F: FnMut(LoadProgress),
{
    let api = Api::new().context("Failed to initialise HuggingFace Hub client")?;

    // ── 1. Download GGUF backbone ─────────────────────────────────────────────
    let file_label = gguf_file.unwrap_or("*.gguf").to_string();
    on_progress(LoadProgress::Fetching {
        step: 1, total: 2,
        file: file_label,
        repo: backbone_repo.into(),
    });
    let backbone_path = match gguf_file {
        Some(fname) => hf_download(&api, backbone_repo, fname)
            .with_context(|| {
                format!("Failed to download '{fname}' from '{backbone_repo}'.\n\
                         \n\
                         List available files with:\n\
                         \n\
                         \tcargo run --example speak -- \
                         --backbone {backbone_repo} --list-files")
            })?,
        None => hf_download_by_extension(&api, backbone_repo, &[".gguf"])
            .with_context(|| format!("Failed to download GGUF from '{backbone_repo}'"))?,
    };

    // ── 2. Load backbone (Burn codec is compiled in) ──────────────────────────
    on_progress(LoadProgress::Loading {
        step: 2, total: 2, component: "backbone".into(),
    });
    let language = backbone_language(backbone_repo).to_string();
    NeuTTS::load(&backbone_path, &language)
}

/// Download and load a [`NeuTTS`] model from HuggingFace Hub.
///
/// Convenience wrapper around [`load_from_hub_cb`] with a no-op progress
/// callback and automatic GGUF file selection.
/// Use [`load_from_hub_cb`] to specify a particular GGUF file or for progress
/// reporting.
///
/// **Requires the `backbone` Cargo feature.**
#[cfg(feature = "backbone")]
pub fn load_from_hub(backbone_repo: &str) -> Result<NeuTTS> {
    load_from_hub_cb(backbone_repo, None, |_| {})
}

/// List all `.gguf` files available in a HuggingFace backbone repository.
///
/// Useful for discovering which quantisation variants are available before
/// calling [`load_from_hub_cb`] with a specific `gguf_file`.
pub fn list_gguf_files(backbone_repo: &str) -> Result<Vec<String>> {
    let api = Api::new().context("Failed to initialise HuggingFace Hub client")?;
    let files = hf_list_files(&api, backbone_repo)?;
    Ok(files.into_iter().filter(|f| f.ends_with(".gguf")).collect())
}

/// Load the default NeuTTS-Nano Q4 model.
///
/// **Requires the `backbone` Cargo feature.**
#[cfg(feature = "backbone")]
pub fn load_default() -> Result<NeuTTS> {
    load_from_hub("neuphonic/neutts-nano-q4-gguf")
}

/// Download the NeuCodec encoder ONNX to a specified directory.
///
/// The ONNX file is only needed if you want to encode reference audio at
/// runtime.  For the common case (using pre-encoded `.npy` reference codes),
/// you do not need the encoder.
///
/// This is also the helper used by `cargo run --example download_models` to
/// stage the ONNX for build-time Burn conversion.
pub fn download_encoder_onnx(encoder_repo: &str, dest_dir: &std::path::Path) -> Result<PathBuf> {
    let api = Api::new().context("Failed to initialise HuggingFace Hub client")?;
    let path = hf_download_by_extension(&api, encoder_repo, &[".onnx"])
        .with_context(|| format!("Failed to download encoder ONNX from '{encoder_repo}'"))?;

    // Copy to dest_dir so it can be staged for build.rs conversion.
    std::fs::create_dir_all(dest_dir)
        .context("Failed to create model staging directory")?;
    let dest = dest_dir.join("neucodec_encoder.onnx");
    std::fs::copy(&path, &dest)
        .with_context(|| format!("Failed to copy encoder ONNX to {}", dest.display()))?;
    Ok(dest)
}

/// Download the NeuCodec decoder ONNX to a specified directory.
///
/// This is used by `cargo run --example download_models` to stage the ONNX
/// for build-time Burn conversion.  You do **not** need this at runtime — the
/// Burn decoder is compiled into the binary.
pub fn download_decoder_onnx(decoder_repo: &str, dest_dir: &std::path::Path) -> Result<PathBuf> {
    let api = Api::new().context("Failed to initialise HuggingFace Hub client")?;
    let path = hf_download_by_extension(&api, decoder_repo, &[".onnx"])
        .with_context(|| format!("Failed to download decoder ONNX from '{decoder_repo}'"))?;

    std::fs::create_dir_all(dest_dir)
        .context("Failed to create model staging directory")?;
    let dest = dest_dir.join("neucodec_decoder.onnx");
    std::fs::copy(&path, &dest)
        .with_context(|| format!("Failed to copy decoder ONNX to {}", dest.display()))?;
    Ok(dest)
}

/// Load a [`NeuCodecEncoder`](crate::codec::NeuCodecEncoder) at runtime from a
/// local Burn record file (`.bin`) or an ONNX file that has been staged for
/// conversion.
///
/// `source` is resolved as follows:
///
/// 1. If it ends in `.bin` and the file exists, load as a Burn record.
/// 2. If it ends in `.onnx` and the file exists, return an error explaining
///    that ONNX files must be converted to Burn at build time via
///    `build.rs` / `burn-import` (not at runtime).
/// 3. Otherwise treat it as a HuggingFace repo ID and download the ONNX
///    to `models/`, then instruct the user to rebuild.
///
/// For the common case — embedding encoder weights at build time — call
/// [`NeuCodecEncoder::new`](crate::codec::NeuCodecEncoder::new) directly.
pub fn load_encoder(source: &str) -> Result<crate::codec::NeuCodecEncoder> {
    let path = std::path::Path::new(source);

    // Burn record file: load directly.
    if path.extension().and_then(|e| e.to_str()) == Some("bin") && path.exists() {
        return crate::codec::NeuCodecEncoder::load(path)
            .with_context(|| format!("Failed to load Burn encoder from {source}"));
    }

    // ONNX file: must be converted at build time.
    if path.extension().and_then(|e| e.to_str()) == Some("onnx") && path.exists() {
        bail!(
            "ONNX files cannot be loaded at runtime with the Burn backend.\n\
             \n\
             Stage the file for build-time conversion and rebuild:\n\
             \n\
             \tcp {source} models/neucodec_encoder.onnx\n\
             \tcargo build\n"
        );
    }

    // Try as a HuggingFace repo.
    let models_dir = std::path::Path::new("models");
    let staged = models_dir.join("neucodec_encoder.onnx");
    if !staged.exists() {
        println!("Downloading NeuCodec encoder ONNX from HuggingFace…");
        download_encoder_onnx(source, models_dir)?;
        bail!(
            "Encoder ONNX downloaded to models/neucodec_encoder.onnx.\n\
             \n\
             Rebuild to convert it to Burn:\n\
             \n\
             \tcargo build\n\
             \n\
             Then call NeuCodecEncoder::new() — no runtime file path needed."
        );
    }

    // The staged file exists but hasn't been compiled yet — guide the user.
    bail!(
        "models/neucodec_encoder.onnx is staged but the Burn model is not compiled in yet.\n\
         \n\
         Run:\n\
         \n\
         \tcargo build\n\
         \n\
         Then use NeuCodecEncoder::new() at runtime."
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Supported repo helpers
// ─────────────────────────────────────────────────────────────────────────────

/// All supported backbone repo IDs (derived from [`BACKBONE_MODELS`]).
pub fn supported_backbone_repos() -> Vec<&'static str> {
    BACKBONE_MODELS.iter().map(|m| m.repo).collect()
}

/// All GGUF-only backbone repo IDs.
pub fn supported_gguf_repos() -> Vec<&'static str> {
    BACKBONE_MODELS.iter().filter(|m| m.is_gguf).map(|m| m.repo).collect()
}

/// Return the officially supported codec decoder repository ID.
pub fn supported_codec_decoder_repo() -> &'static str {
    "neuphonic/neucodec-onnx-decoder"
}

/// Return the officially supported codec encoder repository ID.
pub fn supported_codec_encoder_repo() -> &'static str {
    "neuphonic/neucodec-onnx-encoder"
}

/// Print a formatted table of all known backbone models to stdout.
///
/// ```text
/// repo                                  name                      lang    params  gguf
/// neuphonic/neutts-nano-q4-gguf         NeuTTS Nano Q4            en-us   0.2B    yes
/// …
/// ```
pub fn print_model_table() {
    println!("{:<45} {:<28} {:<7} {:<6} {}",
        "repo", "name", "lang", "params", "gguf");
    println!("{}", "-".repeat(95));
    for m in BACKBONE_MODELS {
        println!("{:<45} {:<28} {:<7} {:<6} {}",
            m.repo, m.name, m.language, m.params,
            if m.is_gguf { "yes" } else { "no" });
    }
}
