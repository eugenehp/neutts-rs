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
// Language-code map (mirrors Python BACKBONE_LANGUAGE_MAP)
// ─────────────────────────────────────────────────────────────────────────────

/// Map from backbone HuggingFace repo ID → espeak-ng language code.
#[cfg(feature = "backbone")]
fn backbone_language(repo: &str) -> &'static str {
    match repo {
        r if r.contains("german")  => "de",
        r if r.contains("french")  => "fr-fr",
        r if r.contains("spanish") => "es",
        _                          => "en-us",
    }
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
/// * `on_progress`   — Progress callback; see [`LoadProgress`].
#[cfg(feature = "backbone")]
pub fn load_from_hub_cb<F>(
    backbone_repo: &str,
    mut on_progress: F,
) -> Result<NeuTTS>
where
    F: FnMut(LoadProgress),
{
    let api = Api::new().context("Failed to initialise HuggingFace Hub client")?;

    // ── 1. Download GGUF backbone ─────────────────────────────────────────────
    on_progress(LoadProgress::Fetching {
        step: 1, total: 2,
        file: "*.gguf".into(),
        repo: backbone_repo.into(),
    });
    let backbone_path = hf_download_by_extension(&api, backbone_repo, &[".gguf"])
        .with_context(|| format!("Failed to download GGUF from '{backbone_repo}'"))?;

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
/// callback.  Use [`load_from_hub_cb`] for progress reporting.
///
/// **Requires the `backbone` Cargo feature.**
#[cfg(feature = "backbone")]
pub fn load_from_hub(backbone_repo: &str) -> Result<NeuTTS> {
    load_from_hub_cb(backbone_repo, |_| {})
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
// Supported backbone repos list
// ─────────────────────────────────────────────────────────────────────────────

/// Return all officially supported backbone repository IDs.
pub fn supported_backbone_repos() -> Vec<&'static str> {
    vec![
        "neuphonic/neutts-air-q4-gguf",
        "neuphonic/neutts-air-q8-gguf",
        "neuphonic/neutts-nano-q4-gguf",
        "neuphonic/neutts-nano-q8-gguf",
        "neuphonic/neutts-nano-german-q4-gguf",
        "neuphonic/neutts-nano-german-q8-gguf",
        "neuphonic/neutts-nano-french-q4-gguf",
        "neuphonic/neutts-nano-french-q8-gguf",
        "neuphonic/neutts-nano-spanish-q4-gguf",
        "neuphonic/neutts-nano-spanish-q8-gguf",
    ]
}

/// Return the officially supported codec decoder repository ID.
pub fn supported_codec_decoder_repo() -> &'static str {
    "neuphonic/neucodec-onnx-decoder"
}

/// Return the officially supported codec encoder repository ID.
pub fn supported_codec_encoder_repo() -> &'static str {
    "neuphonic/neucodec-onnx-encoder"
}
