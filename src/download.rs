//! HuggingFace Hub model downloader.
//!
//! Downloads (or reuses cached copies of) the GGUF backbone and the NeuCodec
//! ONNX decoder from HuggingFace, then constructs and returns a [`NeuTTS`].
//!
//! Files are cached under `~/.cache/huggingface/hub`; subsequent calls return
//! immediately from cache without a network request.
//!
//! ## Default models
//!
//! | Name                | HuggingFace repo                                  |
//! |---------------------|---------------------------------------------------|
//! | NeuTTS-Nano Q4      | `neuphonic/neutts-nano-q4-gguf`                   |
//! | NeuTTS-Nano Q8      | `neuphonic/neutts-nano-q8-gguf`                   |
//! | NeuTTS-Air Q4       | `neuphonic/neutts-air-q4-gguf`                    |
//! | NeuTTS-Air Q8       | `neuphonic/neutts-air-q8-gguf`                    |
//! | NeuCodec Decoder    | `neuphonic/neucodec-onnx-decoder`                 |
//! | NeuCodec Dec. (int8)| `neuphonic/neucodec-onnx-decoder-int8`            |

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
/// The total step count is **4**:
///
/// | Step | Event                                         |
/// |------|-----------------------------------------------|
/// | 1/4  | `Fetching` backbone GGUF                      |
/// | 2/4  | `Fetching` NeuCodec ONNX                      |
/// | 3/4  | `Loading` backbone into llama.cpp             |
/// | 4/4  | `Loading` codec into ONNX Runtime             |
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
/// # Arguments
///
/// * `backbone_repo` — HuggingFace repo for the GGUF backbone, e.g.
///   `"neuphonic/neutts-nano-q4-gguf"`.
/// * `codec_repo`    — HuggingFace repo for the NeuCodec ONNX decoder, e.g.
///   `"neuphonic/neucodec-onnx-decoder"`.
/// * `on_progress`   — Progress callback; see [`LoadProgress`].
///
/// # Example
///
/// ```no_run
/// let model = neutts::download::load_from_hub_cb(
///     "neuphonic/neutts-nano-q4-gguf",
///     "neuphonic/neucodec-onnx-decoder",
///     |p| println!("{p:?}"),
/// ).unwrap();
/// ```
#[cfg(feature = "backbone")]
pub fn load_from_hub_cb<F>(
    backbone_repo: &str,
    codec_repo: &str,
    mut on_progress: F,
) -> Result<NeuTTS>
where
    F: FnMut(LoadProgress),
{
    let api = Api::new().context("Failed to initialise HuggingFace Hub client")?;

    // ── 1. Download GGUF backbone ─────────────────────────────────────────────
    on_progress(LoadProgress::Fetching {
        step: 1, total: 4,
        file: "*.gguf".into(),
        repo: backbone_repo.into(),
    });
    let backbone_path = hf_download_by_extension(&api, backbone_repo, &[".gguf"])
        .with_context(|| format!("Failed to download GGUF from '{backbone_repo}'"))?;

    // ── 2. Download NeuCodec ONNX decoder ─────────────────────────────────────
    on_progress(LoadProgress::Fetching {
        step: 2, total: 4,
        file: "*.onnx".into(),
        repo: codec_repo.into(),
    });
    let codec_path = hf_download_by_extension(&api, codec_repo, &[".onnx"])
        .with_context(|| format!("Failed to download ONNX from '{codec_repo}'"))?;

    // ── 3 + 4. Load both models ───────────────────────────────────────────────
    on_progress(LoadProgress::Loading {
        step: 3, total: 4, component: "backbone".into(),
    });
    let language = backbone_language(backbone_repo).to_string();

    on_progress(LoadProgress::Loading {
        step: 4, total: 4, component: "codec".into(),
    });
    NeuTTS::load(&backbone_path, &codec_path, &language)
}

/// Download and load a [`NeuTTS`] model from HuggingFace Hub.
///
/// Convenience wrapper around [`load_from_hub_cb`] with a no-op progress
/// callback.  Use [`load_from_hub_cb`] for progress reporting.
///
/// **Requires the `backbone` Cargo feature.**
#[cfg(feature = "backbone")]
pub fn load_from_hub(backbone_repo: &str, codec_repo: &str) -> Result<NeuTTS> {
    load_from_hub_cb(backbone_repo, codec_repo, |_| {})
}

/// Load the default NeuTTS-Nano Q4 model with the default NeuCodec ONNX decoder.
///
/// **Requires the `backbone` Cargo feature.**
#[cfg(feature = "backbone")]
pub fn load_default() -> Result<NeuTTS> {
    load_from_hub(
        "neuphonic/neutts-nano-q4-gguf",
        "neuphonic/neucodec-onnx-decoder",
    )
}

/// Download only the NeuCodec ONNX decoder (no backbone).
///
/// Useful on mobile where the backbone runs server-side and only the local
/// codec is needed.
pub fn load_codec_only(codec_repo: &str) -> Result<crate::codec::NeuCodecDecoder> {
    let api = Api::new().context("Failed to initialise HuggingFace Hub client")?;
    let codec_path = hf_download_by_extension(&api, codec_repo, &[".onnx"])
        .with_context(|| format!("Failed to download ONNX from '{codec_repo}'"))?;
    crate::codec::NeuCodecDecoder::load(&codec_path)
        .context("Failed to load NeuCodec ONNX decoder")
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

/// Return all officially supported codec repository IDs.
pub fn supported_codec_repos() -> Vec<&'static str> {
    vec![
        "neuphonic/neucodec-onnx-decoder",
        "neuphonic/neucodec-onnx-decoder-int8",
    ]
}
