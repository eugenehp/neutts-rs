//! Basic NeuTTS example — downloads models and synthesises speech with voice cloning.
//!
//! ## Usage
//!
//! ```sh
//! # Default: NeuTTS-Nano Q4, reference audio from samples/
//! cargo run --example basic --features espeak
//!
//! # Custom text and reference
//! cargo run --example basic --features espeak -- \
//!   --text "The quick brown fox jumps over the lazy dog." \
//!   --ref-audio samples/jo.wav \
//!   --ref-text  samples/jo.txt \
//!   --output    output.wav
//!
//! # Different backbone (Air model)
//! cargo run --example basic --features espeak -- \
//!   --backbone neuphonic/neutts-air-q4-gguf
//! ```
//!
//! ## Requirements
//!
//! - espeak-ng installed (`brew install espeak-ng` / `apt install espeak-ng`)
//! - Reference codes as a `.npy` file (see README for how to generate)
//! - Internet access for the first run (models are cached afterwards)

use std::path::{Path, PathBuf};

fn main() -> anyhow::Result<()> {
    // ── Parse CLI arguments ───────────────────────────────────────────────────
    let mut args = std::env::args().skip(1).peekable();

    let mut backbone = "neuphonic/neutts-nano-q4-gguf".to_string();
    let mut codec    = "neuphonic/neucodec-onnx-decoder".to_string();
    let mut text     = "Hello from Rust! NeuTTS brings voice cloning to your local device.".to_string();
    let mut ref_codes_path = PathBuf::from("samples/jo.npy");
    let mut ref_text_path  = PathBuf::from("samples/jo.txt");
    let mut output   = "output.wav".to_string();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backbone"   => { if let Some(v) = args.next() { backbone = v; } }
            "--codec"      => { if let Some(v) = args.next() { codec = v; } }
            "--text"       => { if let Some(v) = args.next() { text = v; } }
            "--ref-codes"  => { if let Some(v) = args.next() { ref_codes_path = PathBuf::from(v); } }
            "--ref-audio"  => {
                // Accept a .wav path and derive the .npy path (same stem).
                if let Some(v) = args.next() {
                    let p = Path::new(&v);
                    ref_codes_path = p.with_extension("npy");
                }
            }
            "--ref-text"   => {
                if let Some(v) = args.next() {
                    ref_text_path = PathBuf::from(v);
                }
            }
            "--output" | "--out" => { if let Some(v) = args.next() { output = v; } }
            "--help" | "-h" => {
                println!(
                    "Usage: basic [OPTIONS]\n\
                     \n\
                     Options:\n\
                     \t--backbone  REPO  Backbone HuggingFace repo (default: neutts-nano-q4-gguf)\n\
                     \t--codec     REPO  Codec HuggingFace repo    (default: neucodec-onnx-decoder)\n\
                     \t--text      TEXT  Text to synthesise\n\
                     \t--ref-codes PATH  Pre-encoded reference codes (.npy file)\n\
                     \t--ref-audio PATH  Reference audio (.wav) — derives .npy from same stem\n\
                     \t--ref-text  PATH  Transcript of reference audio (.txt file or string)\n\
                     \t--output/--out PATH  Output WAV file (default: output.wav)"
                );
                return Ok(());
            }
            _ => {}
        }
    }

    // ── Check espeak-ng ───────────────────────────────────────────────────────
    if !neutts::phonemize::is_espeak_available("en-us") {
        eprintln!(
            "WARNING: espeak-ng not found.\n\
             Install with:  brew install espeak-ng          (macOS)\n\
             Or:            apt install espeak-ng           (Debian/Ubuntu)\n\
             Or:            apk add espeak-ng               (Alpine)"
        );
    }

    // ── Load reference text ───────────────────────────────────────────────────
    let ref_text = if ref_text_path.exists() {
        std::fs::read_to_string(&ref_text_path)
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    } else {
        // Treat the path as a literal string if it doesn't exist as a file.
        ref_text_path.to_string_lossy().into_owned()
    };

    if ref_text.is_empty() {
        anyhow::bail!(
            "Reference text is empty. Provide --ref-text <PATH|TEXT> or create {}",
            ref_text_path.display()
        );
    }

    // ── Print configuration ───────────────────────────────────────────────────
    println!("Backbone    : {backbone}");
    println!("Codec       : {codec}");
    println!("Text        : {text:?}");
    println!("Ref codes   : {}", ref_codes_path.display());
    println!("Ref text    : {ref_text:?}");
    println!("Output      : {output}");
    println!();

    // ── Download / load models ────────────────────────────────────────────────
    println!("Loading models…");
    let tts = neutts::download::load_from_hub_cb(&backbone, &codec, |p| {
        use neutts::download::LoadProgress;
        match &p {
            LoadProgress::Fetching { step, total, file, repo } =>
                println!("  [{step}/{total}] Fetching {file} from {repo}…"),
            LoadProgress::Loading { step, total, component } =>
                println!("  [{step}/{total}] Loading {component}…"),
        }
    })?;

    // ── Load reference codes ──────────────────────────────────────────────────
    if !ref_codes_path.exists() {
        // Collect bundled samples so the error message is helpful.
        let available: Vec<String> = std::fs::read_dir("samples")
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) == Some("npy") {
                    Some(format!("  samples/{}", p.file_name()?.to_string_lossy()))
                } else {
                    None
                }
            })
            .collect();

        let hint = if available.is_empty() {
            String::new()
        } else {
            format!(
                "\n\nBundled samples you can use with --ref-codes / --ref-audio:\n{}",
                available.join("\n")
            )
        };

        anyhow::bail!(
            "Reference codes file not found: {}{}\n\
             \nTo generate your own, run in Python:\n\
             \n\
             \tfrom neutts import NeuTTS\n\
             \timport numpy as np\n\
             \ttts = NeuTTS(codec_repo='neuphonic/neucodec')\n\
             \tcodes = tts.encode_reference('reference.wav')\n\
             \tnp.save('ref_codes.npy', codes.numpy().astype('int32'))\n",
            ref_codes_path.display(),
            hint,
        );
    }
    println!("Loading reference codes from {}…", ref_codes_path.display());
    let ref_codes = tts.load_ref_codes(&ref_codes_path)?;
    println!("  → {} codec tokens (~{:.1} s of reference audio)",
        ref_codes.len(), ref_codes.len() as f32 / 50.0);

    // ── Synthesise ────────────────────────────────────────────────────────────
    println!("\nSynthesising speech…");
    let audio = tts.infer(&text, &ref_codes, &ref_text)?;
    println!(
        "  → Generated {} samples ({:.2} s)",
        audio.len(),
        audio.len() as f32 / neutts::SAMPLE_RATE as f32,
    );

    // ── Save WAV ──────────────────────────────────────────────────────────────
    tts.write_wav(&audio, Path::new(&output))?;
    println!("\nDone! Saved to '{output}'");
    Ok(())
}
