//! Phonemisation using the `libespeak-ng` C library.
//!
//! Mirrors the Python `BasePhonemizer` + `FrenchPhonemizer` from `neutts/phonemizers.py`.
//! The full implementation is compiled only when the **`espeak`** Cargo feature
//! is enabled.  Without it, all public functions are still present but return
//! informative errors so downstream callers using `generate_from_ipa()` compile
//! without a system libespeak-ng.
//!
//! ## Supported languages
//!
//! | Language   | Code    | Notes                                      |
//! |------------|---------|--------------------------------------------|
//! | English    | `en-us` | Default for NeuTTS-Nano / NeuTTS-Air       |
//! | German     | `de`    | NeuTTS-Nano-German                         |
//! | French     | `fr-fr` | NeuTTS-Nano-French; dashes stripped        |
//! | Spanish    | `es`    | NeuTTS-Nano-Spanish                        |
//!
//! ## Enabling
//!
//! ```toml
//! neutts = { version = "…", features = ["espeak"] }
//! ```
//!
//! ## Build requirements (when `espeak` feature is on)
//!
//! | Platform       | Requirement                                              |
//! |----------------|----------------------------------------------------------|
//! | Alpine / Linux | `apk add espeak-ng-dev` / `apt install libespeak-ng-dev` |
//! | macOS          | `brew install espeak-ng`                                 |
//! | iOS / Android  | Cross-compiled `libespeak-ng.{a,so}`; set `ESPEAK_LIB_DIR` at build time and [`set_data_path`] at runtime |

use std::path::{Path, PathBuf};
use anyhow::Result;
#[cfg(not(feature = "espeak"))]
use anyhow::anyhow;
use once_cell::sync::OnceCell;

// ─── Runtime data-path ────────────────────────────────────────────────────────

static DATA_PATH: OnceCell<PathBuf> = OnceCell::new();

/// Set the path to the `espeak-ng-data` directory.
///
/// **Required on iOS and Android**: bundle `espeak-ng-data/` with the app and
/// call this with its runtime path before any call to [`phonemize`].
///
/// Optional on desktop — if not called the library searches its compiled-in
/// system path.
pub fn set_data_path(path: &Path) {
    let _ = DATA_PATH.set(path.to_path_buf());
}

// ─── espeak feature: full FFI implementation ──────────────────────────────────

#[cfg(feature = "espeak")]
mod inner {
    use std::ffi::{CStr, CString};
    use std::os::raw::{c_char, c_int, c_void};
    use std::sync::Mutex;

    use anyhow::{anyhow, Result};
    use once_cell::sync::OnceCell;

    use super::DATA_PATH;

    // NEUTTS_ESPEAK_STAMP is set by build.rs to the content of
    // espeak-static/install/lib/espeak-ng-merged.stamp.  When the merged
    // library is rebuilt (stamp changes → build script re-runs → this value
    // changes), Cargo recompiles this module and re-bundles the fresh objects.
    #[allow(dead_code)]
    const _ESPEAK_STAMP: &str = env!("NEUTTS_ESPEAK_STAMP");

    extern "C" {
        fn espeak_ng_InitializePath(path: *const c_char);
        fn espeak_ng_Initialize(context: *mut c_void) -> c_int;
        fn espeak_ng_SetVoiceByName(name: *const c_char) -> c_int;
        fn espeak_TextToPhonemes(
            textptr: *mut *const c_void,
            textmode: c_int,
            phonememode: c_int,
        ) -> *const c_char;
    }

    const CHARS_UTF8:  c_int = 1;
    const PHONEMES_IPA: c_int = 0x02;

    /// Serialises all espeak-ng calls (the library uses global state).
    pub(super) static LOCK: Mutex<()> = Mutex::new(());

    /// Cached initialisation result, keyed by language code.
    /// We re-initialise when the language changes.
    pub(super) static INIT: OnceCell<std::result::Result<(), String>> = OnceCell::new();

    /// Initialise espeak-ng with the given language voice.
    pub(super) fn do_init(lang: &str) -> std::result::Result<(), String> {
        unsafe {
            // Resolve data path in priority order:
            //   1. Runtime call to set_data_path()           — highest priority
            //   2. NEUTTS_ESPEAK_DATA_DIR env var baked in   — set by build.rs
            //      at compile time when building from source
            //   3. NULL → espeak-ng uses its compiled-in     — system installs
            //      system path (e.g. /usr/lib/…/espeak-ng-data)
            let path_cstr: Option<CString> = DATA_PATH
                .get()
                .map(|p| p.to_string_lossy().into_owned())
                .or_else(|| option_env!("NEUTTS_ESPEAK_DATA_DIR").map(str::to_owned))
                .and_then(|s| CString::new(s.into_bytes()).ok());
            let path_ptr: *const c_char =
                path_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
            espeak_ng_InitializePath(path_ptr);
            let status = espeak_ng_Initialize(std::ptr::null_mut());
            if status != 0 {
                return Err(format!(
                    "espeak_ng_Initialize failed (status {:#010x})",
                    status
                ));
            }
            let voice = CString::new(lang).unwrap();
            let rc = espeak_ng_SetVoiceByName(voice.as_ptr());
            if rc != 0 {
                return Err(format!(
                    "espeak_ng_SetVoiceByName(\"{lang}\") failed (rc {rc})"
                ));
            }
        }
        Ok(())
    }

    pub(super) fn is_available(lang: &str) -> bool {
        let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());
        INIT.get_or_init(|| do_init(lang)).is_ok()
    }

    pub(super) fn run_phonemize(text: &str, lang: &str) -> Result<String> {
        let _g = LOCK.lock().unwrap_or_else(|p| p.into_inner());

        INIT.get_or_init(|| do_init(lang))
            .as_ref()
            .map_err(|e| anyhow!("espeak-ng: {}", e))?;

        let text_c = CString::new(text)
            .map_err(|_| anyhow!("phonemize: text contains a null byte"))?;

        let mut current: *const c_void = text_c.as_ptr() as *const c_void;
        let mut parts: Vec<String> = Vec::new();

        unsafe {
            while !current.is_null() {
                let ptr = espeak_TextToPhonemes(&mut current, CHARS_UTF8, PHONEMES_IPA);
                if ptr.is_null() { continue; }
                let chunk = CStr::from_ptr(ptr)
                    .to_str()
                    .map_err(|_| anyhow!("espeak-ng returned non-UTF-8 phonemes"))?
                    .trim()
                    .to_owned();
                if !chunk.is_empty() { parts.push(chunk); }
            }
        }
        Ok(parts.join(" "))
    }
}

// ─── Public API ────────────────────────────────────────────────────────────────

/// Returns `true` if espeak-ng is available and initialises successfully for
/// the given language code.
///
/// Always returns `false` when the `espeak` Cargo feature is disabled.
pub fn is_espeak_available(lang: &str) -> bool {
    #[cfg(feature = "espeak")]
    { inner::is_available(lang) }
    #[cfg(not(feature = "espeak"))]
    { let _ = lang; false }
}

/// Convert `text` to IPA phonemes using the espeak-ng voice for `lang`.
///
/// `lang` should be an espeak-ng language code such as `"en-us"`, `"de"`,
/// `"fr-fr"`, or `"es"`.
///
/// French output has dashes stripped to match the Python `FrenchPhonemizer`.
///
/// **Requires the `espeak` Cargo feature.**  Returns an error when the feature
/// is disabled — use `NeuTTS::infer_from_ipa()` to bypass phonemisation.
pub fn phonemize(text: &str, lang: &str) -> Result<String> {
    #[cfg(feature = "espeak")]
    {
        let raw = inner::run_phonemize(text, lang)?;
        // French: strip dashes (matches Python FrenchPhonemizer.clean).
        let cleaned = if lang.starts_with("fr") {
            raw.replace('-', "")
        } else {
            raw
        };
        // Normalise whitespace — mirrors Python `" ".join(phones.split())`
        let tokens: Vec<&str> = cleaned.split_whitespace().collect();
        Ok(tokens.join(" "))
    }
    #[cfg(not(feature = "espeak"))]
    {
        let _ = (text, lang);
        Err(anyhow!(
            "phonemize() requires the `espeak` Cargo feature.\n\
             Enable it: neutts = {{ features = [\"espeak\"] }}\n\
             Or use NeuTTS::infer_from_ipa() to bypass phonemisation."
        ))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "espeak"))]
mod tests {
    use super::*;

    #[test]
    fn test_available_en_us() {
        assert!(is_espeak_available("en-us"));
    }

    #[test]
    fn test_phonemize_hello() {
        let ipa = phonemize("Hello world", "en-us").expect("phonemize failed");
        assert!(!ipa.is_empty(), "expected non-empty IPA");
        println!("IPA (en-us): {ipa}");
    }

    #[test]
    fn test_phonemize_empty() {
        let ipa = phonemize("", "en-us").expect("phonemize failed");
        assert!(ipa.trim().is_empty(), "expected empty IPA for empty input");
    }

    #[test]
    fn test_phonemize_whitespace_normalised() {
        let ipa = phonemize("Hello world.", "en-us").expect("phonemize failed");
        assert!(!ipa.starts_with(' '), "should not start with space");
        assert!(!ipa.ends_with(' '),   "should not end with space");
    }

    #[test]
    fn test_french_no_dashes() {
        let ipa = phonemize("bonjour", "fr-fr").expect("phonemize failed");
        assert!(!ipa.contains('-'), "French output should have dashes stripped: {ipa}");
    }
}
