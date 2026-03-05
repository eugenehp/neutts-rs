// build.rs — NeuTTS build script
//
// When the `espeak` feature is enabled this script locates (or builds from
// source) a static libespeak-ng.a and wires it into the link.
//
// Resolution order
// ────────────────
// 1. espeak-static/lib/libespeak-ng.a  — cached from a previous source build
// 2. ESPEAK_LIB_DIR env var            — explicit directory
// 3. pkg-config                        — system / Homebrew install
// 4. Platform path walk                — well-known directories
// 5. Build from source                 — clone espeak-ng, build with cmake,
//                                        cache in espeak-static/

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // ── RoPE feature conflict check ───────────────────────────────────────────
    //
    // `fast` and `precise` select mutually exclusive sin/cos implementations.
    // Setting both at once is a programmer error.
    let feat_fast    = std::env::var("CARGO_FEATURE_FAST").is_ok();
    let feat_precise = std::env::var("CARGO_FEATURE_PRECISE").is_ok();
    if feat_fast && feat_precise {
        panic!(
            "\n\
             \nFeatures `fast` and `precise` are mutually exclusive.\n\
             \nPick one:\n\
             \n\
             \t  --features fast      # polynomial approx, ~1e-4 error (default)\n\
             \t  --features precise   # stdlib sin/cos, full accuracy\n"
        );
    }

    // ── espeak-ng linking ─────────────────────────────────────────────────────
    if std::env::var("CARGO_FEATURE_ESPEAK").is_ok() {
        link_espeak();
    }

    // ── NeuCodec safetensors weight files ────────────────────────────────────
    let decoder_path = Path::new("models/neucodec_decoder.safetensors");
    if decoder_path.exists() {
        println!("cargo::rustc-cfg=neucodec_decoder_available");
        println!(
            "cargo::warning=NeuCodec decoder weights found: {}",
            decoder_path.display()
        );
    } else {
        println!(
            "cargo::warning=NeuCodec decoder weights not found at {}. \
             Run `python scripts/convert_weights.py` to generate them.",
            decoder_path.display()
        );
    }

    let encoder_path = Path::new("models/neucodec_encoder.safetensors");
    if encoder_path.exists() {
        println!("cargo::rustc-cfg=neucodec_encoder_available");
    }

    println!("cargo::rustc-check-cfg=cfg(neucodec_decoder_available)");
    println!("cargo::rustc-check-cfg=cfg(neucodec_encoder_available)");

    // ── NEUTTS_ESPEAK_STAMP ───────────────────────────────────────────────────
    let stamp_path = Path::new("espeak-static/install/lib/espeak-ng-merged.stamp");
    let stamp = if stamp_path.exists() {
        std::fs::read_to_string(stamp_path).unwrap_or_default()
    } else {
        "system".to_string()
    };
    println!("cargo::rustc-env=NEUTTS_ESPEAK_STAMP={stamp}");
}

// ─────────────────────────────────────────────────────────────────────────────
// espeak-ng linking
// ─────────────────────────────────────────────────────────────────────────────

fn link_espeak() {
    println!("cargo::rerun-if-env-changed=ESPEAK_LIB_DIR");
    println!("cargo::rerun-if-env-changed=PKG_CONFIG_PATH");
    // Rebuild if cached archive appears / disappears
    println!("cargo::rerun-if-changed=espeak-static/lib/libespeak-ng.a");

    let target_os   = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    // 1. Cached source build (merged or individual archives)
    let cached_merged = Path::new("espeak-static/lib/libespeak-ng-merged.a");
    let cached_plain  = Path::new("espeak-static/lib/libespeak-ng.a");
    if cached_merged.exists() || cached_plain.exists() {
        emit_static("espeak-static/lib", &target_os);
        emit_data_dir("espeak-static/install/share/espeak-ng-data")
            .or_else(|| emit_data_dir("espeak-static/install/lib/espeak-ng-data"))
            .or_else(|| emit_data_dir("espeak-static/src/espeak-ng-data"));
        return;
    }

    // 2. ESPEAK_LIB_DIR
    if let Ok(dir) = std::env::var("ESPEAK_LIB_DIR") {
        if Path::new(&dir).join("libespeak-ng.a").exists() {
            emit_static(&dir, &target_os);
            return;
        }
        // dir is set but archive is absent — fall through to build from source
        eprintln!("neutts build.rs: ESPEAK_LIB_DIR={dir:?} set but libespeak-ng.a not found — building from source");
    }

    // Mobile without a pre-built static lib: fail early
    if matches!(&*target_os, "ios" | "android") {
        panic!(
            "\n\nSet ESPEAK_LIB_DIR to a directory containing libespeak-ng.a \
             built for {target_os}/{target_arch}.\n"
        );
    }

    // 3. pkg-config (static preferred)
    if let Some(dir) = pkg_config_libdir(&target_os) {
        if Path::new(&dir).join("libespeak-ng.a").exists() {
            emit_static(&dir, &target_os);
            return;
        }
        if target_os == "linux" && has_dylib(&dir) {
            println!("cargo::rustc-link-search=native={dir}");
            println!("cargo::rustc-link-lib=espeak-ng");
            return;
        }
    }

    // 4. Platform path walk
    for dir in candidate_dirs(&target_os, &target_arch) {
        let s = dir.to_string_lossy();
        if dir.join("libespeak-ng.a").exists() {
            emit_static(&s, &target_os);
            return;
        }
        if target_os == "linux" && has_dylib(&s) {
            println!("cargo::rustc-link-search=native={s}");
            println!("cargo::rustc-link-lib=espeak-ng");
            return;
        }
    }

    // 5. Build from source
    eprintln!("neutts build.rs: libespeak-ng not found — building from source (runs once, ~1–2 min)");
    build_from_source(&target_os);

    // After a successful source build the archive is at espeak-static/lib/
    let lib_dir = "espeak-static/lib";
    if !Path::new(lib_dir).join("libespeak-ng.a").exists() {
        panic!(
            "\n\nneutts build.rs: source build finished but libespeak-ng.a not found in {lib_dir}.\n\
             Check the cmake output above for errors.\n"
        );
    }
    emit_static(lib_dir, &target_os);
    emit_data_dir("espeak-static/install/share/espeak-ng-data")
        .or_else(|| emit_data_dir("espeak-static/install/lib/espeak-ng-data"));
}

/// Emit cargo directives for a static link from `dir`.
///
/// Links `libespeak-ng-merged.a` if present (a single archive containing
/// espeak-ng + speechPlayer + ucd), otherwise links the three libraries
/// separately in dependency order.
fn emit_static(dir: &str, target_os: &str) {
    println!("cargo::rustc-link-search=native={dir}");
    if Path::new(dir).join("libespeak-ng-merged.a").exists() {
        println!("cargo::rustc-link-lib=static=espeak-ng-merged");
    } else {
        // espeak-ng references symbols in speechPlayer and ucd, so those must
        // follow it in link order so the linker can resolve them.
        println!("cargo::rustc-link-lib=static=espeak-ng");
        if Path::new(dir).join("libspeechPlayer.a").exists() {
            println!("cargo::rustc-link-lib=static=speechPlayer");
        }
        if Path::new(dir).join("libucd.a").exists() {
            println!("cargo::rustc-link-lib=static=ucd");
        }
    }
    // espeak-ng is C++; static link requires the C++ runtime
    if target_os == "macos" {
        println!("cargo::rustc-link-lib=dylib=c++");
    } else {
        println!("cargo::rustc-link-lib=dylib=stdc++");
    }
}

/// Emit NEUTTS_ESPEAK_DATA_DIR if `dir` exists.  Returns Some(()) on success.
fn emit_data_dir(dir: &str) -> Option<()> {
    let p = Path::new(dir);
    if p.exists() {
        // Canonicalise so phonemize.rs gets an absolute path regardless of cwd
        if let Ok(abs) = std::fs::canonicalize(p) {
            println!("cargo::rustc-env=NEUTTS_ESPEAK_DATA_DIR={}", abs.display());
            return Some(());
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Build from source
// ─────────────────────────────────────────────────────────────────────────────

const ESPEAK_VERSION: &str = "1.52.0";
const ESPEAK_REPO:    &str = "https://github.com/espeak-ng/espeak-ng";

fn build_from_source(target_os: &str) {
    let src_dir  = PathBuf::from("espeak-static/src");
    let bld_dir  = PathBuf::from("espeak-static/build");
    let inst_dir = std::fs::canonicalize(".")
        .unwrap()
        .join("espeak-static/install");

    std::fs::create_dir_all(&src_dir).expect("mkdir espeak-static/src");
    std::fs::create_dir_all(&bld_dir).expect("mkdir espeak-static/build");

    // ── Clone if needed ───────────────────────────────────────────────────────
    if !src_dir.join("CMakeLists.txt").exists() {
        eprintln!("neutts build.rs: cloning espeak-ng {ESPEAK_VERSION}…");
        let ok = Command::new("git")
            .args([
                "clone", "--depth=1",
                "--branch", &format!("{ESPEAK_VERSION}"),
                ESPEAK_REPO,
                src_dir.to_str().unwrap(),
            ])
            .status()
            .unwrap_or_else(|e| panic!("git not found ({e}) — install git and retry"));
        if !ok.success() {
            panic!(
                "\n\ngit clone of espeak-ng failed.\n\
                 Check your internet connection, then retry `cargo build`.\n\
                 Or install espeak-ng manually and set ESPEAK_LIB_DIR.\n"
            );
        }
    }

    // ── cmake configure ───────────────────────────────────────────────────────
    eprintln!("neutts build.rs: cmake configure…");
    let nproc = num_cpus();

    let mut cfg = Command::new("cmake");
    cfg.current_dir(&bld_dir)
        .arg(src_dir.canonicalize().unwrap())
        .arg(format!("-DCMAKE_INSTALL_PREFIX={}", inst_dir.display()))
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg("-DUSE_MBROLA=OFF")
        .arg("-DUSE_LIBSONIC=OFF")
        .arg("-DUSE_LIBPCAUDIO=OFF")
        .arg("-DUSE_ASYNC=OFF");

    // On macOS embed the data path so the binary works without NEUTTS_ESPEAK_DATA_DIR
    if target_os == "macos" {
        cfg.arg("-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0");
    }

    let ok = cfg.status().unwrap_or_else(|e| {
        panic!(
            "\n\ncmake not found ({e}).\n\
             Install cmake:\n\
             \n\
             \t  macOS:  brew install cmake\n\
             \t  Linux:  apt install cmake  /  dnf install cmake\n"
        )
    });
    if !ok.success() {
        panic!("\n\ncmake configure failed — see output above.\n");
    }

    // ── cmake build + install ─────────────────────────────────────────────────
    eprintln!("neutts build.rs: cmake build (using {nproc} threads)…");
    let ok = Command::new("cmake")
        .current_dir(&bld_dir)
        .args([
            "--build", ".",
            "--target", "install",
            "--config", "Release",
            "--parallel", &nproc.to_string(),
        ])
        .status()
        .expect("cmake --build failed");
    if !ok.success() {
        panic!("\n\ncmake build failed — see output above.\n");
    }

    // ── Gather archives and create a merged static library ───────────────────
    //
    // cmake produces three archives:
    //   libespeak-ng.a     — main library  (references ucd + speechPlayer)
    //   libspeechPlayer.a  — Klatt speech player
    //   libucd.a           — Unicode character database
    //
    // We merge all three into libespeak-ng-merged.a so callers need only one
    // -l flag, and link order is no longer a concern.
    let lib_dir = PathBuf::from("espeak-static/lib");
    std::fs::create_dir_all(&lib_dir).expect("mkdir espeak-static/lib");

    // Locate every .a produced by the build (in build tree or install tree)
    let search_roots = [
        bld_dir.clone(),
        inst_dir.join("lib"),
        inst_dir.join("lib64"),
    ];
    let wanted = ["libespeak-ng.a", "libspeechPlayer.a", "libucd.a"];
    let mut found: Vec<PathBuf> = Vec::new();
    for root in &search_roots {
        for name in &wanted {
            let p = find_file(root, name);
            if let Some(p) = p {
                let dest = lib_dir.join(name);
                std::fs::copy(&p, &dest).unwrap_or_else(|e| panic!("copy {name}: {e}"));
                if !found.iter().any(|f: &PathBuf| f.file_name() == dest.file_name()) {
                    found.push(dest);
                }
            }
        }
    }

    if found.is_empty() {
        panic!("\n\nneutts build.rs: no .a files found after espeak-ng cmake build.\n");
    }

    // Merge into libespeak-ng-merged.a
    merge_archives(&lib_dir, &found, &target_os);

    eprintln!("neutts build.rs: espeak-ng built and cached in espeak-static/");
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn num_cpus() -> usize {
    // Prefer MAKEFLAGS -jN, fall back to nproc / sysctl, default to 4.
    if let Ok(mf) = std::env::var("MAKEFLAGS") {
        for part in mf.split_whitespace() {
            if let Some(n) = part.strip_prefix("-j") {
                if let Ok(n) = n.parse::<usize>() {
                    return n;
                }
            }
        }
    }
    // nproc (Linux) / sysctl (macOS)
    Command::new("nproc")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok())
        .or_else(|| {
            Command::new("sysctl")
                .args(["-n", "hw.logicalcpu"])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .and_then(|s| s.trim().parse().ok())
        })
        .unwrap_or(4)
}

fn has_dylib(dir: &str) -> bool {
    let d = Path::new(dir);
    if d.join("libespeak-ng.dylib").exists() || d.join("libespeak-ng.so").exists() {
        return true;
    }
    std::fs::read_dir(d).ok().map_or(false, |entries| {
        entries.flatten().any(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with("libespeak-ng.so.")
        })
    })
}

fn pkg_config_libdir(target_os: &str) -> Option<String> {
    let mut extra: Vec<String> = Vec::new();
    if target_os == "macos" {
        for prefix in ["/opt/homebrew", "/usr/local"] {
            for sub in ["lib/pkgconfig", "share/pkgconfig"] {
                let p = format!("{prefix}/{sub}");
                if Path::new(&p).is_dir() {
                    extra.push(p);
                }
            }
        }
        if let Some(keg) = brew_prefix("espeak-ng") {
            let p = format!("{keg}/lib/pkgconfig");
            if Path::new(&p).is_dir() {
                extra.insert(0, p);
            }
        }
    }
    if let Ok(existing) = std::env::var("PKG_CONFIG_PATH") {
        extra.push(existing);
    }
    let pkg_path = extra.join(":");

    let out = Command::new("pkg-config")
        .args(["--variable=libdir", "espeak-ng"])
        .env("PKG_CONFIG_PATH", &pkg_path)
        .output()
        .ok()?;
    if out.status.success() {
        Some(String::from_utf8(out.stdout).ok()?.trim().to_owned())
    } else {
        None
    }
}

fn brew_prefix(formula: &str) -> Option<String> {
    let out = Command::new("brew")
        .args(["--prefix", formula])
        .output()
        .ok()?;
    if out.status.success() {
        Some(String::from_utf8(out.stdout).ok()?.trim().to_owned())
    } else {
        None
    }
}

/// Recursively search `root` for a file named `name`.
fn find_file(root: &Path, name: &str) -> Option<PathBuf> {
    if !root.is_dir() { return None; }
    let direct = root.join(name);
    if direct.exists() { return Some(direct); }
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let p = entry.path();
        if p.is_dir() {
            if let Some(found) = find_file(&p, name) {
                return Some(found);
            }
        }
    }
    None
}

/// Merge multiple `.a` archives into `lib_dir/libespeak-ng-merged.a`.
///
/// * macOS — uses `libtool -static` (ships with Xcode CLT).
/// * Linux  — uses an `ar` MRI script.
fn merge_archives(lib_dir: &Path, archives: &[PathBuf], target_os: &str) {
    let merged = lib_dir.join("libespeak-ng-merged.a");
    let archive_args: Vec<&str> = archives.iter()
        .map(|p| p.to_str().unwrap())
        .collect();

    eprintln!("neutts build.rs: merging {} archives → {}", archives.len(), merged.display());

    if target_os == "macos" {
        // libtool is always present on macOS with Xcode CLT
        let mut cmd = Command::new("libtool");
        cmd.arg("-static").arg("-o").arg(&merged);
        cmd.args(&archive_args);
        let ok = cmd.status().expect("libtool not found");
        if !ok.success() { panic!("\n\nlibtool merge failed.\n"); }
    } else {
        // Use ar MRI script
        let mut script = String::from("CREATE libespeak-ng-merged.a\n");
        for a in archives {
            script.push_str(&format!("ADDLIB {}\n", a.display()));
        }
        script.push_str("SAVE\nEND\n");

        let script_path = lib_dir.join("merge.mri");
        std::fs::write(&script_path, &script).expect("write ar MRI script");

        let ok = Command::new("ar")
            .arg("-M")
            .stdin(std::process::Stdio::piped())
            .spawn()
            .and_then(|mut child| {
                use std::io::Write;
                child.stdin.as_mut().unwrap().write_all(script.as_bytes())?;
                child.wait()
            })
            .expect("ar not found");
        let _ = std::fs::remove_file(&script_path);
        if !ok.success() { panic!("\n\nar merge failed.\n"); }
    }
}

fn candidate_dirs(target_os: &str, target_arch: &str) -> Vec<PathBuf> {
    let mut dirs: Vec<String> = Vec::new();
    if target_os == "macos" {
        if let Some(keg) = brew_prefix("espeak-ng") {
            dirs.push(format!("{keg}/lib"));
        }
        for prefix in ["/opt/homebrew", "/usr/local"] {
            dirs.push(format!("{prefix}/opt/espeak-ng/lib"));
            dirs.push(format!("{prefix}/lib"));
        }
    } else {
        let multiarch = match target_arch {
            "x86_64"      => "x86_64-linux-gnu",
            "aarch64"     => "aarch64-linux-gnu",
            "arm"         => "arm-linux-gnueabihf",
            "riscv64"     => "riscv64-linux-gnu",
            _             => "",
        };
        if !multiarch.is_empty() {
            dirs.push(format!("/usr/lib/{multiarch}"));
        }
        dirs.extend(["/usr/lib64", "/usr/lib", "/usr/local/lib"].map(String::from));
    }
    dirs.into_iter().map(PathBuf::from).filter(|p| p.is_dir()).collect()
}
