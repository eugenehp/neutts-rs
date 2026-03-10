// build.rs — NeuTTS build script
//
// When the `espeak` feature is enabled this script locates (or builds from
// source) a static espeak-ng library and wires it into the link.
//
// ── Resolution order ─────────────────────────────────────────────────────────
//
//  1. ESPEAK_LIB_DIR env var       — explicit pre-built directory; both
//                                    libespeak-ng.a (Unix/MinGW) and
//                                    espeak-ng.lib (MSVC) are accepted.
//  2. espeak-static/lib/ (manifest-relative)
//                                  — result of `bash scripts/build-espeak.sh`
//                                    or `.\scripts\build-espeak-windows.ps1`
//  3. pkg-config                   — system / Homebrew install (non-Windows)
//  4. Platform path walk           — well-known directories  (non-Windows)
//  5. Build from source            — clone espeak-ng, cmake, cache in OUT_DIR
//
// ── Platform / toolchain matrix ──────────────────────────────────────────────
//
//  target_os  target_env  lib filename        merger          C++ runtime
//  ─────────  ──────────  ──────────────────  ─────────────── ─────────────
//  macos      (any)       libespeak-ng.a      libtool -static c++ (dylib)
//  linux      (any)       libespeak-ng.a      ar MRI          stdc++ (dylib)
//  windows    gnu         libespeak-ng.a      ar MRI          stdc++ (dylib)
//  windows    msvc        espeak-ng.lib       lib.exe         (auto by MSVC)
//
// ── Cross-compilation ─────────────────────────────────────────────────────────
//
//  windows-gnu from Linux/macOS
//      Install mingw-w64:
//          Linux:  sudo apt install gcc-mingw-w64-x86-64
//          macOS:  brew install mingw-w64
//      Set ESPEAK_CROSS_PREFIX (default: x86_64-w64-mingw32-).
//      The build script uses the bundled cmake/mingw-toolchain.cmake.
//
//  windows-msvc from Linux/macOS
//      Requires a pre-built library; set ESPEAK_LIB_DIR to its directory.
//      (Use xwin + clang-cl or a Windows runner to produce the library.)
//
// ── Windows path-length note ─────────────────────────────────────────────────
//
//  std::fs::canonicalize() on Windows calls GetFinalPathNameByHandleW which
//  always returns the \\?\ extended-length prefix form, even for short paths.
//  MSVC cl.exe rejects \\?\ paths with "Cannot open source file".
//
//  Fix: never call canonicalize().  Use OUT_DIR directly — it is already an
//  absolute, clean path set by Cargo.  ESPEAK_BUILD_DIR overrides the build
//  root for projects with unusually long OUT_DIR paths.

use std::env;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn main() {
    // ── RoPE feature conflict check ───────────────────────────────────────────
    if env::var("CARGO_FEATURE_FAST").is_ok() && env::var("CARGO_FEATURE_PRECISE").is_ok() {
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
    if env::var("CARGO_FEATURE_ESPEAK").is_ok() {
        link_espeak();
    }

    // ── NeuCodec safetensors weight files ─────────────────────────────────────
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
    println!("cargo::rerun-if-env-changed=ESPEAK_BUILD_DIR");
    println!("cargo::rerun-if-env-changed=ESPEAK_CROSS_PREFIX");
    println!("cargo::rerun-if-env-changed=PKG_CONFIG_PATH");

    // Track the manifest-local cache so Cargo re-runs the script if the
    // pre-built file appears or is deleted.
    println!("cargo::rerun-if-changed=espeak-static/lib/libespeak-ng.a");
    println!("cargo::rerun-if-changed=espeak-static/lib/espeak-ng.lib");

    let target_os  = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    // ── 1. ESPEAK_LIB_DIR — explicit directory ────────────────────────────────
    if let Ok(dir) = env::var("ESPEAK_LIB_DIR") {
        let p = Path::new(&dir);
        if has_static_lib(p, &target_env) {
            emit_static(&dir, &target_os, &target_env);
            return;
        }
        // Dir is set but library is absent — fall through to build from source.
        eprintln!(
            "neutts build.rs: ESPEAK_LIB_DIR={dir:?} is set but \
             {} not found there — falling through to build from source",
            static_lib_filename(&target_env)
        );
    }

    // Mobile / iOS / Android without explicit lib dir: fail early.
    if matches!(&*target_os, "ios" | "android") {
        panic!(
            "\n\nSet ESPEAK_LIB_DIR to a directory containing {} \
             built for {target_os}/{target_arch}.\n",
            static_lib_filename(&target_env)
        );
    }

    // ── 2. Manifest-local espeak-static/lib/ (from build-espeak.sh / .ps1) ───
    let manifest_lib = Path::new("espeak-static/lib");
    if has_static_lib(manifest_lib, &target_env) {
        emit_static(&manifest_lib.to_string_lossy(), &target_os, &target_env);
        emit_data_dir("espeak-static/install/share/espeak-ng-data")
            .or_else(|| emit_data_dir("espeak-static/install/lib/espeak-ng-data"))
            .or_else(|| emit_data_dir("espeak-static/src/espeak-ng-data"));
        return;
    }

    // ── 3. pkg-config (non-Windows) ───────────────────────────────────────────
    if target_os != "windows" {
        if let Some(dir) = pkg_config_libdir(&target_os) {
            let p = Path::new(&dir);
            if p.join("libespeak-ng.a").exists() {
                emit_static(&dir, &target_os, &target_env);
                return;
            }
            if target_os == "linux" && has_dylib(p) {
                println!("cargo::rustc-link-search=native={dir}");
                println!("cargo::rustc-link-lib=espeak-ng");
                return;
            }
        }

        // ── 4. Platform path walk ─────────────────────────────────────────────
        for dir in candidate_dirs(&target_os, &target_arch) {
            let s = dir.to_string_lossy();
            if dir.join("libespeak-ng.a").exists() {
                emit_static(&s, &target_os, &target_env);
                return;
            }
            if target_os == "linux" && has_dylib(&dir) {
                println!("cargo::rustc-link-search=native={s}");
                println!("cargo::rustc-link-lib=espeak-ng");
                return;
            }
        }
    }

    // ── 5. Build from source ──────────────────────────────────────────────────
    eprintln!(
        "neutts build.rs: {} not found — building from source (runs once, ~1–2 min)",
        static_lib_filename(&target_env)
    );
    let lib_dir = build_from_source(&target_os, &target_env);

    if !has_static_lib(&lib_dir, &target_env) {
        panic!(
            "\n\nneutts build.rs: source build finished but {} not found in {}.\n\
             Check the cmake output above for errors.\n",
            static_lib_filename(&target_env),
            lib_dir.display()
        );
    }

    let lib_dir_str = lib_dir.to_string_lossy();
    emit_static(&lib_dir_str, &target_os, &target_env);

    // Data dir lives next to lib/ in our OUT_DIR layout.
    let inst = lib_dir.parent().unwrap_or(&lib_dir);
    emit_data_dir(&inst.join("share/espeak-ng-data").to_string_lossy())
        .or_else(|| emit_data_dir(&inst.join("lib/espeak-ng-data").to_string_lossy()));
}

// ─────────────────────────────────────────────────────────────────────────────
// Emit cargo link directives
// ─────────────────────────────────────────────────────────────────────────────

/// Emit `rustc-link-search` and `rustc-link-lib` for a static espeak-ng
/// installation under `dir`.
///
/// Prefers the merged archive (one `-l` flag, no link-order sensitivity) and
/// falls back to linking the three individual archives when the merged one is
/// absent.
fn emit_static(dir: &str, target_os: &str, target_env: &str) {
    println!("cargo::rustc-link-search=native={dir}");

    let d = Path::new(dir);
    if d.join(merged_lib_filename(target_env)).exists() {
        // Single merged archive — espeak-ng + speechPlayer + ucd all in one.
        println!("cargo::rustc-link-lib=static=espeak-ng-merged");
    } else {
        // Individual archives in dependency order:
        //   espeak-ng → speechPlayer → ucd
        println!("cargo::rustc-link-lib=static=espeak-ng");
        // Companion archives (present when built by cmake without merging).
        for (file, name) in [
            (companion_filename("speechPlayer", target_env), "speechPlayer"),
            (companion_filename("ucd",           target_env), "ucd"),
        ] {
            if d.join(&file).exists() {
                println!("cargo::rustc-link-lib=static={name}");
            }
        }
    }

    // espeak-ng is a C++ library; static linking requires the C++ runtime.
    //
    // macOS          : libc++ (LLVM, ships with Xcode CLT)
    // Linux          : libstdc++ (GCC)
    // Windows / GNU  : libstdc++ (MinGW)
    // Windows / MSVC : nothing — the MSVC CRT is auto-linked by link.exe
    //                  when it sees C++ objects; no explicit flag needed.
    match (target_os, target_env) {
        ("macos", _)         => println!("cargo::rustc-link-lib=dylib=c++"),
        ("windows", "msvc")  => { /* MSVC CRT auto-linked */ }
        _                    => println!("cargo::rustc-link-lib=dylib=stdc++"),
    }
}

/// Emit `NEUTTS_ESPEAK_DATA_DIR` compile-time env if `dir` exists.
fn emit_data_dir(dir: &str) -> Option<()> {
    let p = Path::new(dir);
    if !p.exists() {
        return None;
    }
    // Avoid canonicalize() on Windows (adds \\?\ prefix that breaks things).
    // The path from OUT_DIR is already absolute; for relative paths make
    // them absolute by joining with the manifest dir.
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else {
        let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
        manifest.join(p)
    };
    println!("cargo::rustc-env=NEUTTS_ESPEAK_DATA_DIR={}", abs.display());
    Some(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Filename helpers — vary by target ABI
// ─────────────────────────────────────────────────────────────────────────────

/// Filename of the main static archive on this target.
///   windows-msvc → "espeak-ng.lib"
///   everything else → "libespeak-ng.a"
fn static_lib_filename(target_env: &str) -> &'static str {
    if target_env == "msvc" { "espeak-ng.lib" } else { "libespeak-ng.a" }
}

/// Filename of the merged (fat) archive.
fn merged_lib_filename(target_env: &str) -> &'static str {
    if target_env == "msvc" { "espeak-ng-merged.lib" } else { "libespeak-ng-merged.a" }
}

/// Filename for a named companion library (speechPlayer, ucd).
fn companion_filename(name: &str, target_env: &str) -> String {
    if target_env == "msvc" {
        format!("{name}.lib")
    } else {
        format!("lib{name}.a")
    }
}

/// Returns `true` if `dir` contains a static espeak-ng library for
/// the given target ABI (checks plain + merged archive names).
fn has_static_lib(dir: &Path, target_env: &str) -> bool {
    dir.join(static_lib_filename(target_env)).exists()
        || dir.join(merged_lib_filename(target_env)).exists()
}

// ─────────────────────────────────────────────────────────────────────────────
// Build from source
// ─────────────────────────────────────────────────────────────────────────────

const ESPEAK_VERSION: &str = "1.52.0";
const ESPEAK_REPO:    &str = "https://github.com/espeak-ng/espeak-ng";

/// Build espeak-ng from source and return the path to the directory containing
/// the produced static archive(s).
///
/// All intermediate files are placed under `OUT_DIR` (or `ESPEAK_BUILD_DIR`)
/// to guarantee short, absolute, `\\?\`-free paths on Windows.
fn build_from_source(target_os: &str, target_env: &str) -> PathBuf {
    // ── Build root ────────────────────────────────────────────────────────────
    //
    // IMPORTANT: do NOT call std::fs::canonicalize() anywhere in this function.
    // On Windows, canonicalize() returns \\?\-prefixed paths which MSVC cl.exe
    // rejects.  OUT_DIR and ESPEAK_BUILD_DIR are already absolute.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_root = env::var("ESPEAK_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| out_dir.join("espeak-build"));

    let src_dir  = build_root.join("src");
    let bld_dir  = build_root.join("bld");
    let inst_dir = build_root.join("inst");
    let lib_dir  = build_root.join("lib");

    std::fs::create_dir_all(&src_dir).expect("mkdir espeak-build/src");
    std::fs::create_dir_all(&bld_dir).expect("mkdir espeak-build/bld");
    std::fs::create_dir_all(&inst_dir).expect("mkdir espeak-build/inst");
    std::fs::create_dir_all(&lib_dir).expect("mkdir espeak-build/lib");

    // ── Clone (or reuse cached) ───────────────────────────────────────────────
    if !src_dir.join("CMakeLists.txt").exists() {
        eprintln!("neutts build.rs: cloning espeak-ng {ESPEAK_VERSION}…");
        let ok = Command::new("git")
            .args([
                "clone", "--depth=1",
                "--branch", ESPEAK_VERSION,
                ESPEAK_REPO,
            ])
            .arg(&src_dir)
            .status()
            .unwrap_or_else(|e| panic!("git not found ({e}) — install git and retry"));
        if !ok.success() {
            panic!(
                "\n\ngit clone of espeak-ng failed.\n\
                 Check your internet connection, then retry `cargo build`.\n\
                 Or run scripts/build-espeak.sh (Unix) / scripts\\build-espeak-windows.ps1 (Windows)\n\
                 and set ESPEAK_LIB_DIR to the resulting lib directory.\n"
            );
        }
    }

    // ── cmake configure ───────────────────────────────────────────────────────
    eprintln!("neutts build.rs: cmake configure…");
    let nproc = num_cpus();

    let mut cfg = Command::new("cmake");
    cfg.current_dir(&bld_dir);

    // Source dir: pass as-is (already absolute, no canonicalize).
    cfg.arg(&src_dir);

    cfg.arg(format!("-DCMAKE_INSTALL_PREFIX={}", inst_dir.display()));
    cfg.arg("-DCMAKE_BUILD_TYPE=Release");
    cfg.arg("-DBUILD_SHARED_LIBS=OFF");
    cfg.arg("-DUSE_MBROLA=OFF");
    cfg.arg("-DUSE_LIBSONIC=OFF");
    cfg.arg("-DUSE_LIBPCAUDIO=OFF");
    cfg.arg("-DUSE_ASYNC=OFF");

    match target_os {
        "macos" => {
            cfg.arg("-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0");
        }
        "windows" if target_env == "gnu" => {
            // Cross-compiling to windows-gnu from a Unix host, or native MinGW.
            // Prefer the bundled toolchain file; fall back to letting cmake auto-detect.
            let toolchain = Path::new("scripts/cmake/mingw-toolchain.cmake");
            if toolchain.exists() {
                cfg.arg(format!("-DCMAKE_TOOLCHAIN_FILE={}", toolchain.display()));
            }
            // Let the user override the cross-compiler prefix.
            let prefix = env::var("ESPEAK_CROSS_PREFIX")
                .unwrap_or_else(|_| "x86_64-w64-mingw32-".to_string());
            cfg.arg(format!("-DCMAKE_C_COMPILER={prefix}gcc"));
            cfg.arg(format!("-DCMAKE_CXX_COMPILER={prefix}g++"));
            cfg.arg(format!("-DCMAKE_RC_COMPILER={prefix}windres"));
            // Ninja avoids MSBuild's path-length quoting issues and works on all hosts.
            if cmake_generator_available("Ninja") {
                cfg.arg("-GNinja");
            }
        }
        "windows" if target_env == "msvc" => {
            // Native Windows MSVC build.
            // Ninja avoids the \\?\ path issues caused by MSBuild's .vcxproj generator.
            if cmake_generator_available("Ninja") {
                cfg.arg("-GNinja");
            }
        }
        _ => {}
    }

    let ok = cfg.status().unwrap_or_else(|e| {
        panic!(
            "\n\ncmake not found ({e}).\n\
             Install cmake:\n\
             \n\
             \t  macOS  :  brew install cmake\n\
             \t  Linux  :  apt install cmake  /  dnf install cmake\n\
             \t  Windows:  winget install Kitware.CMake\n"
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
        .expect("cmake --build failed to launch");
    if !ok.success() {
        let hint = if target_os == "windows" && target_env == "msvc" {
            "\nHint: if MSVC reported path errors, run the build from a shorter directory\n\
             path or set ESPEAK_BUILD_DIR=C:\\es to use a short build root.\n\
             Alternatively, pre-build with scripts\\build-espeak-windows.ps1\n\
             and set ESPEAK_LIB_DIR to the resulting lib directory.\n"
        } else {
            ""
        };
        panic!("\n\ncmake build failed — see output above.{hint}\n");
    }

    // ── Collect produced archives ─────────────────────────────────────────────
    //
    // cmake builds several archives depending on the espeak-ng version:
    //   libespeak-ng.a / espeak-ng.lib    — main library
    //   libspeechPlayer.a / speechPlayer.lib — Klatt speech player
    //   libucd.a / ucd.lib                — Unicode character database
    //
    // We copy them to lib_dir and merge into one fat archive so callers need
    // only a single -l flag with no link-order concerns.
    let search_roots = [bld_dir.clone(), inst_dir.join("lib"), inst_dir.join("lib64")];
    let wanted: &[&str] = if target_env == "msvc" {
        &["espeak-ng.lib", "speechPlayer.lib", "ucd.lib"]
    } else {
        &["libespeak-ng.a", "libspeechPlayer.a", "libucd.a"]
    };

    let mut found: Vec<PathBuf> = Vec::new();
    for root in &search_roots {
        for name in wanted {
            if let Some(p) = find_file(root, name) {
                let dest = lib_dir.join(name);
                std::fs::copy(&p, &dest)
                    .unwrap_or_else(|e| panic!("copy {name}: {e}"));
                if !found.iter().any(|f: &PathBuf| f.file_name() == dest.file_name()) {
                    found.push(dest);
                }
            }
        }
    }

    if found.is_empty() {
        panic!(
            "\n\nneutts build.rs: no static archives found after espeak-ng cmake build.\n\
             Build tree: {}\n",
            bld_dir.display()
        );
    }

    // ── Merge into one fat archive ────────────────────────────────────────────
    merge_archives(&lib_dir, &found, target_os, target_env);

    // ── Copy espeak-ng-data ───────────────────────────────────────────────────
    let data_dest = inst_dir.join("share/espeak-ng-data");
    if !data_dest.exists() {
        for candidate in [
            src_dir.join("espeak-ng-data"),
            bld_dir.join("espeak-ng-data"),
            inst_dir.join("lib/espeak-ng-data"),
        ] {
            if candidate.is_dir() {
                copy_dir(&candidate, &data_dest)
                    .unwrap_or_else(|e| eprintln!("neutts build.rs: copy espeak-ng-data: {e}"));
                break;
            }
        }
    }

    eprintln!("neutts build.rs: espeak-ng built → {}", lib_dir.display());
    lib_dir
}

// ─────────────────────────────────────────────────────────────────────────────
// Archive merging
// ─────────────────────────────────────────────────────────────────────────────

/// Merge `archives` into a single fat archive inside `lib_dir`.
///
/// Output filename:
///   MSVC   → lib_dir/espeak-ng-merged.lib
///   Others → lib_dir/libespeak-ng-merged.a
fn merge_archives(lib_dir: &Path, archives: &[PathBuf], target_os: &str, target_env: &str) {
    let merged = lib_dir.join(merged_lib_filename(target_env));
    eprintln!(
        "neutts build.rs: merging {} archive(s) → {}",
        archives.len(),
        merged.display()
    );

    if target_os == "macos" {
        // libtool -static ships with Xcode CLT.
        let mut cmd = Command::new("libtool");
        cmd.arg("-static").arg("-o").arg(&merged);
        cmd.args(archives);
        let ok = cmd.status().expect("libtool not found (install Xcode CLT)");
        if !ok.success() {
            panic!("\n\nlibtool merge failed.\n");
        }

    } else if target_os == "windows" && target_env == "msvc" {
        // MSVC: use lib.exe which ships with every Visual Studio / Build Tools install.
        // Try `lib` first (MSVC), then `llvm-lib` (if LLVM is on PATH).
        let lib_exe = find_cmd(&["lib", "llvm-lib"])
            .unwrap_or_else(|| {
                panic!(
                    "\n\nNeither `lib` (MSVC) nor `llvm-lib` (LLVM) found on PATH.\n\
                     Install Visual Studio Build Tools and run from a Developer Command Prompt,\n\
                     or install LLVM (winget install LLVM.LLVM).\n"
                )
            });
        let out_arg = format!("/OUT:{}", merged.display());
        let mut cmd = Command::new(&lib_exe);
        cmd.arg(&out_arg);
        cmd.args(archives);
        let ok = cmd.status().unwrap_or_else(|e| panic!("{lib_exe} failed to launch: {e}"));
        if !ok.success() {
            panic!("\n\n{lib_exe} merge failed.\n");
        }

    } else {
        // Linux / windows-gnu / other Unix: ar MRI script.
        //
        // For cross-compilation, ESPEAK_CROSS_PREFIX selects the cross-ar
        // (e.g. "x86_64-w64-mingw32-" → "x86_64-w64-mingw32-ar").
        // Fall back to plain "ar" for native builds.
        let prefix = env::var("ESPEAK_CROSS_PREFIX").unwrap_or_default();
        let ar = format!("{prefix}ar");

        let mut mri = format!("CREATE {}\n", merged.display());
        for a in archives {
            mri.push_str(&format!("ADDLIB {}\n", a.display()));
        }
        mri.push_str("SAVE\nEND\n");

        let mut child = Command::new(&ar)
            .arg("-M")
            .stdin(Stdio::piped())
            .spawn()
            .unwrap_or_else(|e| panic!("`{ar}` not found ({e}). Install binutils."));
        child.stdin.as_mut().unwrap().write_all(mri.as_bytes()).expect("write ar MRI");
        let ok = child.wait().expect("ar -M wait");
        if !ok.success() {
            panic!("\n\nar MRI merge failed.\n");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Return the number of logical CPUs to pass to cmake --parallel.
fn num_cpus() -> usize {
    // Respect MAKEFLAGS -jN if set.
    if let Ok(mf) = env::var("MAKEFLAGS") {
        for part in mf.split_whitespace() {
            if let Some(n) = part.strip_prefix("-j") {
                if let Ok(n) = n.parse::<usize>() {
                    return n;
                }
            }
        }
    }
    Command::new("nproc").output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok())
        .or_else(|| {
            Command::new("sysctl").args(["-n", "hw.logicalcpu"]).output().ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .and_then(|s| s.trim().parse().ok())
        })
        .unwrap_or(4)
}

fn has_dylib(dir: &Path) -> bool {
    if dir.join("libespeak-ng.dylib").exists() || dir.join("libespeak-ng.so").exists() {
        return true;
    }
    std::fs::read_dir(dir).ok().map_or(false, |entries| {
        entries.flatten().any(|e| {
            e.file_name().to_string_lossy().starts_with("libespeak-ng.so.")
        })
    })
}

fn pkg_config_libdir(target_os: &str) -> Option<String> {
    let mut extra: Vec<String> = Vec::new();
    if target_os == "macos" {
        for prefix in ["/opt/homebrew", "/usr/local"] {
            for sub in ["lib/pkgconfig", "share/pkgconfig"] {
                let p = format!("{prefix}/{sub}");
                if Path::new(&p).is_dir() { extra.push(p); }
            }
        }
        if let Some(keg) = brew_prefix("espeak-ng") {
            let p = format!("{keg}/lib/pkgconfig");
            if Path::new(&p).is_dir() { extra.insert(0, p); }
        }
    }
    if let Ok(existing) = env::var("PKG_CONFIG_PATH") { extra.push(existing); }
    let pkg_path = extra.join(":");
    let out = Command::new("pkg-config")
        .args(["--variable=libdir", "espeak-ng"])
        .env("PKG_CONFIG_PATH", &pkg_path)
        .output().ok()?;
    if out.status.success() {
        Some(String::from_utf8(out.stdout).ok()?.trim().to_owned())
    } else {
        None
    }
}

fn brew_prefix(formula: &str) -> Option<String> {
    let out = Command::new("brew").args(["--prefix", formula]).output().ok()?;
    if out.status.success() {
        Some(String::from_utf8(out.stdout).ok()?.trim().to_owned())
    } else {
        None
    }
}

/// Recursively find a file named `name` under `root`.
fn find_file(root: &Path, name: &str) -> Option<PathBuf> {
    if !root.is_dir() { return None; }
    let direct = root.join(name);
    if direct.exists() { return Some(direct); }
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let p = entry.path();
        if p.is_dir() {
            if let Some(found) = find_file(&p, name) { return Some(found); }
        }
    }
    None
}

/// Recursively copy a directory.
fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let dst_path = dst.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &dst_path)?;
        } else {
            std::fs::copy(entry.path(), dst_path)?;
        }
    }
    Ok(())
}

/// Return the first command from `candidates` that is found on PATH.
fn find_cmd(candidates: &[&str]) -> Option<String> {
    for &cmd in candidates {
        let probe = if cfg!(target_os = "windows") {
            Command::new("cmd").args(["/C", &format!("where {cmd}")]).output()
        } else {
            Command::new("sh").args(["-c", &format!("command -v {cmd}")]).output()
        };
        if probe.map(|o| o.status.success()).unwrap_or(false) {
            return Some(cmd.to_string());
        }
    }
    None
}

/// Returns `true` if cmake can use the given generator name.
/// We test by running `cmake -G <name> --version` in a temp dir.
fn cmake_generator_available(generator: &str) -> bool {
    let tmp = env::temp_dir().join("cmake-gen-probe");
    let _ = std::fs::create_dir_all(&tmp);
    // A deliberately empty CMakeLists.txt so cmake can probe the generator.
    let _ = std::fs::write(tmp.join("CMakeLists.txt"), "cmake_minimum_required(VERSION 3.15)\n");
    let out = Command::new("cmake")
        .arg(format!("-G{generator}"))
        .arg(".")
        .current_dir(&tmp)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    let _ = std::fs::remove_dir_all(&tmp);
    out.map(|s| s.success()).unwrap_or(false)
}

fn candidate_dirs(target_os: &str, target_arch: &str) -> Vec<PathBuf> {
    let mut dirs: Vec<String> = Vec::new();
    if target_os == "macos" {
        if let Some(keg) = brew_prefix("espeak-ng") { dirs.push(format!("{keg}/lib")); }
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
            "s390x"       => "s390x-linux-gnu",
            "powerpc64le" => "powerpc64le-linux-gnu",
            _             => "",
        };
        if !multiarch.is_empty() { dirs.push(format!("/usr/lib/{multiarch}")); }
        dirs.extend(["/usr/lib64", "/usr/lib", "/usr/local/lib"].map(String::from));
    }
    dirs.into_iter().map(PathBuf::from).filter(|p| p.is_dir()).collect()
}
