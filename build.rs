//! Build script — locates and links `libespeak-ng` for the `espeak` feature.
//!
//! ## Resolution order
//!
//! 1. **`ESPEAK_LIB_DIR`** env var — explicit lib directory.  Required for
//!    mobile cross-compilation.
//!
//! 2. **pkg-config** — augmented with Homebrew's pkgconfig dirs on macOS.
//!
//! 3. **Platform path walk** — Homebrew Cellar/opt dirs on macOS; multi-arch
//!    and standard lib dirs on Linux.
//!
//! 4. **Auto-build from source** — downloads espeak-ng from GitHub, builds a
//!    static library with cmake, and caches the result in
//!    `<crate-root>/espeak-static/`.  Skipped if `ESPEAK_NO_DOWNLOAD=1`.
//!
//! The build also emits `NEUTTS_ESPEAK_DATA_DIR` so `phonemize.rs` can find
//! the data directory at compile time (used when building from source).

use std::path::{Path, PathBuf};
use std::process::Command;

// ─── espeak-ng version pinned for reproducible source builds ─────────────────
// 1.52.0 is the first release to use cmake (1.51.x used autoconf).
const ESPEAK_VERSION: &str = "1.52.0";
// SHA-256 verification is intentionally omitted — the tarball comes from the
// official GitHub releases page over HTTPS.  Add integrity checks here if your
// threat model requires it.

fn main() {
    let target_os   = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();

    println!("cargo:rerun-if-env-changed=ESPEAK_LIB_DIR");
    println!("cargo:rerun-if-env-changed=ESPEAK_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=ESPEAK_NO_DOWNLOAD");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR");
    println!("cargo:rerun-if-env-changed=ESPEAK_BUILD_SCRIPT");
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");

    // When the merged espeak-ng library is rebuilt, the stamp file changes.
    // Watching it re-runs this build script, which then emits a new
    // NEUTTS_ESPEAK_STAMP env-var value — that change forces Cargo to
    // recompile (and re-bundle) the Rust crate with the fresh library.
    let stamp_path = Path::new(&manifest_dir)
        .join("espeak-static/install/lib/espeak-ng-merged.stamp");
    println!("cargo:rerun-if-changed={}", stamp_path.display());

    // Only link espeak when the feature is enabled.
    if std::env::var("CARGO_FEATURE_ESPEAK").is_err() {
        return;
    }

    // Emit the stamp value as a rustc-env so that when the merged library is
    // rebuilt (stamp changes → build script re-runs → this value changes),
    // Cargo recompiles the crate and picks up the new bundled objects.
    let stamp_val = std::fs::read_to_string(&stamp_path)
        .unwrap_or_else(|_| "none".to_owned());
    println!("cargo:rustc-env=NEUTTS_ESPEAK_STAMP={}", stamp_val.trim());

    // ── 1. Explicit override ──────────────────────────────────────────────────
    if let Ok(dir) = std::env::var("ESPEAK_LIB_DIR") {
        if !Path::new(&dir).join("libespeak-ng.a").exists() {
            if let Ok(script) = std::env::var("ESPEAK_BUILD_SCRIPT") {
                run_shell_script(&script, &target_os);
            }
        }
        emit_static_link(&dir, &target_os);
        return;
    }

    // Mobile without explicit dir: fail early — no source build.
    if matches!(&*target_os, "ios" | "android") {
        panic!(
            "\n\nESPEAK_LIB_DIR is not set.\n\
             Cross-compiling for {target_os} requires a pre-built static libespeak-ng:\n\
             \n\
             \t1. Cross-compile espeak-ng for your target ABI.\n\
             \t2. Set ESPEAK_LIB_DIR to the directory containing libespeak-ng.a\n"
        );
    }

    // ── 2. pkg-config ─────────────────────────────────────────────────────────
    if let Some(dir) = pkg_config_libdir(&target_os) {
        if let Some(()) = try_link_dir(&dir, &target_os) { return; }
    }

    // ── 3. Platform path walk ─────────────────────────────────────────────────
    for dir in candidate_dirs(&target_os, &target_arch) {
        if let Some(()) = try_link_dir(dir.to_str().unwrap_or(""), &target_os) {
            return;
        }
    }

    // ── 4. Auto-build from source ─────────────────────────────────────────────
    if std::env::var("ESPEAK_NO_DOWNLOAD").is_ok() {
        panic!(
            "\n\nespeak-ng not found and ESPEAK_NO_DOWNLOAD is set.\n\
             Unset ESPEAK_NO_DOWNLOAD or install espeak-ng manually.\n\n"
        );
    }

    eprintln!(
        "cargo:warning=espeak-ng not found on system — \
         downloading and building from source (espeak-ng {ESPEAK_VERSION}).\n\
         Set ESPEAK_NO_DOWNLOAD=1 to suppress this."
    );

    match build_from_source(&manifest_dir, &target_os) {
        Ok(lib_dir) => {
            let d = lib_dir.to_string_lossy();
            // emit_static_link() is NOT used here: the source build copies each
            // companion archive individually (libucd.a, libspeechPlayer.a, …)
            // so we enumerate install/lib/*.a and emit a separate link directive
            // for each.  This avoids the fragile libtool-merge approach.
            if has_dylib(&d) {
                println!("cargo:rustc-link-search=native={d}");
                println!("cargo:rustc-link-lib=espeak-ng");
            } else {
                link_all_static_in_dir(&lib_dir, &target_os);
            }
            emit_data_dir(&lib_dir);
        }
        Err(e) => panic!("\n\nespeak-ng source build failed: {e}\n\n"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Source build
// ─────────────────────────────────────────────────────────────────────────────

/// Download, configure, and build espeak-ng from source.
///
/// Everything is stored under `<crate-root>/espeak-static/` so it survives
/// `cargo clean` and only needs to be done once.
///
/// ## Layout produced
/// ```text
/// espeak-static/
///   espeak-ng-1.52.0.tar.gz   (cached tarball)
///   src/                      (extracted source)
///   build/                    (cmake build tree)
///   install/
///     lib/
///       libespeak-ng.a        ← linked by Cargo
///     include/…
///     share/
///       espeak-ng-data/       ← runtime data (baked into library path)
/// ```
fn build_from_source(manifest_dir: &str, target_os: &str) -> Result<PathBuf, String> {
    let root        = Path::new(manifest_dir).join("espeak-static");
    let src_dir     = root.join("src");
    let build_dir   = root.join("build");
    let install_dir = root.join("install");
    let lib_dir     = install_dir.join("lib");

    for d in &[&src_dir, &build_dir, &install_dir] {
        std::fs::create_dir_all(d)
            .map_err(|e| format!("create {}: {e}", d.display()))?;
    }

    // ── Already built? ────────────────────────────────────────────────────────
    let static_lib = lib_dir.join("libespeak-ng.a");
    // Stamp written only after a successful merged-archive build.
    // Format: "<version>\n" so a version bump forces a rebuild automatically.
    let stamp = lib_dir.join("espeak-ng-merged.stamp");
    let stamp_ok = stamp.exists()
        && std::fs::read_to_string(&stamp)
            .map(|s| s.trim() == ESPEAK_VERSION)
            .unwrap_or(false);

    if stamp_ok && (static_lib.exists() || has_dylib(lib_dir.to_str().unwrap_or(""))) {
        eprintln!(
            "cargo:warning=Using cached espeak-ng {ESPEAK_VERSION} at {}",
            lib_dir.display()
        );
        return Ok(lib_dir);
    }

    if static_lib.exists() && !stamp_ok {
        eprintln!(
            "cargo:warning=espeak-ng cache is stale (no merge stamp) — re-merging…"
        );
        let _ = std::fs::remove_file(&static_lib);
    }

    // ── Download tarball ──────────────────────────────────────────────────────
    let tarball = root.join(format!("espeak-ng-{ESPEAK_VERSION}.tar.gz"));
    if !tarball.exists() {
        let url = format!(
            "https://github.com/espeak-ng/espeak-ng/archive/refs/tags/{ESPEAK_VERSION}.tar.gz"
        );
        eprintln!("cargo:warning=Downloading {url}");
        download(&url, &tarball)?;
    }

    // ── Extract ───────────────────────────────────────────────────────────────
    if !src_dir.join("CMakeLists.txt").exists() {
        eprintln!("cargo:warning=Extracting espeak-ng {ESPEAK_VERSION}…");
        run_cmd(
            Command::new("tar")
                .args([
                    "-xzf", tarball.to_str().unwrap(),
                    "-C",   src_dir.to_str().unwrap(),
                    "--strip-components=1",
                ]),
            "tar",
        )?;
        if !src_dir.join("CMakeLists.txt").exists() {
            return Err(format!(
                "tar extracted to {} but CMakeLists.txt is missing.\n\
                 Delete espeak-static/ and retry.",
                src_dir.display()
            ));
        }
    }

    // Augment PATH so cmake/make/ninja are found on macOS where Cargo often
    // runs with a minimal PATH that omits /opt/homebrew/bin.
    let path_env = augmented_path(target_os);

    // ── cmake configure ───────────────────────────────────────────────────────
    // Use explicit -S / -B to avoid cmake misinterpreting the source path.
    // espeak-ng 1.52.0 cmake notes:
    //  • BUILD_SHARED_LIBS defaults to OFF (static library is the default)
    //  • Optional deps (libsonic, libpcaudio, mbrola) are auto-detected;
    //    they're omitted from the library if not found — no -DUSE_* needed
    //  • PATH_ESPEAK_DATA is baked in from CMAKE_INSTALL_PREFIX/share/espeak-ng-data
    eprintln!(
        "cargo:warning=Configuring espeak-ng with cmake (install → {})…",
        install_dir.display()
    );
    let mut cmake_args: Vec<String> = vec![
        "-S".into(), src_dir.to_string_lossy().into_owned(),
        "-B".into(), build_dir.to_string_lossy().into_owned(),
        format!("-DCMAKE_INSTALL_PREFIX={}", install_dir.display()),
        "-DCMAKE_BUILD_TYPE=Release".into(),
        "-DBUILD_SHARED_LIBS=OFF".into(),
    ];
    // Match Rust's macOS deployment target so espeak-ng C objects are compiled
    // for the same minimum OS version.  Without this the linker emits
    // "was built for newer macOS (26.x) than being linked (11.0)" for every
    // espeak-ng object file.
    if target_os == "macos" {
        let osx_target = std::env::var("MACOSX_DEPLOYMENT_TARGET")
            .unwrap_or_else(|_| "11.0".to_owned());
        cmake_args.push(format!("-DCMAKE_OSX_DEPLOYMENT_TARGET={osx_target}"));
    }

    run_cmd(
        Command::new("cmake")
            .env("PATH", &path_env)
            .args(&cmake_args),
        "cmake -S … -B …",
    )?;

    // ── Build ─────────────────────────────────────────────────────────────────
    let jobs = available_parallelism();
    eprintln!("cargo:warning=Building espeak-ng with {jobs} jobs…");
    run_cmd(
        Command::new("cmake")
            .env("PATH", &path_env)
            .args([
                "--build", build_dir.to_str().unwrap(),
                "-j", &jobs.to_string(),
            ]),
        "cmake --build",
    )?;

    // ── Install headers + data ────────────────────────────────────────────────
    // `cmake --install` with espeak-ng 1.52.0 does NOT install the static
    // archive (install(TARGETS espeak-ng LIBRARY) skips ARCHIVE targets).
    // We run it anyway to get headers and the espeak-ng-data directory, then
    // copy the .a from the build tree manually.
    eprintln!("cargo:warning=Installing espeak-ng headers and data…");
    let _ = Command::new("cmake")
        .env("PATH", &path_env)
        .args(["--install", build_dir.to_str().unwrap()])
        .status(); // ignore failure — .a copy below is the critical step

    // ── Copy each companion archive individually to install/lib/ ─────────────
    //
    // espeak-ng 1.52.0 cmake builds three separate static libraries:
    //
    //   build/src/libespeak-ng/libespeak-ng.a   main TTS library
    //   build/src/ucd-tools/libucd.a            Unicode helpers (ucd_isalpha…)
    //   build/src/speechPlayer/libspeechPlayer.a Klatt synthesizer
    //
    // libespeak-ng.a calls into libucd and libspeechPlayer but does NOT bundle
    // them.  Rather than merging with libtool (fragile), we copy each archive
    // to install/lib/ and emit a separate cargo:rustc-link-lib directive for
    // each.  The linker then resolves all symbols automatically.
    if !stamp_ok || !static_lib.exists() {
        std::fs::create_dir_all(&lib_dir)
            .map_err(|e| format!("create lib_dir: {e}"))?;
        copy_companion_libs(&build_dir, &lib_dir)?;
        // Write stamp only after all libraries are in place.
        std::fs::write(&stamp, format!("{ESPEAK_VERSION}\n"))
            .map_err(|e| eprintln!("cargo:warning=Could not write merge stamp: {e}")).ok();
    }

    // ── Copy espeak-ng-data if cmake --install missed it ─────────────────────
    let data_dest = install_dir.join("share").join("espeak-ng-data");
    if !data_dest.exists() {
        let data_src = src_dir.join("espeak-ng-data");
        if data_src.exists() {
            copy_dir_all(&data_src, &data_dest)
                .map_err(|e| format!("copy espeak-ng-data: {e}"))?;
            eprintln!("cargo:warning=Copied espeak-ng-data → {}", data_dest.display());
        }
    }

    // ── Verify ────────────────────────────────────────────────────────────────
    if static_lib.exists() {
        eprintln!(
            "cargo:warning=espeak-ng {ESPEAK_VERSION} built → {}",
            static_lib.display()
        );
    } else if has_dylib(lib_dir.to_str().unwrap_or("")) {
        eprintln!(
            "cargo:warning=espeak-ng {ESPEAK_VERSION} (dylib) → {}",
            lib_dir.display()
        );
    } else {
        return Err(format!(
            "Build completed but no library found in {}.\n\
             Inspect the build output above for cmake errors.",
            lib_dir.display()
        ));
    }

    Ok(lib_dir)
}

/// Copy every `.a` found under `build_dir` into `dest_lib_dir`, keeping their
/// original filenames (`libespeak-ng.a`, `libucd.a`, `libspeechPlayer.a`, ...).
///
/// We deliberately avoid the libtool-merge approach: merging is fragile (the
/// merged archive may silently omit symbols).  Instead we keep separate archives
/// and emit a separate `cargo:rustc-link-lib` for each — the linker resolves
/// all cross-archive symbol references automatically.
fn copy_companion_libs(build_dir: &Path, dest_lib_dir: &Path) -> Result<(), String> {
    let mut found: Vec<PathBuf> = Vec::new();
    collect_static_libs(build_dir, &mut found);

    if found.is_empty() {
        return Err(format!(
            "No .a files found in cmake build tree at {}.\n             Make sure `cmake --build` completed without errors.",
            build_dir.display()
        ));
    }

    eprintln!("cargo:warning=Copying {} archive(s) to {}:", found.len(), dest_lib_dir.display());
    for src in &found {
        let dst = dest_lib_dir.join(src.file_name().unwrap());
        std::fs::copy(src, &dst)
            .map_err(|e| format!("copy {} -> {}: {e}", src.display(), dst.display()))?;
        eprintln!("cargo:warning=  {}", src.file_name().unwrap().to_string_lossy());
    }
    Ok(())
}

/// Emit `cargo:rustc-link-lib=static=<name>` for every `.a` in `lib_dir`.
///
/// This is the companion to `copy_companion_libs`: once all archives are in
/// `lib_dir`, this function tells Cargo to link all of them.  We also emit
/// the search path and (on macOS) the C++ stdlib link.
fn link_all_static_in_dir(lib_dir: &Path, target_os: &str) {
    let d = lib_dir.to_string_lossy();
    println!("cargo:rustc-link-search=native={d}");

    let mut linked: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(lib_dir) {
        let mut entries: Vec<_> = rd.flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |e| e == "a"))
            .collect();
        entries.sort();
        for path in entries {
            if let Some(stem) = path.file_stem() {
                let name = stem.to_string_lossy();
                let lib_name = name.strip_prefix("lib").unwrap_or(&name);
                println!("cargo:rustc-link-lib=static={lib_name}");
                linked.push(lib_name.to_string());
            }
        }
    }
    eprintln!("cargo:warning=Linking static: {}", linked.join(", "));
    link_cxx(target_os);
}

#[allow(dead_code)]
/// Walk `dir` recursively and return the first file whose name equals `name`.
fn find_file_in_dir(dir: &Path, name: &str) -> Option<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else { return None; };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(f) = find_file_in_dir(&path, name) { return Some(f); }
        } else if path.file_name().map_or(false, |n| n == name) {
            return Some(path);
        }
    }
    None
}

/// Walk `dir` recursively, appending every `.a` file to `out`.
fn collect_static_libs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return; };
    let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_static_libs(&path, out);
        } else if path.extension().map_or(false, |e| e == "a") {
            out.push(path);
        }
    }
}

/// Recursively copy `src` directory to `dst`.
fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let target = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_all(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

/// Emit `NEUTTS_ESPEAK_DATA_DIR` so `phonemize::do_init` can find the data
/// directory without a runtime call to `set_data_path()`.
fn emit_data_dir(lib_dir: &Path) {
    // install prefix is the parent of lib/
    let prefix = lib_dir.parent().unwrap_or(lib_dir);
    let candidates = [
        prefix.join("share").join("espeak-ng-data"), // cmake install layout
        prefix.join("lib").join("espeak-ng-data"),   // older layout
        lib_dir.join("espeak-ng-data"),              // rare flat layout
    ];
    for c in &candidates {
        if c.exists() {
            println!("cargo:rustc-env=NEUTTS_ESPEAK_DATA_DIR={}", c.display());
            eprintln!("cargo:warning=espeak-ng data → {}", c.display());
            return;
        }
    }
    eprintln!(
        "cargo:warning=espeak-ng data dir not found under {} — \
         call neutts::phonemize::set_data_path() at runtime",
        prefix.display()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Link helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Try to link from `dir`.  Returns `Some(())` if a library was found and
/// the linker directives were emitted; `None` otherwise.
fn try_link_dir(dir: &str, target_os: &str) -> Option<()> {
    if Path::new(dir).join("libespeak-ng.a").exists() {
        emit_static_link(dir, target_os);
        return Some(());
    }
    if matches!(target_os, "linux" | "macos") && has_dylib(dir) {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=espeak-ng");
        return Some(());
    }
    None
}

fn emit_static_link(dir: &str, target_os: &str) {
    let static_lib = Path::new(dir).join("libespeak-ng.a");
    if !static_lib.exists() {
        panic!(
            "\n\nESPEAK_LIB_DIR={dir:?}: libespeak-ng.a not found.\n\n"
        );
    }
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=static=espeak-ng");
    link_cxx(target_os);
}

fn link_cxx(target_os: &str) {
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

fn has_dylib(dir: &str) -> bool {
    let dir = Path::new(dir);
    if dir.join("libespeak-ng.so").exists()    { return true; }
    if dir.join("libespeak-ng.dylib").exists() { return true; }
    std::fs::read_dir(dir).ok().map_or(false, |mut e| {
        e.any(|e| e.ok().map_or(false, |e| {
            let name = e.file_name();
            let s    = name.to_string_lossy();
            s.starts_with("libespeak-ng.so.")
                || (s.starts_with("libespeak-ng.") && s.ends_with(".dylib"))
        }))
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Discovery helpers
// ─────────────────────────────────────────────────────────────────────────────

fn pkg_config_libdir(target_os: &str) -> Option<String> {
    let mut extra: Vec<String> = Vec::new();
    if target_os == "macos" {
        for prefix in ["/opt/homebrew", "/usr/local"] {
            let p = format!("{prefix}/lib/pkgconfig");
            if Path::new(&p).is_dir() { extra.push(p); }
            let p = format!("{prefix}/share/pkgconfig");
            if Path::new(&p).is_dir() { extra.push(p); }
        }
        if let Some(keg) = brew_prefix("espeak-ng") {
            let p = format!("{keg}/lib/pkgconfig");
            if Path::new(&p).is_dir() { extra.insert(0, p); }
        }
    }
    let existing = std::env::var("PKG_CONFIG_PATH").unwrap_or_default();
    if !existing.is_empty() { extra.push(existing); }
    let pkg_path = extra.join(":");
    let out = Command::new("pkg-config")
        .args(["--variable=libdir", "espeak-ng"])
        .env("PKG_CONFIG_PATH", &pkg_path)
        .output().ok()?;
    if out.status.success() {
        Some(String::from_utf8(out.stdout).ok()?.trim().to_owned())
    } else { None }
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
        // Cellar version dirs: /opt/homebrew/Cellar/espeak-ng/<version>/lib
        for cellar in ["/opt/homebrew/Cellar/espeak-ng", "/usr/local/Cellar/espeak-ng"] {
            if let Ok(rd) = std::fs::read_dir(cellar) {
                for entry in rd.flatten() {
                    dirs.push(format!("{}/lib", entry.path().display()));
                }
            }
        }
    } else {
        let multiarch = match &*target_arch {
            "x86_64"      => "x86_64-linux-gnu",
            "aarch64"     => "aarch64-linux-gnu",
            "arm"         => "arm-linux-gnueabihf",
            "riscv64"     => "riscv64-linux-gnu",
            "s390x"       => "s390x-linux-gnu",
            "powerpc64le" => "powerpc64le-linux-gnu",
            _             => "",
        };
        if !multiarch.is_empty() {
            dirs.push(format!("/usr/lib/{multiarch}"));
        }
        dirs.extend(["/usr/lib64", "/usr/lib", "/usr/local/lib"].map(String::from));
    }
    dirs.into_iter()
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
        .collect()
}

fn brew_prefix(formula: &str) -> Option<String> {
    // Try well-known full paths first — Cargo's PATH often omits Homebrew.
    for brew in &["/opt/homebrew/bin/brew", "/usr/local/bin/brew", "brew"] {
        if let Ok(out) = Command::new(brew).args(["--prefix", formula]).output() {
            if out.status.success() {
                if let Ok(s) = String::from_utf8(out.stdout) {
                    let s = s.trim().to_owned();
                    if !s.is_empty() { return Some(s); }
                }
            }
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Process helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Download `url` to `dest` using `curl` or `wget`.
fn download(url: &str, dest: &Path) -> Result<(), String> {
    // curl
    if let Ok(status) = Command::new("curl")
        .args(["-fsSL", "-o", dest.to_str().unwrap(), url])
        .status()
    {
        if status.success() { return Ok(()); }
    }
    // wget
    if let Ok(status) = Command::new("wget")
        .args(["-q", "-O", dest.to_str().unwrap(), url])
        .status()
    {
        if status.success() { return Ok(()); }
    }
    Err(format!(
        "Failed to download {url}\n\
         Install curl or wget, or download manually and set ESPEAK_LIB_DIR."
    ))
}

/// Run a command and return `Ok(())` on success, `Err(message)` on failure.
fn run_cmd(cmd: &mut Command, label: &str) -> Result<(), String> {
    match cmd.status() {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(format!(
            "`{label}` not found in PATH.\n\
             Install it:\n\
             \t  macOS  :  brew install cmake\n\
             \t  Ubuntu :  sudo apt install cmake\n\
             \t  Alpine :  apk add cmake"
        )),
        Err(e) => Err(format!("{label}: {e}")),
        Ok(s) if !s.success() => Err(format!("{label} exited with {s}")),
        Ok(_) => Ok(()),
    }
}

fn run_shell_script(script: &str, target_os: &str) {
    let path = std::fs::canonicalize(script)
        .unwrap_or_else(|_| Path::new(script).to_path_buf());
    if !path.exists() {
        eprintln!("neutts build.rs: {script:?} not found — skipping.");
        return;
    }
    let path_env = augmented_path(target_os);
    let status = Command::new("bash")
        .args(["-c", &format!("exec 1>&2; bash '{}'", path.display())])
        .env("PATH", &path_env)
        .status()
        .unwrap_or_else(|e| panic!("failed to run {script:?}: {e}"));
    if !status.success() {
        panic!("\n\n{script:?} failed ({status}).\n\n");
    }
}

fn augmented_path(target_os: &str) -> String {
    let current = std::env::var("PATH").unwrap_or_default();
    if target_os == "macos" {
        format!("/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:{current}")
    } else {
        format!("/usr/local/bin:/usr/bin:/bin:{current}")
    }
}

fn available_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
