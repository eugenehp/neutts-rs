#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/build-espeak.sh
#
# Download, build, and install a static libespeak-ng into
#   <crate-root>/espeak-static/install/
#
# This script produces the exact same layout that build.rs's auto-build
# (step 4) produces, so you can pre-run it manually and the build script will
# pick up the cached result without downloading again.
#
# Usage:
#   bash scripts/build-espeak.sh
#
# After running, build with:
#   cargo build --features espeak          # build.rs finds the cache automatically
#
# Or point explicitly:
#   ESPEAK_LIB_DIR=espeak-static/install/lib cargo build --features espeak
#
# Requirements:
#   cmake  (brew install cmake  /  sudo apt install cmake)
#   curl or wget
#   A C++ compiler (clang on macOS, gcc/g++ on Linux)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ESPEAK_VERSION="1.52.0"  # first release with cmake (1.51.x used autoconf)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_ROOT="$(dirname "$SCRIPT_DIR")"
ROOT="$CRATE_ROOT/espeak-static"
SRC_DIR="$ROOT/src"
BUILD_DIR="$ROOT/build"
INSTALL_DIR="$ROOT/install"
TARBALL="$ROOT/espeak-ng-${ESPEAK_VERSION}.tar.gz"
TARBALL_URL="https://github.com/espeak-ng/espeak-ng/archive/refs/tags/${ESPEAK_VERSION}.tar.gz"

# ── Detect parallelism ────────────────────────────────────────────────────────
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# ── Colour output helpers ─────────────────────────────────────────────────────
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
blue()  { printf '\033[34m%s\033[0m\n' "$*"; }
warn()  { printf '\033[33mWARN: %s\033[0m\n' "$*" >&2; }
die()   { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

bold "espeak-ng $ESPEAK_VERSION static build"
blue "  install → $INSTALL_DIR"

# ── Check prerequisites ───────────────────────────────────────────────────────
if ! command -v cmake >/dev/null 2>&1; then
    die "cmake not found.\n  macOS: brew install cmake\n  Ubuntu: sudo apt install cmake"
fi
if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    die "Neither curl nor wget found. Install one to download the tarball."
fi

# ── Already done? ─────────────────────────────────────────────────────────────
STATIC_LIB="$INSTALL_DIR/lib/libespeak-ng.a"
STAMP="$INSTALL_DIR/lib/espeak-ng-merged.stamp"
if [ -f "$STATIC_LIB" ] && [ -f "$STAMP" ] && grep -q "^$ESPEAK_VERSION" "$STAMP" 2>/dev/null; then
    green "Already built and merged: $STATIC_LIB"
    echo "Nothing to do. Delete espeak-static/ to force a rebuild."
    exit 0
fi
# Remove stale lib so it gets re-merged cleanly.
rm -f "$STATIC_LIB" "$STAMP"

mkdir -p "$SRC_DIR" "$BUILD_DIR" "$INSTALL_DIR"

# ── Download ──────────────────────────────────────────────────────────────────
if [ ! -f "$TARBALL" ]; then
    blue "Downloading espeak-ng $ESPEAK_VERSION…"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$TARBALL" "$TARBALL_URL"
    else
        wget -q -O "$TARBALL" "$TARBALL_URL"
    fi
    green "  downloaded → $TARBALL"
fi

# ── Extract ───────────────────────────────────────────────────────────────────
if [ ! -f "$SRC_DIR/CMakeLists.txt" ]; then
    blue "Extracting…"
    tar -xzf "$TARBALL" -C "$SRC_DIR" --strip-components=1
    green "  extracted → $SRC_DIR"
fi

# ── cmake configure ───────────────────────────────────────────────────────────
blue "Configuring (cmake)…"
cmake \
    -S "$SRC_DIR" \
    -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DBUILD_SHARED_LIBS=OFF

# ── Build ─────────────────────────────────────────────────────────────────────
blue "Building with $JOBS jobs…"
cmake --build "$BUILD_DIR" -j "$JOBS"

# ── Install headers + data ────────────────────────────────────────────────────
# espeak-ng 1.52.0's cmake install() skips static archives (LIBRARY keyword).
# Run it to get headers and data, then copy the .a from the build tree.
blue "Installing headers and data…"
cmake --install "$BUILD_DIR" || true   # allowed to fail (skips .a)

# ── Merge ALL .a files from build tree into one fat libespeak-ng.a ───────────
#
# espeak-ng 1.52.0 builds several companion archives alongside libespeak-ng.a:
#   build/src/ucd-tools/libucd.a             (ucd_isalpha, ucd_isdigit, …)
#   build/src/speechPlayer/libspeechPlayer.a (speechPlayer_initialize, …)
#
# libespeak-ng.a references symbols from these but does NOT bundle them.
# We must merge everything into one fat archive so the linker is satisfied.
# (Same approach as kittentts-rs ios/build_rust_ios.sh.)
if [ ! -f "$STATIC_LIB" ]; then
    mkdir -p "$INSTALL_DIR/lib"

    # Collect all .a files under the cmake build tree.
    mapfile -t ALL_LIBS < <(find "$BUILD_DIR" -name "*.a" 2>/dev/null | sort)

    if [ "${#ALL_LIBS[@]}" -eq 0 ]; then
        die "No .a files found after building espeak-ng in $BUILD_DIR"
    fi

    blue "Merging ${#ALL_LIBS[@]} archive(s) into libespeak-ng.a…"
    for L in "${ALL_LIBS[@]}"; do echo "  $L"; done

    if command -v libtool >/dev/null 2>&1; then
        # macOS: libtool -static is always available with Xcode CLT
        libtool -static -o "$STATIC_LIB" "${ALL_LIBS[@]}"
    else
        # Linux: use ar -M with an MRI script
        {
            echo "CREATE $STATIC_LIB"
            for L in "${ALL_LIBS[@]}"; do echo "ADDLIB $L"; done
            echo "SAVE"
            echo "END"
        } | ar -M
    fi

    green "  merged → $STATIC_LIB"
    echo "$ESPEAK_VERSION" > "$STAMP"
fi

# ── Copy espeak-ng-data if cmake --install missed it ─────────────────────────
DATA_DEST="$INSTALL_DIR/share/espeak-ng-data"
if [ ! -d "$DATA_DEST" ] && [ -d "$SRC_DIR/espeak-ng-data" ]; then
    mkdir -p "$INSTALL_DIR/share"
    cp -r "$SRC_DIR/espeak-ng-data" "$DATA_DEST"
    green "  copied espeak-ng-data → $DATA_DEST"
fi

# ── Result ────────────────────────────────────────────────────────────────────
if [ -f "$STATIC_LIB" ]; then
    green "\nDone!  Static library: $STATIC_LIB"
else
    # Shared lib is acceptable too (e.g. on systems where cmake ignores BUILD_SHARED_LIBS)
    DYLIB=$(find "$INSTALL_DIR/lib" -name "libespeak-ng*.dylib" -o -name "libespeak-ng*.so*" 2>/dev/null | head -1)
    if [ -n "$DYLIB" ]; then
        green "\nDone! (dynamic library) $DYLIB"
    else
        die "Build completed but no library found in $INSTALL_DIR/lib"
    fi
fi

echo ""
blue "Data directory:"
DATA_DIR=$(find "$INSTALL_DIR/share" -type d -name "espeak-ng-data" 2>/dev/null | head -1)
if [ -n "$DATA_DIR" ]; then
    echo "  $DATA_DIR"
    green ""
    echo "Build with espeak support (build.rs finds the cache automatically):"
    echo "  cargo build --features espeak"
    echo ""
    echo "Or point explicitly:"
    echo "  ESPEAK_LIB_DIR=$INSTALL_DIR/lib cargo build --features espeak"
else
    warn "espeak-ng-data not found under $INSTALL_DIR — you may need to call"
    warn "neutts::phonemize::set_data_path() at runtime"
fi
