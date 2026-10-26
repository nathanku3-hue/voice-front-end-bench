#!/usr/bin/env bash
# Build every C++ target with the zig-provided clang, without CMake.
#
# Why this exists: this Windows host has no MSVC C++ toolchain, no g++ and no
# CMake, and WSL is blocked. zig cc (clang 21.1.0, mingw target) is the only
# working C++17 compiler, and it needs FFTW's win64 import library prepared with
# `zig dlltool`. CMakeLists.txt remains the canonical build for Linux/CI; this
# script is the local reproduction of it.
#
# Usage:  bash scripts/build_windows_zig.sh [--clean]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ZIG="${ZIG:-.workbuddy-ai/venv-cpp/Lib/site-packages/ziglang/zig.exe}"
OUT=build/win-zig
FFTW_DIR="$OUT/fftw"

if [[ ! -x "$ZIG" ]]; then
  echo "zig not found at $ZIG (pip install ziglang into .workbuddy-ai/venv-cpp)" >&2
  exit 1
fi

if [[ "${1:-}" == "--clean" ]]; then
  rm -rf "$OUT"
fi
mkdir -p "$OUT" "$FFTW_DIR"

# FFTW 3.3.5 win64: the DLL plus an import library. Both are gitignored; fetch
# https://fftw.org/pub/fftw/fftw-3.3.5-dll64.zip and unpack into /tmp/fftw.
if [[ ! -f "$FFTW_DIR/libfftw3-3.dll.a" ]]; then
  if [[ -f /tmp/fftw/libfftw3-3.def ]]; then
    "$ZIG" dlltool -m i386:x86-64 -d /tmp/fftw/libfftw3-3.def \
      -l "$FFTW_DIR/libfftw3-3.dll.a" -D libfftw3-3.dll
    cp /tmp/fftw/libfftw3-3.dll "$OUT/"
    cp /tmp/fftw/fftw3.h "$FFTW_DIR/"
    echo "prepared FFTW import library from /tmp/fftw"
  else
    echo "missing FFTW: unpack fftw-3.3.5-dll64.zip into /tmp/fftw first" >&2
    exit 1
  fi
fi

CXXFLAGS=(-std=c++17 -O2 -Wall -Wextra -Iinclude -Itools -I"$FFTW_DIR")
LINK=("$FFTW_DIR/libfftw3-3.dll.a")

build() {  # build <output-name> <sources...>
  local name="$1"; shift
  echo "  $name"
  "$ZIG" c++ "${CXXFLAGS[@]}" "$@" "${LINK[@]}" -lpsapi -o "$OUT/$name.exe"
}

echo "building C++ targets into $OUT"
# Every target that links streaming_fbank.cpp also needs cpu_isa.cpp: the ISA
# dispatch is a separate translation unit so the CPUID probe runs once and the
# hot path never touches it.
FANGBANK_SRC=(src/streaming_fbank.cpp src/cpu_isa.cpp)
build test_streaming             tests/test_streaming.cpp "${FANGBANK_SRC[@]}"
build test_bounded_fbank_bridge  tests/test_bounded_fbank_bridge.cpp src/bounded_fbank_bridge.cpp "${FANGBANK_SRC[@]}"
build test_zero_alloc_steady_state tests/test_zero_alloc_steady_state.cpp src/alloc_probe.cpp src/bounded_fbank_bridge.cpp "${FANGBANK_SRC[@]}"
build test_hysteresis_flow       tests/test_hysteresis_flow.cpp src/bounded_fbank_bridge.cpp "${FANGBANK_SRC[@]}"
build test_mel_sparsity          tests/test_mel_sparsity.cpp "${FANGBANK_SRC[@]}"
build test_ctc_greedy_decoder    tests/test_ctc_greedy_decoder.cpp src/ctc_greedy_decoder.cpp
build test_fast_log              tests/test_fast_log.cpp
build test_axi4s_adapter         tests/test_axi4s_adapter.cpp src/alloc_probe.cpp src/axi4s_adapter.cpp src/bounded_fbank_bridge.cpp "${FANGBANK_SRC[@]}"
build fbank_stage_profile        tools/fbank_stage_profile.cpp "${FANGBANK_SRC[@]}"
build fbank_feature_dump         tools/fbank_feature_dump.cpp "${FANGBANK_SRC[@]}"
build fbank_ctc_pipeline         tools/fbank_ctc_pipeline.cpp src/bounded_fbank_bridge.cpp "${FANGBANK_SRC[@]}" src/ctc_greedy_decoder.cpp
build axi4s_sim                  tools/axi4s_sim.cpp src/axi4s_adapter.cpp src/bounded_fbank_bridge.cpp "${FANGBANK_SRC[@]}"

cp "$OUT/libfftw3-3.dll" "$OUT/" 2>/dev/null || true
echo "done"
