#!/bin/sh
set -eu

model_root=${1:-/opt/asr-models/icefall}
install_root=${2:-/opt/asr-language-id}
worker_source=${3:-/work/native/asr_language_id.cc}
source_root=$(mktemp -d)
trap 'rm -rf "$source_root"' EXIT HUP INT TERM
tar -xzf "$model_root/whisper.cpp-source.tar.gz" -C "$source_root"
cmake -S "$source_root/whisper.cpp" -B "$source_root/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$install_root" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DGGML_OPENMP=OFF \
    -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=OFF \
    -DWHISPER_BUILD_SERVER=OFF
cmake --build "$source_root/build" --target whisper -j 2
cmake --install "$source_root/build"
mkdir -p "$install_root/bin"
c++ -O3 -std=c++17 "$worker_source" \
    -I"$install_root/include" -L"$install_root/lib" -lwhisper \
    -Wl,-rpath,"$install_root/lib" -o "$install_root/bin/whisper-lid"
