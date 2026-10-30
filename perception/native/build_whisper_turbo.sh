#!/bin/sh
# Build the CUDA Whisper Turbo ASR worker from the image-owned whisper.cpp
# source tarball. Intended for a builder image that has cmake, g++ and nvcc;
# the runtime image itself does not ship a compiler.
set -eu

model_root=${1:-/opt/asr-models/icefall}
install_root=${2:-/opt/whisper-turbo}
worker_source=${3:-/work/native/whisper_turbo_worker.cc}
json_header=${4:-/work/native/json.hpp}
cuda_home=${CUDA_HOME:-/usr/local/cuda}

source_root=$(mktemp -d)
trap 'rm -rf "$source_root"' EXIT HUP INT TERM

tar -xzf "$model_root/whisper.cpp-source.tar.gz" -C "$source_root"
cmake -S "$source_root/whisper.cpp" -B "$source_root/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$install_root" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DGGML_NATIVE=ON -DGGML_OPENMP=OFF \
    -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=OFF \
    -DWHISPER_BUILD_SERVER=OFF
cmake --build "$source_root/build" --target whisper -j 2
cmake --install "$source_root/build"

mkdir -p "$install_root/bin"
c++ -O3 -std=c++17 "$worker_source" \
    -I"$source_root/whisper.cpp/include" -I"$source_root/whisper.cpp/ggml/include" \
    -I"$(dirname "$json_header")" \
    -L"$install_root/lib" -L"$cuda_home/lib64" \
    -lwhisper -lggml -lggml-base -lggml-cpu -lggml-cuda \
    -lcublas -lcudart \
    -Wl,-rpath,"$install_root/lib" \
    -o "$install_root/bin/whisper_turbo_worker"
