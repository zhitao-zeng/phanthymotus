"""Build the CPU ASR runtime from a fresh sherpa-onnx v1.13.6 source tree.

Run with a matching Python ABI and setuptools/wheel/CMake/compiler installed.
--deps-dir can reuse CMake dependency sources without downloading them.
The source tree is a disposable build input; installed runtimes are not patched.
"""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def replace(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise RuntimeError("Unexpected sherpa-onnx source layout: " + str(path))
    path.write_text(text.replace(old, new))


def patch(source):
    native = Path(__file__).resolve().parent
    csrc = source / 'sherpa-onnx/csrc'
    replace(source / 'CMakeLists.txt', 'set(SHERPA_ONNX_VERSION "1.13.6")',
            'set(SHERPA_ONNX_VERSION "1.13.6+prefix1")')
    shutil.copyfile(native / 'prefix_lm.h', csrc / 'prefix-lm.h')
    shutil.copyfile(native / 'prefix_decoder.cc', csrc / 'offline-transducer-modified-beam-search-decoder.cc')
    replace(csrc / 'offline-lm.h', 'class OfflineLM {', 'class PrefixLm;\n\nclass OfflineLM {')
    replace(csrc / 'offline-lm.h', '  virtual ~OfflineLM() = default;',
            '  virtual ~OfflineLM() = default;\n  virtual PrefixLm *GetPrefixLm() { return nullptr; }')
    replace(csrc / 'offline-rnn-lm.h', '  ~OfflineRnnLM() override;',
            '  ~OfflineRnnLM() override;\n  PrefixLm *GetPrefixLm() override;')
    file = csrc / 'offline-rnn-lm.cc'
    replace(file, '#include "sherpa-onnx/csrc/offline-rnn-lm.h"',
            '#include "sherpa-onnx/csrc/offline-rnn-lm.h"\n#include "sherpa-onnx/csrc/prefix-lm.h"')
    replace(file, '  Ort::Value Rescore(Ort::Value x, Ort::Value x_lens) {',
            '  PrefixLm *GetPrefixLm() { return prefix_.get(); }\n\n'
            '  Ort::Value Rescore(Ort::Value x, Ort::Value x_lens) {\n'
            '    if (prefix_) throw std::runtime_error("Prefix LM requires transducer beam search");')
    replace(file, '    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);',
            '    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);\n'
            '    if (input_names_.size() == 3) prefix_ = std::make_unique<PrefixLm>(sess_.get());')
    replace(file, '  std::unique_ptr<Ort::Session> sess_;',
            '  std::unique_ptr<Ort::Session> sess_;\n  std::unique_ptr<PrefixLm> prefix_;')
    replace(file, 'OfflineRnnLM::~OfflineRnnLM() = default;',
            'OfflineRnnLM::~OfflineRnnLM() = default;\n\n'
            'PrefixLm *OfflineRnnLM::GetPrefixLm() { return impl_->GetPrefixLm(); }')
    init = source / 'sherpa-onnx/python/sherpa_onnx/__init__.py'
    init.write_text(init.read_text() + '\nXASR_PREFIX_LM_VERSION = 1\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--wheel-dir', type=Path, required=True)
    ap.add_argument('--ort-include', type=Path, required=True)
    ap.add_argument('--ort-lib', type=Path, required=True)
    ap.add_argument('--deps-dir', type=Path)
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--patched', action='store_true', help='Resume this already-patched build')
    args = ap.parse_args()
    source = args.source.resolve()
    if not args.patched:
        patch(source)
    # This is an ASR-only build; disabled TTS symbols are not exported by pybind.
    init = source / 'sherpa-onnx/python/sherpa_onnx/__init__.py'
    init.write_text(''.join(line for line in init.read_text().splitlines(keepends=True)
                           if not line.startswith('    OfflineTts')
                           and line.strip() != 'GenerationConfig,'))
    options = ['-DCMAKE_BUILD_TYPE=Release', '-DSHERPA_ONNX_ENABLE_BINARY=OFF',
               '-DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF', '-DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF',
               '-DSHERPA_ONNX_ENABLE_TTS=OFF', '-DSHERPA_ONNX_ENABLE_GPU=OFF',
               '-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF']
    if args.deps_dir:
        for directory in sorted(args.deps_dir.resolve().glob('*-src')):
            if directory.name != 'onnxruntime-src':
                options.append('-DFETCHCONTENT_SOURCE_DIR_' + directory.name[:-4].upper() + '=' + str(directory))
    env = dict(os.environ, SHERPA_ONNX_CMAKE_ARGS=shlex.join(options),
               SHERPA_ONNX_MAKE_ARGS='-j' + str(args.jobs),
               SHERPA_ONNXRUNTIME_INCLUDE_DIR=str(args.ort_include.resolve()),
               SHERPA_ONNXRUNTIME_LIB_DIR=str(args.ort_lib.resolve()))
    env.pop('SHERPA_ONNX_SPLIT_PYTHON_PACKAGE', None)
    subprocess.run([sys.executable, 'setup.py', 'bdist_wheel', '--dist-dir',
                    str(args.wheel_dir.resolve())], cwd=source, env=env, check=True)


if __name__ == '__main__':
    main()
