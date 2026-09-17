"""Install an immutable CPU prefix-LM wheel in an isolated image directory."""
import argparse
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

from model_downloader import ensure_verified_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', required=True)
    ap.add_argument('--filename', required=True)
    ap.add_argument('--size', type=int, required=True)
    ap.add_argument('--sha256', required=True)
    args = ap.parse_args()
    if platform.machine() != 'aarch64':
        raise RuntimeError('This prefix runtime artifact requires ARM64')
    with tempfile.TemporaryDirectory(prefix='asr-prefix-runtime-') as directory:
        ensure_verified_bundle('asr_prefix_runtime', directory, args.base_url,
                               {args.filename: {'size': args.size, 'sha256': args.sha256}})
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                        '--target', '/opt/asr-prefix-runtime',
                        str(Path(directory) / args.filename)], check=True)
    subprocess.run([sys.executable, '-c',
                    "import sys; sys.path.insert(0, '/opt/asr-prefix-runtime'); "
                    "import sherpa_onnx; assert sherpa_onnx.XASR_PREFIX_LM_VERSION == 1; "
                    "print('CPU prefix LM runtime', sherpa_onnx.__version__)"], check=True)


if __name__ == '__main__':
    main()
