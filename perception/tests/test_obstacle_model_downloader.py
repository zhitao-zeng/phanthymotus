import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_ROOT))

from utils.obstacle_model_downloader import download_manifest, load_manifest


class ObstacleModelDownloaderTest(unittest.TestCase):
    def _write_manifest(self, root, source, destination, *, sha256=None, size=None):
        data = source.read_bytes()
        manifest = root / "artifacts.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifacts": [
                        {
                            "name": "test-model",
                            "url": source.as_uri(),
                            "size_bytes": len(data) if size is None else size,
                            "sha256": hashlib.sha256(data).hexdigest()
                            if sha256 is None
                            else sha256,
                            "destination": str(destination),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_download_manifest_verifies_and_installs_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"verified-model-bytes")
            destination = root / "models" / "model.bin"
            manifest = self._write_manifest(root, source, destination)

            download_manifest(manifest, retries=1)

            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertEqual(list(destination.parent.glob(".model.bin.*")), [])

    def test_checksum_failure_does_not_replace_existing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"untrusted-new-bytes")
            destination = root / "models" / "model.bin"
            destination.parent.mkdir()
            destination.write_bytes(b"existing-model")
            manifest = self._write_manifest(
                root,
                source,
                destination,
                sha256="0" * 64,
            )

            with self.assertRaisesRegex(RuntimeError, "test-model"):
                download_manifest(manifest, retries=1)

            self.assertEqual(destination.read_bytes(), b"existing-model")

    def test_size_failure_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"model")
            destination = root / "models" / "model.bin"
            manifest = self._write_manifest(
                root,
                source,
                destination,
                size=4,
            )

            with self.assertRaisesRegex(RuntimeError, "test-model"):
                download_manifest(manifest, retries=1)
            self.assertFalse(destination.exists())

    def test_manifest_rejects_relative_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"model")
            manifest = self._write_manifest(
                root,
                source,
                Path("relative/model.bin"),
            )

            with self.assertRaisesRegex(ValueError, "absolute"):
                load_manifest(manifest)

    def test_destination_root_maps_container_paths_for_local_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"model")
            manifest = self._write_manifest(
                root,
                source,
                Path("/models/obstacle/model.bin"),
            )

            download_manifest(
                manifest,
                retries=1,
                destination_root=root / "local-root",
            )

            self.assertEqual(
                (root / "local-root/models/obstacle/model.bin").read_bytes(),
                b"model",
            )


if __name__ == "__main__":
    unittest.main()
