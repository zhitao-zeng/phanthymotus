import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class OfflineASRHotwordsTest(unittest.TestCase):
    def test_prepares_only_configured_hotwords_as_char_bpe(self):
        from plugins.asr_offline import _prepare_hotwords_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "hotwords.txt"
            source.write_text("# comment\n甲 乙\n丙丁\n\n", encoding="utf-8")

            output, modeling_unit = _prepare_hotwords_file(source, root / "out")
            lines = output.read_text(encoding="utf-8").splitlines()

            self.assertEqual(modeling_unit, "bpe")
            self.assertIn("甲 乙 :2.0", lines)
            self.assertEqual(lines.count("丙 丁 :2.0"), 1)
            self.assertFalse(any("戊 己" in line for line in lines))
            self.assertFalse(any("comment" in line for line in lines))

            rescored_output, _ = _prepare_hotwords_file(
                source, root / "rescored", hotwords_score=2.5
            )
            rescored_lines = rescored_output.read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertIn("甲 乙 :2.5", rescored_lines)
            self.assertTrue(all(line.endswith(":2.5") for line in rescored_lines))


if __name__ == "__main__":
    unittest.main()
