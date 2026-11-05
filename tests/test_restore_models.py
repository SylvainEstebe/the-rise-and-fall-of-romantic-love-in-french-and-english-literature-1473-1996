import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from script.restore_models import restore_archive


class RestoreTests(unittest.TestCase):
    def test_restore_and_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "results/models/example.nc"
            data = b"test posterior bytes"
            manifest = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}
            archive = root / "models.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr(name, data)
            for _ in range(2):
                restore_archive(archive, manifest, root / "restored")
            self.assertEqual((root / "restored" / name).read_bytes(), data)

    def test_bad_checksum_is_not_installed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "results/models/example.nc"
            archive = root / "models.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr(name, b"bad")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                restore_archive(archive, {name: {"bytes": 3, "sha256": "0" * 64}}, root / "restored")
            self.assertFalse((root / "restored" / name).exists())

    def test_parent_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "models.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../outside.nc", b"bad")
            with self.assertRaisesRegex(ValueError, "Unsafe archive path"):
                restore_archive(archive, {}, root / "restored")


if __name__ == "__main__":
    unittest.main()
