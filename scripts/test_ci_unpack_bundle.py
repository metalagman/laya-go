"""Safety contracts for the build-time protected CI archive extractor."""

import importlib.util
import io
import pathlib
import tarfile
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("ci-unpack-bundle.py")
SPEC = importlib.util.spec_from_file_location("ci_unpack_bundle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXTRACTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXTRACTOR)


class BundleArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.archive = self.root / "bundle.tar.gz"

    def write_archive(self, members):
        with tarfile.open(self.archive, "w:gz") as output:
            for name, kind, data in members:
                item = tarfile.TarInfo(name)
                if kind == "file":
                    item.size = len(data)
                    output.addfile(item, io.BytesIO(data))
                elif kind == "symlink":
                    item.type = tarfile.SYMTYPE
                    item.linkname = data
                    output.addfile(item)

    def test_regular_root_layout(self):
        self.write_archive([("manifest.json", "file", b"{}")])
        destination = self.root / "bundle"
        EXTRACTOR.unpack(self.archive, destination)
        self.assertEqual((destination / "manifest.json").read_bytes(), b"{}")

    def test_rejects_parent_traversal(self):
        self.write_archive([("../escape", "file", b"secret")])
        with self.assertRaisesRegex(ValueError, "invalid or duplicate path"):
            EXTRACTOR.unpack(self.archive, self.root / "bundle")
        self.assertFalse((self.root / "escape").exists())

    def test_rejects_symlink(self):
        self.write_archive([("manifest.json", "file", b"{}"), ("model.onnx", "symlink", "../escape")])
        with self.assertRaisesRegex(ValueError, "link or special"):
            EXTRACTOR.unpack(self.archive, self.root / "bundle")

    def test_rejects_duplicate_normalized_path(self):
        self.write_archive([("manifest.json", "file", b"{}"), ("./manifest.json", "file", b"{}")])
        with self.assertRaisesRegex(ValueError, "invalid or duplicate path"):
            EXTRACTOR.unpack(self.archive, self.root / "bundle")


if __name__ == "__main__":
    unittest.main()
