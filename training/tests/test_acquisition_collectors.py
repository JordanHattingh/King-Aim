from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
ACQUISITION = TRAINING / "data_acquisition"
sys.path.insert(0, str(ACQUISITION))

from acquire_wikimedia import metadata_value
from download_utils import image_fingerprint, safe_name, store_image
from provenance import ProvenanceRecord, append_record, load_records


class AcquisitionCollectorTests(unittest.TestCase):
    def test_wikimedia_metadata_strips_html(self) -> None:
        metadata = {"Artist": {"value": "<a href='x'>Jordan</a> &amp; Team"}}
        self.assertEqual("Jordan & Team", " ".join(metadata_value(metadata, "Artist").split()))

    def test_safe_name_rejects_path_characters(self) -> None:
        self.assertEqual("File-player-one.jpg", safe_name("File: player/one.jpg", "fallback"))

    def test_image_fingerprint_and_storage(self) -> None:
        from PIL import Image
        import io

        buffer = io.BytesIO()
        Image.new("RGB", (16, 12), (255, 0, 0)).save(buffer, format="PNG")
        data = buffer.getvalue()
        digest, perceptual, width, height, extension = image_fingerprint(data)
        self.assertEqual((16, 12, ".png"), (width, height, extension))
        self.assertEqual(64, len(digest))
        self.assertEqual(64, len(perceptual))
        with tempfile.TemporaryDirectory() as directory:
            path, *_ = store_image(Path(directory), "sample", data, 8, 8)
            self.assertTrue(path.is_file())

    def test_provenance_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            record = ProvenanceRecord(
                "id", "image.png", "self_captured", None, None, "Jordan", "self-captured", None,
                None, None, "2026-07-11T00:00:00+00:00", "a" * 64, "b" * 64, 1920, 1080,
                "game", "session", False, "pending manual review", None,
            )
            append_record(path, record)
            self.assertEqual("session", load_records(path)[0]["session_id"])


if __name__ == "__main__":
    unittest.main()
