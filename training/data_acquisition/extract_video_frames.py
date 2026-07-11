"""Extract grouped, hashed frames from gameplay video without split leakage."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from provenance import ProvenanceRecord, append_record, imported_now

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract controlled-interval and scene-change gameplay frames")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=0.75)
    parser.add_argument("--scene-change", action="store_true")
    parser.add_argument("--scene-threshold", type=float, default=24.0)
    parser.add_argument("--min-width", type=int, default=1280)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-type", choices=("self_captured", "permissioned"), default="self_captured")
    parser.add_argument("--license", default="self-captured")
    args = parser.parse_args()
    if not args.input.is_dir() or args.interval <= 0:
        parser.error("--input must exist and --interval must be positive")
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("opencv-python is required for frame extraction") from exc
    args.output.mkdir(parents=True, exist_ok=True)
    extracted = 0
    for video in sorted(path for path in args.input.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES):
        session = hashlib.sha256(str(video.resolve()).encode()).hexdigest()[:16]
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            print(f"warning: could not open {video}")
            continue
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        interval_frames = max(1, round(fps * args.interval))
        previous_small = None
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36))
            scene_delta = float(cv2.absdiff(previous_small, small).mean()) if previous_small is not None else 0.0
            should_save = frame_index % interval_frames == 0 or (args.scene_change and scene_delta >= args.scene_threshold)
            previous_small = small
            if should_save and width >= args.min_width:
                filename = f"{video.stem}-{session}-{frame_index:08d}.jpg"
                destination = args.output / filename
                if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"Failed to write {destination}")
                digest = hashlib.sha256(destination.read_bytes()).hexdigest()
                append_record(args.manifest, ProvenanceRecord(
                    image_id=digest[:24], local_filename=str(destination.resolve()), source_type=args.source_type,
                    source_url=None, source_page=None, creator=None, license=args.license, license_url=None,
                    permission_evidence=None, attribution_text=None, imported_at_utc=imported_now(), sha256=digest,
                    perceptual_hash=None, width=width, height=height, game_category=None, session_id=session,
                    accepted=True, rejection_reason=None, dataset_split=None,
                ))
                extracted += 1
            frame_index += 1
        capture.release()
    print(f"Extracted {extracted} grouped frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
