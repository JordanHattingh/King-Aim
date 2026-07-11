"""Bounded HTTP image download and fingerprint helpers."""

from __future__ import annotations

import hashlib
import io
import re
import urllib.request
from pathlib import Path

USER_AGENT = "KingAim-DatasetCollector/2.5 (personal accessibility dataset)"


def safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return cleaned[:120] or fallback


def image_fingerprint(data: bytes) -> tuple[str, str, int, int, str]:
    from PIL import Image

    digest = hashlib.sha256(data).hexdigest()
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        extension = ".jpg" if image.format in {"JPEG", "MPO"} else f".{(image.format or 'bin').lower()}"
        grayscale = image.convert("L").resize((16, 16))
        flattened = getattr(grayscale, "get_flattened_data", None)
        pixels = list(flattened() if flattened else grayscale.getdata())
    mean = sum(pixels) / len(pixels)
    perceptual = f"{sum((1 << index) for index, value in enumerate(pixels) if value >= mean):064x}"
    return digest, perceptual, width, height, extension


def download_image(url: str, max_bytes: int = 25 * 1024 * 1024, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "image/*"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get_content_type()
        if not content_type.startswith("image/"):
            raise ValueError(f"Expected image content, got {content_type}")
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > max_bytes:
            raise ValueError(f"Image exceeds {max_bytes} bytes")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"Image exceeds {max_bytes} bytes")
    image_fingerprint(data)
    return data


def store_image(output: Path, stem: str, data: bytes, min_width: int, min_height: int) -> tuple[Path, str, str, int, int]:
    digest, perceptual, width, height, extension = image_fingerprint(data)
    if width < min_width or height < min_height:
        raise ValueError(f"Image {width}x{height} below minimum {min_width}x{min_height}")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{safe_name(stem, digest[:16])}-{digest[:12]}{extension}"
    destination.write_bytes(data)
    return destination, digest, perceptual, width, height
