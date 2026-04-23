from typing import Tuple

import numpy as np
from PIL import Image


def _foreground_bbox(rgb: Image.Image, white_threshold: int = 245) -> Tuple[int, int, int, int]:
    array = np.asarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an RGB image.")

    foreground = np.any(array < white_threshold, axis=2)
    rows = np.flatnonzero(foreground.any(axis=1))
    cols = np.flatnonzero(foreground.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return 0, 0, rgb.width, rgb.height

    y1, y2 = rows[0], rows[-1]
    x1, x2 = cols[0], cols[-1]
    return int(x1), int(y1), int(x2) + 1, int(y2) + 1


def auto_crop_to_square(image: Image.Image, white_threshold: int = 245, margin_ratio: float = 0.12) -> Image.Image:
    rgb = image.convert("RGB")
    x1, y1, x2, y2 = _foreground_bbox(rgb, white_threshold=white_threshold)

    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)
    margin = int(round(max(crop_w, crop_h) * margin_ratio))

    left = max(0, x1 - margin)
    top = max(0, y1 - margin)
    right = min(rgb.width, x2 + margin)
    bottom = min(rgb.height, y2 + margin)
    cropped = rgb.crop((left, top, right, bottom))

    side = max(cropped.width, cropped.height)
    canvas = Image.new("RGB", (side, side), color=(255, 255, 255))
    offset_x = (side - cropped.width) // 2
    offset_y = (side - cropped.height) // 2
    canvas.paste(cropped, (offset_x, offset_y))
    return canvas


def reduce_white_background_shadows(
    image: Image.Image,
    brightness_floor: int = 150,
    neutral_threshold: int = 20,
    protect_white_threshold: int = 235,
    strength: float = 0.85,
) -> Image.Image:
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32)

    channel_max = array.max(axis=2)
    channel_min = array.min(axis=2)
    channel_spread = channel_max - channel_min
    brightness = array.mean(axis=2)

    bright_neutral_mask = (brightness >= float(brightness_floor)) & (channel_spread <= float(neutral_threshold))
    protected_foreground_mask = np.any(array < float(protect_white_threshold), axis=2)
    background_shadow_mask = bright_neutral_mask & ~protected_foreground_mask

    if not np.any(background_shadow_mask):
        return rgb

    normalized_brightness = np.clip(
        (brightness - float(brightness_floor)) / max(1.0, 255.0 - float(brightness_floor)),
        0.0,
        1.0,
    )
    blend = (1.0 - normalized_brightness) * float(strength)
    blend = np.clip(blend, 0.0, 1.0)

    output = array.copy()
    output[background_shadow_mask] = (
        output[background_shadow_mask] * (1.0 - blend[background_shadow_mask, None])
        + 255.0 * blend[background_shadow_mask, None]
    )
    output = np.clip(output, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(output, mode="RGB")
