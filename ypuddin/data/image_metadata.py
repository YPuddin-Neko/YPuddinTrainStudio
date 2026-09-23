"""Pillow transparency shared by inspection, training and mask editing."""

from PIL import Image


def has_alpha(image: Image.Image) -> bool:
    # PNG color keys also apply to RGB and grayscale images, not only palettes.
    return any(band.upper() == "A" for band in image.getbands()) or "transparency" in image.info


def has_color_key(image: Image.Image) -> bool:
    return "transparency" in image.info and (
        image.mode in ("RGB", "L", "I") or image.mode.startswith("I;16")
    )


def alpha_channel(image: Image.Image) -> Image.Image | None:
    if has_color_key(image) and (image.mode == "I" or image.mode.startswith("I;16")):
        import numpy as np

        key = image.info["transparency"]
        if not isinstance(key, int) or not 0 <= key <= 65535:
            raise ValueError("invalid 16-bit PNG transparency color key")
        # RGBA conversion clamps grayscale samples to 8 bits and loses tRNS.
        # Compare the original samples before creating the 8-bit binary alpha.
        return Image.fromarray((np.asarray(image) != key).astype(np.uint8) * 255)
    return image.convert("RGBA").getchannel("A") if has_alpha(image) else None
