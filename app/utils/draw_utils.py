import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.core.config import settings

_font_cache = {}


def _get_font_candidates():
    # Comma-separated env/config keeps settings serialization simple.
    return [p.strip() for p in settings.KOREAN_FONT_PATHS.split(",") if p.strip()]


def get_korean_font(font_size=24):
    if font_size in _font_cache:
        return _font_cache[font_size]

    font = None
    for font_path in _get_font_candidates():
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except Exception:
            continue

    if font is None:
        font = ImageFont.load_default()

    _font_cache[font_size] = font
    return font


def put_korean_text(frame, text, position, font_size=24, color=(255, 255, 255)):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = get_korean_font(font_size)
        rgb_color = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=rgb_color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return frame
