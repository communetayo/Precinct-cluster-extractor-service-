# FastAPI router (not a standalone app) - mounted onto the existing
# Render service alongside main.py, with two lines added there:
#   from er_ocr_endpoint import router as ocr_router
#   app.include_router(ocr_router)
#
# Uses easyocr specifically, not pytesseract - Render's filesystem is
# read-only at runtime, so apt-get (needed to install the tesseract
# system binary) simply fails there. easyocr installs entirely through
# pip, no system dependency at all, which is why it was chosen over
# the more commonly-used alternative.
#
# requirements.txt needs: fastapi, uvicorn, easyocr, opencv-python-headless,
# numpy, requests (pydantic comes in transitively via fastapi)

import io
import re
from difflib import get_close_matches

import cv2
import numpy as np
import requests
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# Loaded once, at import time, rather than per-request - easyocr's
# own model loading is the slow part, and re-doing it on every single
# call would make this endpoint impractically slow.
import easyocr
_reader = easyocr.Reader(["en"], gpu=False)


class OcrRequest(BaseModel):
    image_url: str
    candidate_names: list[str]


def _normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9 ]", "", text.upper()).strip()


@router.post("/extract-er-ocr")
def extract_er_ocr(req: OcrRequest):
    try:
        resp = requests.get(req.image_url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        return {"status": "failed", "error": f"Could not download the photo: {e}"}

    try:
        image_array = np.frombuffer(resp.content, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            return {"status": "failed", "error": "Could not decode the image."}

        # readtext returns a list of (bounding_box, text, confidence) -
        # bounding_box is 4 corner points, used below to figure out
        # which detected number sits on the same physical line as
        # which detected candidate name.
        results = _reader.readtext(image)
    except Exception as e:
        return {"status": "failed", "error": f"OCR itself failed: {e}"}

    # Splits every detected text block into "looks like a name" versus
    # "looks like a number" - a real ER has candidate names on the
    # left and their vote count on the right, on the same row.
    name_blocks = []
    number_blocks = []
    for bbox, text, confidence in results:
        cleaned = _normalize(text)
        if not cleaned:
            continue
        y_center = sum(point[1] for point in bbox) / 4
        x_center = sum(point[0] for point in bbox) / 4
        if re.fullmatch(r"\d+", cleaned):
            number_blocks.append({"value": int(cleaned), "x": x_center, "y": y_center})
        else:
            name_blocks.append({"text": cleaned, "x": x_center, "y": y_center})

    extracted_votes: dict[str, int] = {}
    normalized_candidates = {_normalize(name): name for name in req.candidate_names}

    for block in name_blocks:
        # Only counts as a real match above a genuine similarity
        # threshold - a garbled OCR read that doesn't clearly match
        # any real candidate is skipped entirely rather than guessed
        # at, since a wrong match here would misattribute real votes.
        matches = get_close_matches(
            block["text"], normalized_candidates.keys(), n=1, cutoff=0.75
        )
        if not matches:
            continue
        real_name = normalized_candidates[matches[0]]
        if real_name in extracted_votes:
            continue  # Already matched this candidate from an earlier block.

        # The nearest number block on roughly the same horizontal row,
        # and to the right of the name (since that's the real layout
        # of an ER - name on the left, vote count on the right).
        same_row_numbers = [
            n for n in number_blocks
            if abs(n["y"] - block["y"]) < 20 and n["x"] > block["x"]
        ]
        if same_row_numbers:
            closest = min(same_row_numbers, key=lambda n: n["x"] - block["x"])
            extracted_votes[real_name] = closest["value"]

    return {"status": "completed", "extracted_votes": extracted_votes}
