import time
from typing import Any

from rapidocr import RapidOCR


class LegacyRapidOCRAdapter:
    """
    Adapts the new RapidOCR output to the old
    rapidocr-onnxruntime return format.

    Expected output:
        rows, elapsed = engine(image)

    Each row:
        (box, text, confidence)
    """

    def __init__(self):
        self.engine = RapidOCR()

    def __call__(self, image: Any):
        start_time = time.perf_counter()

        result = self.engine(image)

        if result is None:
            return [], time.perf_counter() - start_time

        boxes = getattr(result, "boxes", None)
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)

        if boxes is None or texts is None:
            raise RuntimeError(
                "RapidOCR returned an unexpected output format."
            )

        if scores is None:
            scores = [0.0] * len(texts)

        if not (len(boxes) == len(texts) == len(scores)):
            raise RuntimeError(
                "RapidOCR boxes, text and confidence counts do not match."
            )

        rows = []

        for box, text, score in zip(boxes, texts, scores):

            if text is None or not str(text).strip():
                continue

            if hasattr(box, "tolist"):
                box = box.tolist()
            else:
                box = list(box)

            rows.append((
                box,
                str(text),
                float(score)
            ))

        elapsed = getattr(result, "elapse", None)

        if not isinstance(elapsed, (int, float)):
            elapsed = time.perf_counter() - start_time

        return rows, float(elapsed)


# One shared OCR engine for the application
ocr_engine = LegacyRapidOCRAdapter()