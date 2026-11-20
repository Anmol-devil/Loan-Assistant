import time
from typing import Any


class LegacyRapidOCRAdapter:
    """
    Lazy-loads RapidOCR on first use and adapts its output
    to the (box, text, confidence) tuple format.
    """

    def __init__(self):
        self.engine = None

    def _get_engine(self):
        if self.engine is None:
            try:
                from rapidocr import RapidOCR
                self.engine = RapidOCR()
            except ImportError:
                from rapidocr_onnxruntime import RapidOCR
                self.engine = RapidOCR()
        return self.engine

    def __call__(self, image: Any):
        start_time = time.perf_counter()
        result = self._get_engine()(image)

        if result is None:
            return [], time.perf_counter() - start_time

        # Support rapidocr-onnxruntime tuple output: (rows, elapsed)
        if isinstance(result, tuple) and len(result) == 2:
            rows_raw, elapsed = result
            if not rows_raw:
                return [], float(elapsed or (time.perf_counter() - start_time))
            rows = [
                (
                    box.tolist() if hasattr(box, "tolist") else list(box),
                    str(text),
                    float(score),
                )
                for box, text, score in rows_raw
                if text is not None and str(text).strip()
            ]
            return rows, float(elapsed or (time.perf_counter() - start_time))

        # Support rapidocr >= 2.x / 3.x object output
        boxes = getattr(result, "boxes", None)
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)

        if boxes is None or texts is None:
            raise RuntimeError("RapidOCR returned an unexpected output format.")

        if scores is None:
            scores = [0.0] * len(texts)

        if not (len(boxes) == len(texts) == len(scores)):
            raise RuntimeError("RapidOCR boxes, text and confidence counts do not match.")

        rows = []
        for box, text, score in zip(boxes, texts, scores):
            if text is None or not str(text).strip():
                continue
            box = box.tolist() if hasattr(box, "tolist") else list(box)
            rows.append((box, str(text), float(score)))

        elapsed = getattr(result, "elapse", None)
        if not isinstance(elapsed, (int, float)):
            elapsed = time.perf_counter() - start_time

        return rows, float(elapsed)


# One shared lazy-loaded OCR engine for the application
ocr_engine = LegacyRapidOCRAdapter()