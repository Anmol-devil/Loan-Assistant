import os
import subprocess
import sys
import time
from typing import Any


class LegacyRapidOCRAdapter:
    def __init__(self):
        self.engine = None

    def _get_engine(self):
        if self.engine is None:
            ocr_dir = "/tmp/ocr_libs"
            if os.path.isdir(ocr_dir) and ocr_dir not in sys.path:
                sys.path.insert(0, ocr_dir)
            try:
                from rapidocr import RapidOCR
            except ImportError:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                ocr_req = os.path.join(base_dir, "requirements-ocr.txt")
                os.makedirs(ocr_dir, exist_ok=True)
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "--no-cache-dir", "--target", ocr_dir, "-r", ocr_req,
                ])
                if ocr_dir not in sys.path:
                    sys.path.insert(0, ocr_dir)
                from rapidocr import RapidOCR
            self.engine = RapidOCR()
        return self.engine

    def __call__(self, image: Any):
        start_time = time.perf_counter()
        result = self._get_engine()(image)

        if result is None:
            return [], time.perf_counter() - start_time

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


ocr_engine = LegacyRapidOCRAdapter()