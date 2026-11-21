_OCR_INSTANCE = None

def get_ocr_engine():
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        from rapidocr import RapidOCR
        _OCR_INSTANCE = RapidOCR()
    return _OCR_INSTANCE

def run_ocr_normalized(image_path: str):
    """Returns legacy list of (box, text, conf) across rapidocr v1, v2, and v3."""
    engine = get_ocr_engine()
    raw = engine(image_path)
    if isinstance(raw, tuple):
        return raw[0] or []
    boxes = getattr(raw, "boxes", None)
    txts = getattr(raw, "txts", None)
    scores = getattr(raw, "scores", None)
    if boxes is None or txts is None:
        return []
    if scores is None:
        scores = [1.0] * len(txts)
    return list(zip(boxes, txts, scores))