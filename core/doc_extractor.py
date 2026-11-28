import os
import re
import time
import zipfile
from typing import Any, Dict, Optional

import pymupdf
from PIL import Image

from core.doc_detector import detect_document_type
from multiformated_reconciliation import analyze_document
from module_aggregator import build_module1_output
from universal_doc_extractor import extract_universal_document


def check_password_protection(file_path: str) -> Dict[str, Any]:
    """
    Detects whether a PDF or Office document requires a password to open.
    PyMuPDF automatically tests an empty password on open, so `doc.needs_pass`
    is True ONLY when a real user password is required (not permissions-only locks).
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        try:
            with pymupdf.open(file_path) as doc:
                if doc.needs_pass:
                    return {
                        "is_protected": True,
                        "reason": "This PDF is password-protected and cannot be read without unlocking it first.",
                    }
        except Exception as e:
            err_low = str(e).lower()
            if any(k in err_low for k in ("encrypt", "password", "decrypt")):
                return {
                    "is_protected": True,
                    "reason": "This PDF is encrypted/password-protected and cannot be opened.",
                }

    elif ext in (".docx", ".xlsx"):
        try:
            with open(file_path, "rb") as f:
                header = f.read(8)
            if header == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" and not zipfile.is_zipfile(file_path):
                return {
                    "is_protected": True,
                    "reason": f"This {ext.upper()} file is password-protected and cannot be read.",
                }
        except Exception:
            pass

    return {"is_protected": False, "reason": None}


def _probe_file_sample(file_path: str) -> Dict[str, Any]:
    """Fast lightweight probe of Page 1–2 text and page geometry (for full-page vs card detection)."""
    ext = os.path.splitext(file_path)[1].lower()
    page_count = 1
    aspect_ratio: Optional[float] = None
    sample_text = ""

    if ext == ".pdf":
        with pymupdf.open(file_path) as doc:
            page_count = len(doc)
            if page_count > 0:
                rect = doc[0].rect
                aspect_ratio = round(float(rect.height) / max(1.0, float(rect.width)), 3)
                parts = [doc[i].get_text("text") or "" for i in range(min(2, page_count))]
                sample_text = "\n".join(parts).strip()
    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        with Image.open(file_path) as im:
            w, h = im.size
            aspect_ratio = round(float(h) / max(1.0, float(w)), 3)
    return {
        "ext": ext,
        "page_count": page_count,
        "aspect_ratio": aspect_ratio,
        "sample_text": sample_text,
    }


def _enrich_full_epan_or_eaadhaar(doc_code: str, variant: str, full_text: str, fin_rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backfills fields that sit in the upper letter section of Full A4 e-PAN / e-Aadhaar PDFs
    without altering anything that already extracted cleanly from card scans.
    """
    if variant != "FULL_E_DOCUMENT" or not full_text:
        return fin_rec

    fin_rec = dict(fin_rec or {})
    if doc_code == "PAN_CARD":
        if not fin_rec.get("legal_name") or fin_rec.get("legal_name") == "Unknown Holder":
            m_n = re.search(r"(?:Name|नाम)\s*[:/\-]\s*([A-Z][A-Za-z\s\.]{3,45})", full_text)
            if m_n:
                fin_rec["legal_name"] = re.sub(r"\s+", " ", m_n.group(1)).strip().title()
        if not fin_rec.get("father_name"):
            m_f = re.search(r"Father'?s\s+Name\s*[:/\-]?\s*\n?\s*([A-Z][A-Za-z\s\.]{3,45})", full_text, re.I)
            if m_f:
                fin_rec["father_name"] = re.sub(r"\s+", " ", m_f.group(1)).strip().title()
        if not fin_rec.get("date_of_birth"):
            m_d = re.search(r"\b(0?[1-9]|[12]\d|3[01])[-/\.](0?[1-9]|1[0-2])[-/\.](\d{4})\b", full_text)
            if m_d:
                fin_rec["date_of_birth"] = m_d.group(0)

    elif doc_code == "AADHAAR_CARD":
        # Strict privacy: always keep Aadhaar number redacted in unified output
        fin_rec["uid_masked"] = "[Aadhaar Redacted]"
        if not fin_rec.get("pin_code"):
            m_pin = re.search(r"\b([1-9]\d{5})\b", full_text)
            if m_pin:
                fin_rec["pin_code"] = m_pin.group(1)

    return fin_rec


def detect_and_extract_document(
    file_path: str,
    original_filename: Optional[str] = None,
    use_llm: bool = True,
    use_glm_ocr: bool = False,
    tenant_id: str = "default_lender",
    include_universal_markdown: bool = False,
) -> Dict[str, Any]:
    """
    Unified Detection + Extraction entrypoint safe for:
      - All 15+ Indian Bank Statements (delegates to BANK_PARSER_HYBRID_V2 + legacy recovery)
      - Full A4 e-PAN / e-Aadhaar PDFs AND cropped mobile/PVC card scans
      - GST (REG-06, GSTR-1, GSTR-3B, GSTR-2B), Udyam, ITR, Form 3CB/3CD, Balance Sheet, Salary Slip
      - Spreadsheets (.xlsx/.csv) & Unknown documents (DocVortex + RapidOCR / GLM-OCR)
    """
    t0 = time.perf_counter()
    eff_name = original_filename or os.path.basename(file_path)
    
    lock_info = check_password_protection(file_path)
    if lock_info["is_protected"]:
        return {
            "status": "PASSWORD_PROTECTED",
            "is_password_protected": True,
            "message": lock_info["reason"],
            "filename": eff_name,
            "detection": {
                "doc_code": "PASSWORD_PROTECTED",
                "doc_category": "Password-Protected Document",
                "layout_variant": "LOCKED_FILE",
                "confidence": 1.0,
            },
            "raw_report": None,
            "module1_contract": None,
            "universal_extraction": None,
            "processing_time_sec": round(time.perf_counter() - t0, 2),
        }
        
    probe = _probe_file_sample(file_path)
    ext = probe["ext"]

    # Spreadsheets go directly to universal extractor
    if ext in (".xlsx", ".xls", ".csv"):
        uni_res = extract_universal_document(file_path, use_glm_ocr=False)
        return {
            "filename": eff_name,
            "detection": {
                "doc_code": "SPREADSHEET_DATA",
                "doc_category": "Financial Spreadsheet",
                "layout_variant": "TABULAR_WORKBOOK",
                "confidence": 1.0,
            },
            "module1_contract": None,
            "universal_extraction": uni_res,
            "processing_time_sec": round(time.perf_counter() - t0, 2),
        }

    # Run proven Module 1 pipeline (handles 15+ banks, OCR scans, PDF/pixel forensics, KYC, GST, 3CD)
    raw_report = analyze_document(file_path, original_filename=eff_name, use_llm=use_llm)
    raw_report["original_filename"] = eff_name
    m1_contract = build_module1_output(file_path, raw_report, tenant_id=tenant_id)

    fin_rec = raw_report.get("financial_reconciliation") or {}
    id_card = m1_contract.get("identity_card") or {}

    # Detect layout variant (Full e-Document vs Card Scan vs Multi-Page Ledger)
    sample_txt = probe["sample_text"]
    if not sample_txt and fin_rec.get("transactions"):
        sample_txt = " ".join((t.get("description") or "") for t in fin_rec["transactions"][:10])

    det = detect_document_type(
        sample_txt,
        filename=eff_name,
        file_ext=ext,
        page_count=probe["page_count"],
        page_aspect_ratio=probe["aspect_ratio"],
    )
    # Trust Module 1's verified doc_code whenever it successfully classified a scanned image
    verified_code = fin_rec.get("document_type") or id_card.get("doc_code") or det["doc_code"]
    if verified_code != "UNKNOWN_DOCUMENT":
        det["doc_code"] = verified_code

    # Apply full e-PAN / e-Aadhaar field backfill without touching card or bank results
    if det["doc_code"] in ("PAN_CARD", "AADHAAR_CARD") and det["layout_variant"] == "FULL_E_DOCUMENT":
        fin_rec = _enrich_full_epan_or_eaadhaar(det["doc_code"], det["layout_variant"], sample_txt, fin_rec)
        raw_report["financial_reconciliation"] = fin_rec
        m1_contract["extracted_financials"] = fin_rec

    uni_res = None
    if include_universal_markdown or det["doc_code"] == "UNKNOWN_DOCUMENT":
        uni_res = extract_universal_document(file_path, use_glm_ocr=use_glm_ocr)

    return {
        "filename": eff_name,
        "detection": det,
        "raw_report": raw_report,
        "module1_contract": m1_contract,
        "universal_extraction": uni_res,
        "processing_time_sec": round(time.perf_counter() - t0, 2),
    }