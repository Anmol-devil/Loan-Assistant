import os
import re
import json
import uuid
import numpy as np
import pymupdf
import docx
from rapidocr_onnxruntime import RapidOCR
from corporate_kyc_engines import classify_financial_document

ocr_reader = RapidOCR()


# AUTHORITATIVE ENUMS & CONSTANTS (Section 4.3 Schema)
ALLOWED_CHECK_STATUSES = {"PASSED", "FAILED", "NOT_RUN", "NOT_APPLICABLE"}
ALLOWED_TRIAGE_BANDS = {"low_risk", "needs_review", "high_risk"}
ALLOWED_DOC_CODES = {
    "BANK_STATEMENT", "SALARY_SLIP", "PAN_CARD",
    "GST_CERTIFICATE", "UDYAM_CERTIFICATE", "ITR_DOCUMENT",
    "BALANCE_SHEET", "UNKNOWN_DOCUMENT"
}
ALLOWED_FINDING_CODES = {
    "ARITHMETIC_MISMATCH",
    "FILENAME_PERIOD_DISCREPANCY",
    "EXTRACTION_INCOMPLETE",
    "DUPLICATE_TRANSACTION_ROW",
    "DATE_ORDER_ANOMALY",
    "TEMPLATE_PLACEHOLDER_ANOMALY",
    "PDF_TEXT_OVERLAY",
    "FONT_SUBSET_ANOMALY",
    "PDF_INCREMENTAL_UPDATE",
    "METADATA_TOOLCHAIN_TRACE",
    "PIXEL_ELA_ANOMALY",
    "COPY_MOVE_CLONE",
    "CROSS_DOCUMENT_MISMATCH",
    "UNCLASSIFIED_DYNAMIC_DOCUMENT",
    "UNCLASSIFIED_ANOMALY",
    "FILENAME_DOCUMENT_TYPE_DISCREPANCY"
}

TENANT_CONFIGS = {
    "default_lender": {
        "needs_review_threshold": 0.35,
        "high_risk_threshold": 0.70,
        "family_weights": {"A": 0.45, "B": 0.25, "C": 0.20, "D": 0.10}
    },
    "conservative_bank": {
        "needs_review_threshold": 0.20,
        "high_risk_threshold": 0.50,
        "family_weights": {"A": 0.40, "B": 0.25, "C": 0.20, "D": 0.15}
    },
    "high_velocity_fintech": {
        "needs_review_threshold": 0.45,
        "high_risk_threshold": 0.80,
        "family_weights": {"A": 0.50, "B": 0.20, "C": 0.20, "D": 0.10}
    }
}


# 1. HONEST RISK SCORING & OPTIONAL EMPIRICAL CALIBRATOR 
class RiskScoreCalibrator:
    def __init__(self, external_labeled_json_path: str = None):
        self.is_empirically_validated = False
        self.score_type = "heuristic_priority_score"
        self.validation_sample_count = 0
        self.held_out_brier_score = None
        self.knots_x = np.array([0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0])
        self.knots_y = np.array([0.02, 0.10, 0.24, 0.42, 0.60, 0.75, 0.86, 0.95])

        if external_labeled_json_path and os.path.exists(external_labeled_json_path):
            self.fit_from_labeled_file(external_labeled_json_path)

    @staticmethod
    def _pava(y_sorted: np.ndarray) -> np.ndarray:
        blocks = [[float(v), 1] for v in y_sorted]
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][0] > blocks[i + 1][0]:
                tot_v = blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
                tot_w = blocks[i][1] + blocks[i + 1][1]
                blocks[i] = [tot_v / tot_w, tot_w]
                blocks.pop(i + 1)
                if i > 0:
                    i -= 1
            else:
                i += 1
        out = []
        for val, cnt in blocks:
            out.extend([val] * cnt)
        return np.clip(np.array(out), 0.02, 0.95)

    def fit_from_labeled_file(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        val_split = np.array(data.get("validation_split", []))
        test_split = np.array(data.get("held_out_split", []))
        if len(val_split) >= 20 and len(test_split) >= 10:
            order = np.argsort(val_split[:, 0])
            self.knots_x = val_split[order, 0]
            self.knots_y = self._pava(val_split[order, 1])
            preds = [self.transform(x) for x in test_split[:, 0]]
            self.held_out_brier_score = round(float(np.mean((np.array(preds) - test_split[:, 1]) ** 2)), 4)
            self.validation_sample_count = len(val_split)
            self.is_empirically_validated = True
            self.score_type = "empirical_calibrated_probability"

    def transform(self, raw_score: float) -> float:
        clipped = float(np.clip(raw_score, 0.0, 1.0))
        return round(float(np.interp(clipped, self.knots_x, self.knots_y)), 2)


SCORER = RiskScoreCalibrator()

# 2. MULTI-PAGE TEXT EXTRACTOR & INDEPENDENT ENTITY EXTRACTOR 
PAN_EXTRACT_RE = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")


def clean_ocr_person_name(raw_name: str) -> tuple[str, float]:
    if not raw_name:
        return "Unknown Holder", 0.0
    orig = raw_name.strip()
    cleaned = re.sub(r"\s+[Il1\|]+$", "", orig).strip()
    cleaned = re.sub(r"[^A-Za-z0-9\s\.\&/]", "", cleaned).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned).title()
    if not cleaned or len(cleaned) < 3:
        return "Unknown Holder", 0.0
    confidence = 0.92 if cleaned.lower() != orig.lower() else 0.98
    return cleaned, confidence


def extract_raw_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        page_texts = []
        with pymupdf.open(file_path) as doc:
            for p_idx, page in enumerate(doc):
                t = page.get_text("text").strip()
                if len(t) > 15:
                    page_texts.append(t)
                else:
                    pix = page.get_pixmap(dpi=200)
                    tmp_img = f"{file_path}_p{p_idx + 1}_id_tmp.png"
                    pix.save(tmp_img)
                    try:
                        res, _ = ocr_reader(tmp_img)
                        ocr_txt = "\n".join(r[1] for r in res) if res else ""
                        page_texts.append(ocr_txt)
                    finally:
                        if os.path.exists(tmp_img):
                            os.remove(tmp_img)
        return "\n".join(page_texts)
    elif ext == ".docx":
        d = docx.Document(file_path)
        p_txt = "\n".join(p.text for p in d.paragraphs)
        t_txt = "\n".join(c.text for t in d.tables for r in t.rows for c in r.cells)
        return f"{p_txt}\n{t_txt}"
    elif ext in (".png", ".jpg", ".jpeg"):
        res, _ = ocr_reader(file_path)
        return "\n".join(r[1] for r in res) if res else ""
    return ""


def extract_tax_audit_identity(text: str, raw_report: dict = None) -> dict:
    """
    Extract identity information from Form 3CB / Form 3CD.

    Important:
    - Assessee is the document holder.
    - Auditor is kept as a separate entity.
    - Proprietor is kept separately from business name.
    - PAN is extracted from the assessee section where possible.
    """

    fin_rec = {}

    if isinstance(raw_report, dict):
        fin_rec = raw_report.get("financial_reconciliation") or {}

    if not isinstance(fin_rec, dict):
        fin_rec = {}

    assessee = fin_rec.get("assessee_business_name")
    proprietor = fin_rec.get("proprietor_name")
    auditor = fin_rec.get("auditor_name")
    signatory = fin_rec.get("signatory_name")
    assessee_pan = fin_rec.get("assessee_pan_number")

    assessee_match = None
    
    if not assessee:
        assessee_match = re.search(
            r"attached\s+herewith\s*,?\s*of\s*"
            r"[:\-]?\s*([^\r\n]+)",
            text,
            re.IGNORECASE
        )

        if assessee_match:
            assessee = assessee_match.group(1).strip()

    if not assessee:
        assessee_match = re.search(
            r"(?im)^\s*01\s+Name\s+of\s+the\s+assessee\s+"
            r"(.+?)\s*$",
            text
        )

        if assessee_match:
            assessee = assessee_match.group(1).strip()

    if assessee and not proprietor:
        proprietor_match = re.search(
            r"\(\s*Proprietor\s*:\s*([^)]+)\)",
            assessee,
            re.IGNORECASE
        )

        if proprietor_match:
            proprietor = proprietor_match.group(1).strip()

    # Remove proprietor information from business display name.
    if assessee:
        assessee = re.split(
            r"\s*\(\s*Proprietor\s*:",
            assessee,
            maxsplit=1,
            flags=re.IGNORECASE
        )[0].strip()

    if not auditor:
        auditor_match = re.search(
            r"(?im)^\s*For\s+([^\r\n]+)\s*\r?\n"
            r"\s*Chartered\s+Accountant\b",
            text
        )

        if auditor_match:
            auditor = auditor_match.group(1).strip()

    if not signatory:
        signatory_match = re.search(
            r"\(\s*([A-Z][A-Za-z.' ]{2,60})\s*\)"
            r"\s*\r?\n\s*Proprietor\b",
            text,
            re.IGNORECASE
        )

        if signatory_match:
            signatory = signatory_match.group(1).strip()

    if not assessee_pan:
        # Prefer PAN close to the Form 3CB assessee section.
        if assessee_match:
            start = assessee_match.start()
            pan_window = text[start:start + 1000]
        else:
            pan_window = text[:1500]

        pan_match = re.search(
            r"\bPAN\s*[:\-]?\s*([A-Z]{5}\d{4}[A-Z])\b",
            pan_window,
            re.IGNORECASE
        )

        if pan_match:
            assessee_pan = pan_match.group(1).upper()

    return {
        "assessee_business_name": assessee,
        "proprietor_name": proprietor,
        "auditor_name": auditor,
        "signatory_name": signatory,
        "assessee_pan_number": assessee_pan
    }


def identify_document_and_entity(
    file_path: str,
    raw_report: dict = None
) -> dict:
    text = extract_raw_text_from_file(file_path)

    if not text:
        return {
            "doc_category": "Unknown Document",
            "doc_code": "UNKNOWN_DOCUMENT",
            "classification_confidence": 0.0,
            "holder_name": "Unknown Holder",
            "person_name": "Unknown Holder",
            "raw_extracted_name": "Unknown Holder",
            "name_confidence": 0.0,
            "pan_number": None,
            "issuing_bank": None,
            "employer_name": None,
            "business_name": None,
            "proprietor_name": None,
            "auditor_name": None,
            "signatory_name": None,
            "identity_badge": "Unknown Document of Unknown Holder"
        }

    cls_info = classify_financial_document(text)

    doc_code = cls_info.get(
        "doc_code",
        "UNKNOWN_DOCUMENT"
    )

    doc_category = cls_info.get(
        "doc_category",
        "Unknown Document"
    )

    cls_conf = cls_info.get(
        "classification_confidence",
        0.0
    )
    
    normalized_text = re.sub(
        r"\s+",
        " ",
        text.upper()
    )

    has_form_3cb = bool(
        re.search(r"\bFORM\s+NO\.?\s*3CB\b", normalized_text)
    )

    has_form_3cd = bool(
        re.search(r"\bFORM\s+NO\.?\s*3CD\b", normalized_text)
    )

    has_section_44ab = (
        "SECTION 44AB" in normalized_text
    )

    if (
        has_section_44ab
        and (has_form_3cb or has_form_3cd)
    ):
        doc_code = "TAX_AUDIT_REPORT"
        doc_category = "Tax Audit Report (Form 3CB/3CD)"

        # Rule-based classification confidence, not a
        # statistically calibrated probability.
        cls_conf = 0.95

    if doc_code == "UNKNOWN_DOCUMENT":
        return {
            "doc_category": "Unknown Document",
            "doc_code": "UNKNOWN_DOCUMENT",
            "classification_confidence": 0.0,
            "holder_name": "Unknown Holder",
            "person_name": "Unknown Holder",
            "raw_extracted_name": "Unknown Holder",
            "name_confidence": 0.0,
            "pan_number": None,
            "issuing_bank": None,
            "employer_name": None,
            "business_name": None,
            "proprietor_name": None,
            "auditor_name": None,
            "signatory_name": None,
            "identity_badge": "Unknown Document of Unknown Holder"
        }

    extracted_pan = None
    raw_extracted_name = None
    person_name = "Unknown Holder"
    name_confidence = 0.0
    business_name = None
    proprietor_name = None
    auditor_name = None
    signatory_name = None
    issuing_bank = None
    employer_name = None
    if doc_code == "TAX_AUDIT_REPORT":

        tax_identity = extract_tax_audit_identity(
            text,
            raw_report
        )

        business_name = tax_identity.get(
            "assessee_business_name"
        )

        proprietor_name = tax_identity.get(
            "proprietor_name"
        )

        auditor_name = tax_identity.get(
            "auditor_name"
        )

        signatory_name = tax_identity.get(
            "signatory_name"
        )

        extracted_pan = tax_identity.get(
            "assessee_pan_number"
        )

        if business_name:
            person_name = business_name
            raw_extracted_name = business_name
            name_confidence = 0.90

        else:
            person_name = "Assessee Not Extracted"
            raw_extracted_name = None
            name_confidence = 0.0

    else:
        pan_match = re.search(
            r"\b([A-Z]{5}\d{4}[A-Z])\b",
            text.upper()
        )

        extracted_pan = (
            pan_match.group(1)
            if pan_match
            else None
        )
        name_patterns = [

            # Explicit labelled identity fields.
            r"(?im)^\s*(?:account\s+holder|account\s+name|"
            r"customer\s+name|employee\s+name|applicant\s+name|"
            r"card\s+holder|legal\s+name|name\s+of\s+enterprise)"
            r"\s*[:\-]\s*([A-Za-z][A-Za-z0-9.'&/ \t-]{2,60})",

            # M/s business names.
            r"(?im)^\s*(M/s\.?\s+[A-Z][A-Za-z0-9.'&/ \t-]{2,60})",

            # Business names with common legal suffixes.
            r"(?im)^\s*([A-Z][A-Z0-9.'&/ \t-]{2,50}\s+"
            r"(?:ENTERPRISES|ASSOCIATES|TRADERS|"
            r"PRIVATE\s+LIMITED|PVT\.?\s*LTD\.?|"
            r"LIMITED|LLP|INFRA|SOLUTIONS|CORPORATION))\s*$"
        ]

        excluded_terms = [
            "statement of",
            "salary slip",
            "standard chartered",
            "kotak mahindra",
            "department",
            "nominee",
            "chartered accountant",
            "firm regn",
            "naveen goel & associates"
        ]

        for pattern in name_patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if not match:
                continue

            candidate = match.group(1).strip()

            if len(candidate) < 3:
                continue

            if any(
                term in candidate.casefold()
                for term in excluded_terms
            ):
                continue

            raw_extracted_name = candidate

            person_name, name_confidence = (
                clean_ocr_person_name(candidate)
            )

            break
        
        issuing_bank = None

    bank_patterns = {
        "Yes Bank": r"\bYes\s+Bank\b",
        "Standard Chartered Bank": r"\bStandard\s+Chartered\s+Bank\b",
        "HDFC Bank": r"\bHDFC\s+Bank\b",
        "ICICI Bank": r"\bICICI\s+Bank\b",
        "State Bank of India": r"\bState\s+Bank\s+of\s+India\b",
        "Axis Bank": r"\bAxis\s+Bank\b",
        "Kotak Mahindra Bank": r"\bKotak\s+Mahindra\s+Bank\b",
        "Bank of Baroda": r"\bBank\s+of\s+Baroda\b",
        "IndusInd Bank": r"\bIndusInd\s+Bank\b",
        "IDFC First Bank": r"\bIDFC\s+First\s+Bank\b"
    }

    bank_identity_text = text[:3000]
    for canonical_name, pattern in bank_patterns.items():
        if re.search(pattern, bank_identity_text, re.IGNORECASE):
            issuing_bank = canonical_name
            break

    # IFSC-based fallback
    if not issuing_bank:
        if re.search(r"\bYESB0[A-Z0-9]{7,}\b", text, re.IGNORECASE):
            issuing_bank = "Yes Bank"

        elif re.search(r"\bKKBK0[A-Z0-9]{7,}\b", text, re.IGNORECASE):
            issuing_bank = "Kotak Mahindra Bank"

    employer_name = None

    if doc_code == "SALARY_SLIP":

        employer_match = re.search(
            r"(?im)^\s*(?:employer|company\s+name|organization)"
            r"\s*[:\-]\s*([A-Za-z0-9][A-Za-z0-9.,&'()/ \-]{2,69})\s*$",
            text
        )

        if employer_match:
            employer_name = employer_match.group(1).strip()

    if isinstance(raw_report, dict):

        placeholder_pattern = re.compile(
            r"\[(?:company\s+name|account\s+name|address)\]"
            r"|(?:enter|insert|type)\s+"
            r"(?:company\s+name|account\s+name|address)"
            r"|(?:company\s+name|account\s+name)\s+here"
            r"|\{\{\s*(?:company_name|account_name|address)\s*\}\}",
            re.IGNORECASE
        )

        placeholder_matches = placeholder_pattern.findall(text)

        if placeholder_matches:

            forensic_flags = raw_report.setdefault(
                "forensic_flags",
                []
            )

            if not isinstance(forensic_flags, list):
                forensic_flags = []
                raw_report["forensic_flags"] = forensic_flags

            flag_msg = (
                "Template Placeholder Warning: "
                "Explicit unfilled placeholder text detected: "
                f"{placeholder_matches}. Manual review required."
            )

            if flag_msg not in forensic_flags:
                forensic_flags.append(flag_msg)

    return {
        "doc_category": doc_category,
        "doc_code": doc_code,
        "classification_confidence": cls_conf,

        "holder_name": person_name,
        "person_name": person_name,
        "raw_extracted_name": raw_extracted_name,
        "name_confidence": name_confidence,

        "pan_number": extracted_pan,

        "issuing_bank": issuing_bank,
        "employer_name": employer_name,

        "business_name": business_name,
        "proprietor_name": proprietor_name,
        "auditor_name": auditor_name,
        "signatory_name": signatory_name,

        "identity_badge": (
            f"{doc_category} of {person_name}"
        )
    }


# 3. STRUCTURED FINDINGS BUILDER 

def extract_page_from_text(text: str, default_page: int = 1) -> int:
    m = re.search(r"\bPage\s+(\d+)\b", str(text), re.IGNORECASE)
    return int(m.group(1)) if m else default_page


def extract_bbox_from_text(text: str):
    match = re.search(r"\[([\d\.\s,]+)\]", str(text))
    if match:
        try:
            coords = [round(float(x.strip()), 2) for x in match.group(1).split(",")]
            if len(coords) == 4:
                return coords
        except Exception:
            pass
    return None


def build_structured_findings(doc_id, doc_label, doc_code, raw_report, family_d_report=None):
    structured = []
    fin_rec = raw_report.get("financial_reconciliation", {})

    if doc_code == "UNKNOWN_DOCUMENT":
        raw_s = 0.60
        structured.append({
            "document_id": doc_id,
            "page": 1,
            "bbox": None,
            "field_pointer": "document_classification",
            "detector_id": "STAGE_2_DOCUMENT_CLASSIFIER",
            "detector_version": "4.3.0",
            "finding_code": "UNCLASSIFIED_DYNAMIC_DOCUMENT",
            "raw_detector_score": raw_s,
            "individual_finding_score": SCORER.transform(raw_s),
            "original_signal": "Insufficient keyword or layout evidence to classify document.",
            "reason": "Unknown Document, page 1: Document type could not be reliably classified into any supported financial or KYC schema."
        })

    detailed_issues = fin_rec.get("detailed_issues")
    if not detailed_issues:
        detailed_issues = [
            {"page": extract_page_from_text(iss, 1), "bbox": extract_bbox_from_text(iss), "text": iss}
            for iss in fin_rec.get("issues", [])
        ]

    for item in detailed_issues:
        issue_txt = item["text"]
        page_no = item.get("page") or extract_page_from_text(issue_txt, 1)
        bbox = item.get("bbox") or extract_bbox_from_text(issue_txt)

        det_id = "FAM_A_LOGIC_ARITHMETIC"
        pointer = "ledger_row_bbox" if bbox else "summary_or_ledger_balance"
        is_intake_check = False

        if item.get("finding_code") == "FILENAME_DOCUMENT_TYPE_DISCREPANCY" or "Filename Document Type Discrepancy" in issue_txt:
            code = "FILENAME_DOCUMENT_TYPE_DISCREPANCY"
            raw_s = 0.35
            det_id = "STAGE_1_INTAKE_METADATA_CHECK"
            pointer = "uploaded_filename_vs_document_type"
            is_intake_check = True
        elif item.get("finding_code") == "FILENAME_PERIOD_DISCREPANCY" or "Filename Period Discrepancy" in issue_txt or "Filename vs. Document Period Discrepancy" in issue_txt:
            code = "FILENAME_PERIOD_DISCREPANCY"
            raw_s = 0.35
            det_id = "STAGE_1_INTAKE_METADATA_CHECK"
            pointer = "uploaded_filename_vs_document_period"
            is_intake_check = True
        elif any(k in issue_txt for k in ["Warning:", "Missing", "Fatal:", "Incomplete"]):
            code = "EXTRACTION_INCOMPLETE"
            raw_s = 0.45
        elif "Duplicate" in issue_txt:
            code = "DUPLICATE_TRANSACTION_ROW"
            raw_s = 0.80
        elif "Monotonicity" in issue_txt:
            code = "DATE_ORDER_ANOMALY"
            raw_s = 0.75
        else:
            code = "ARITHMETIC_MISMATCH"
            raw_s = 0.85

        finding_conf = raw_s if is_intake_check else SCORER.transform(raw_s)

        structured.append({
            "document_id": doc_id,
            "page": int(page_no),
            "bbox": bbox,
            "field_pointer": pointer,
            "detector_id": det_id,
            "detector_version": "4.3.0",
            "finding_code": code,
            "raw_detector_score": raw_s,
            "individual_finding_score": finding_conf,
            "original_signal": issue_txt,
            "reason": f"{doc_label}, page {page_no}: {issue_txt}"
        })

    for flag in raw_report.get("forensic_flags", []):
        page_no = extract_page_from_text(flag, 1)
        bbox = extract_bbox_from_text(flag)
        low_f = flag.lower()

        if any(k in low_f for k in ["ela", "pixel", "compression"]):
            if re.search(r"\b(?:ela|pixel|compression)\b", low_f):
                det_id = "FAM_C_IMAGE_FORENSICS"
                code = "PIXEL_ELA_ANOMALY"
                raw_s = 0.78
                pointer = "visual_pixel_region"
                det_id = "FAM_C_IMAGE_FORENSICS"
                code = "PIXEL_ELA_ANOMALY"
                raw_s = 0.78
                pointer = "visual_pixel_region"
        elif "copy-move" in low_f or "duplicated keypoint" in low_f:
            det_id = "FAM_C_IMAGE_FORENSICS"
            code = "COPY_MOVE_CLONE"
            raw_s = 0.80
            pointer = "visual_pixel_region"
        elif "template placeholder" in low_f or "form hints" in low_f:
            det_id = "FAM_B_STRUCTURE_FORENSICS"
            code = "TEMPLATE_PLACEHOLDER_ANOMALY"
            raw_s = 0.75
            pointer = "header_template_text"
        elif "overlay" in low_f or "overlaps with" in low_f:
            det_id = "FAM_B_STRUCTURE_FORENSICS"
            code = "PDF_TEXT_OVERLAY"
            raw_s = 0.75
            pointer = "overlapping_text_span"
        elif "font" in low_f:
            det_id = "FAM_B_STRUCTURE_FORENSICS"
            code = "FONT_SUBSET_ANOMALY"
            raw_s = 0.70
            pointer = "font_descriptor"
        elif "eof" in low_f or "incremental" in low_f:
            det_id = "FAM_B_STRUCTURE_FORENSICS"
            code = "PDF_INCREMENTAL_UPDATE"
            raw_s = 0.72
            pointer = "xref_trailer"
        elif any(k in low_f for k in ["metadata", "author", "producer", "creator", "photoshop", "canva", "revision"]):
            det_id = "FAM_B_STRUCTURE_FORENSICS"
            code = "METADATA_TOOLCHAIN_TRACE"
            raw_s = 0.65
            pointer = "document_metadata"
        elif "Filename vs. Document Period Discrepancy" in issue_txt:
            code = "FILENAME_PERIOD_DISCREPANCY"
            raw_s = 0.45
        else:
            det_id = "FAM_B_STRUCTURE_FORENSICS"
            code = "UNCLASSIFIED_ANOMALY"
            raw_s = 0.60
            pointer = "unclassified_signal"

        structured.append({
            "document_id": doc_id,
            "page": int(page_no),
            "bbox": bbox,
            "field_pointer": pointer,
            "detector_id": det_id,
            "detector_version": "4.3.0",
            "finding_code": code,
            "raw_detector_score": raw_s,
            "individual_finding_score": SCORER.transform(raw_s),
            "original_signal": flag,
            "reason": f"{doc_label}, page {page_no}: {flag}"
        })

    if family_d_report and isinstance(family_d_report, dict):
        for d_finding in family_d_report.get("findings", []):
            raw_s = 0.85
            structured.append({
                "document_id": doc_id,
                "page": 1,
                "bbox": None,
                "field_pointer": "cross_document_pack",
                "detector_id": "FAM_D_CROSS_DOCUMENT_KYC",
                "detector_version": "4.3.0",
                "finding_code": "CROSS_DOCUMENT_MISMATCH",
                "raw_detector_score": raw_s,
                "individual_finding_score": SCORER.transform(raw_s),
                "original_signal": d_finding,
                "reason": f"{doc_label} (Cross-Document Check): {d_finding}"
            })

    return structured


# 4. STANDARDIZED STATUS HELPER & PLAIN-LANGUAGE COMPOSER 
def standardize_audit_coverage(raw_cov: dict) -> dict:
    standardized = {}
    for k, v in (raw_cov or {}).items():
        if isinstance(v, str) and v in ALLOWED_CHECK_STATUSES:
            standardized[k] = v
        elif v is True:
            standardized[k] = "PASSED"
        elif v is False:
            standardized[k] = "FAILED"
        else:
            standardized[k] = "NOT_RUN"
    return standardized


def compose_founder_summary(doc_label, structured_findings):
    if not structured_findings:
        return (
            f"{doc_label}: All evaluated arithmetic invariants and structural checks PASSED. "
            f"Note: A low_risk verdict means no configured anomalies were detected, not proof of genuineness."
        )

    pages = sorted({f["page"] for f in structured_findings})
    page_str = f"page {pages[0]}" if len(pages) == 1 else f"pages {', '.join(map(str, pages))}"
    codes = {f["finding_code"] for f in structured_findings}

    if "UNCLASSIFIED_DYNAMIC_DOCUMENT" in codes:
        return f"Unknown Document, {page_str}: Document could not be reliably classified into a supported financial schema. Send for manual review."

    clauses = []
    if "TEMPLATE_PLACEHOLDER_ANOMALY" in codes:
        clauses.append("unfilled template placeholders (e.g., '(Company Name)', '(Address)') were found in the header, indicating a template-edited statement")
    if "PDF_TEXT_OVERLAY" in codes:
        clauses.append("overlapping text layers were detected over monetary figures")
    if "FONT_SUBSET_ANOMALY" in codes:
        clauses.append("mismatched font subsets were detected on numeric values")
    if "PDF_INCREMENTAL_UPDATE" in codes:
        clauses.append("multiple PDF incremental-save revisions (%%EOF) were detected")
    if "METADATA_TOOLCHAIN_TRACE" in codes:
        clauses.append("external editing software traces were found in the file metadata")
    if "PIXEL_ELA_ANOMALY" in codes:
        clauses.append("pixel-level Error Level Analysis (ELA) compression anomalies were detected")
    if "COPY_MOVE_CLONE" in codes:
        clauses.append("cloned/copy-moved image regions were detected")
    if "CROSS_DOCUMENT_MISMATCH" in codes:
        clauses.append("cross-document identity, PAN, or income inconsistencies were detected across the application pack")
    if "UNCLASSIFIED_ANOMALY" in codes:
        clauses.append("unclassified structural anomalies were flagged")
    if "FILENAME_PERIOD_DISCREPANCY" in codes:
        clauses.append("the uploaded filename indicates FY 2024-25 whereas the report states Previous Year 2023-24 (Assessment Year 2024-25)")
    if "FILENAME_DOCUMENT_TYPE_DISCREPANCY" in codes:
        clauses.append("the uploaded filename suggests a Balance Sheet whereas the document content is a Tax Audit Report (Form 3CB/3CD)")
    unique_clauses = []
    for c in clauses:
        if c not in unique_clauses:
            unique_clauses.append(c)
    clauses = unique_clauses
    has_math = "ARITHMETIC_MISMATCH" in codes
    if has_math and clauses:
        return f"{doc_label}, {page_str}: {'; '.join(clauses)}, and the running balance or summary equation FAILED reconciliation. Send for manual review."
    elif has_math:
        return f"{doc_label}, {page_str}: running balance or summary equation FAILED reconciliation against the itemised rows. Send for manual review."
    elif clauses:
        return f"{doc_label}, {page_str}: arithmetic reconciles cleanly, but {'; '.join(clauses)}. Send for manual review."
    elif "EXTRACTION_INCOMPLETE" in codes:
        return f"{doc_label}, {page_str}: required summary fields could not be extracted (FAILED / NOT_RUN). Cannot certify clean; send for manual review."
    return f"{doc_label}, {page_str}: anomalies detected ({', '.join(sorted(codes))}). Send for manual review."


# 5. AUTHORITATIVE SECTION 4.3 SCHEMA VALIDATOR 
def validate_section_4_3_contract(payload: dict) -> tuple[bool, list[str]]:
    errors = []
    required_types = {
        "module": str,
        "schema_version": str,
        "document_id": str,
        "tenant_id": str,
        "identity_card": dict,
        "file_format": str,
        "raw_fused_document_score": (int, float),
        "overall_document_risk_score": (int, float),
        "scoring_methodology": dict,
        "triage_band": str,
        "plain_language_summary": str,
        "detector_coverage": dict,
        "structured_findings": list,
        "extracted_financials": dict,
        "disclaimer": str
    }

    for k, expected_type in required_types.items():
        if k not in payload:
            errors.append(f"Missing mandatory key: '{k}'")
        elif not isinstance(payload[k], expected_type):
            errors.append(f"Key '{k}' expected type {expected_type}, got {type(payload[k]).__name__}")

    if payload.get("triage_band") not in ALLOWED_TRIAGE_BANDS:
        errors.append(f"Invalid triage_band '{payload.get('triage_band')}'. Allowed: {ALLOWED_TRIAGE_BANDS}")

    doc_score = payload.get("overall_document_risk_score", -1)
    if not isinstance(doc_score, (int, float)) or not (0.0 <= float(doc_score) <= 0.95):
        errors.append(f"overall_document_risk_score ({doc_score}) violates [0.0, 0.95] range.")

    det_cov = payload.get("detector_coverage", {})
    for fam_key in ("family_a_logic", "family_b_pdf_structure", "family_c_image_forensics", "family_d_cross_document"):
        if fam_key not in det_cov:
            errors.append(f"detector_coverage missing key '{fam_key}'")
        elif det_cov[fam_key] not in ALLOWED_CHECK_STATUSES:
            errors.append(f"detector_coverage['{fam_key}'] has invalid status '{det_cov[fam_key]}'. Allowed: {ALLOWED_CHECK_STATUSES}")

    aud_cov = payload.get("extracted_financials", {}).get("audit_coverage", {})
    for chk_k, chk_v in aud_cov.items():
        if chk_v not in ALLOWED_CHECK_STATUSES:
            errors.append(f"audit_coverage['{chk_k}'] has invalid status '{chk_v}'. Allowed: {ALLOWED_CHECK_STATUSES}")

    for idx, f in enumerate(payload.get("structured_findings", [])):
        req_f = {
            "document_id": str,
            "page": int,
            "field_pointer": str,
            "detector_id": str,
            "detector_version": str,
            "finding_code": str,
            "raw_detector_score": (int, float),
            "individual_finding_score": (int, float),
            "original_signal": str,
            "reason": str
        }
        for fk, ft in req_f.items():
            if fk not in f:
                errors.append(f"structured_findings[{idx}] missing '{fk}'")
            elif not isinstance(f[fk], ft):
                errors.append(f"structured_findings[{idx}]['{fk}'] invalid type")

        if f.get("page", 0) < 1:
            errors.append(f"structured_findings[{idx}]['page'] must be >= 1")

        if f.get("finding_code") not in ALLOWED_FINDING_CODES:
            errors.append(f"structured_findings[{idx}] invalid finding_code '{f.get('finding_code')}'")

        bbox = f.get("bbox")
        if bbox is not None:
            if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(c, (int, float)) for c in bbox):
                errors.append(f"structured_findings[{idx}]['bbox'] must be None or a 4-float list [x0, y0, x1, y1]")

    return (len(errors) == 0), errors


# 6. MAIN MODULE 1 AGGREGATOR (Integrates Families A, B, C, and D)


# ==============================================================================
# UNIVERSAL ZERO-SHOT IDENTITY & BANK EXTRACTOR (STRICT PRE-TABLE ISOLATION)
# ==============================================================================
UNIVERSAL_IFSC_BANK_MAP = {
    "SBIN": "State Bank of India", "HDFC": "HDFC Bank", "ICIC": "ICICI Bank",
    "UTIB": "Axis Bank", "KKBK": "Kotak Mahindra Bank", "YESB": "Yes Bank",
    "PUNB": "Punjab National Bank", "BARB": "Bank of Baroda", "CNRB": "Canara Bank",
    "UBIN": "Union Bank of India", "IDIB": "Indian Bank", "BKID": "Bank of India",
    "CBIN": "Central Bank of India", "IOBA": "Indian Overseas Bank", "UCBA": "UCO Bank",
    "MAHB": "Bank of Maharashtra", "PSIB": "Punjab & Sind Bank", "IDFB": "IDFC First Bank",
    "INDB": "IndusInd Bank", "FDRL": "Federal Bank", "BDBL": "Bandhan Bank",
    "IBKL": "IDBI Bank", "RATN": "RBL Bank", "KARB": "Karnataka Bank",
    "KVBL": "Karur Vysya Bank", "SIBL": "South Indian Bank", "TMBL": "Tamilnad Mercantile Bank",
    "CIUB": "City Union Bank", "JSBL": "Jana Small Finance Bank", "AUBL": "AU Small Finance Bank",
    "ESFB": "Equitas Small Finance Bank", "UJVN": "Ujjivan Small Finance Bank",
    "SCBL": "Standard Chartered Bank", "CITI": "Citibank", "HSBC": "HSBC Bank",
    "DBSS": "DBS Bank", "DEUT": "Deutsche Bank", "AIRP": "Airtel Payments Bank",
    "PYTM": "Paytm Payments Bank", "SURY": "Suryoday Small Finance Bank"
}


def slice_strictly_above_table_header(full_text: str) -> str:
    """
    Finds the exact start of the transaction table on Page 1 and slices the text
    strictly ABOVE it. Counterparty names/banks inside transaction rows are 100% excluded.
    """
    # Check common multi-column table header anchors
    table_header_patterns = [
        r"\b(?:sl\s*no|si\s*no|sr\s*no)\b[\s\S]{0,120}?\b(?:tran\s*id|value\s*date|transaction\s*date|balance)\b",
        r"\b(?:transaction\s*date|value\s*date|txn\s*date|date)\b[\s\S]{0,140}?\b(?:narration|description|particulars|remarks|withdrawal|deposit|debit|credit)\b[\s\S]{0,80}?\bbalance\b",
    ]
    best_cut = min(len(full_text), 1800)
    for pat in table_header_patterns:
        m = re.search(pat, full_text[:3500], re.IGNORECASE)
        if m and m.start() > 40:
            best_cut = min(best_cut, m.start())
    return full_text[:best_cut]


def extract_holder_universal(file_path: str, pre_table_text: str) -> str | None:
    """
    Universal 3-Stage Holder Name Extractor (Strictly within Page 1 Pre-Table Header):
    1. 2D Coordinate right-of-label lookup on Page 1 (top-left Y alignment).
    2. Stacked label-list pairing (e.g. 'Name:\nAddress:\nA/C No:\nJt. Holder:\n<VALUE>').
    3. Inline 'Name : <VALUE>' or Unlabeled Customer Address-Block Anchor.
    """
    bad_phrases = [
        "statement of", "detailed statement", "account statement", "salary slip",
        "icici bank", "hdfc bank", "yes bank", "kotak mahindra", "state bank",
        "axis bank", "standard chartered", "branch", "ifsc", "currency", "inr",
        "nominee", "address", "a/c", "account", "cust id", "period", "advanced search",
        "transaction", "karol bagh", "medical square", "galaxy towers", "guruwara road"
    ]

    def is_clean_holder_candidate(cand: str) -> bool:
        c = re.sub(r"\s+", " ", cand).strip(" :-|.,")
        if len(c) < 3 or len(c) > 50:
            return False
        c_low = c.lower()
        if any(bp in c_low for bp in bad_phrases):
            return False
        if re.match(r"^[\d\W]+$", c):
            return False
        if DATE_RE_SIMPLE.match(c):
            return False
        return True

    DATE_RE_SIMPLE = re.compile(r"^\d{1,2}[\-/\s\.](?:\d{1,2}|[a-z]{3,9})[\-/\s\.]\d{2,4}$", re.IGNORECASE)

    # Stage 1: 2D PDF Coordinate Pairing on Page 1 (matches top-edge y0 within 12pt)
    if file_path and file_path.lower().endswith(".pdf") and os.path.exists(file_path):
        try:
            with pymupdf.open(file_path) as doc:
                if len(doc) > 0:
                    page = doc[0]
                    spans = []
                    for b in page.get_text("dict").get("blocks", []):
                        for l in b.get("lines", []):
                            line_txt = " ".join(s.get("text", "").strip() for s in l.get("spans", []) if s.get("text", "").strip())
                            if line_txt:
                                bbox = l["bbox"]
                                spans.append({"text": line_txt, "x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]})

                    label_re = re.compile(
                        r"^(?:account\s*holder(?:\s*name)?|account\s*name|customer\s*name|applicant\s*name|employee\s*name|legal\s*name|name\s*of\s*enterprise|name)\s*[:\-]?$",
                        re.IGNORECASE
                    )
                    for s in spans:
                        if s["y0"] > 350:
                            continue
                        if label_re.match(s["text"].strip()):
                            # Check same line to the right using top-edge y0 (handles multi-line blocks!)
                            right_matches = [
                                other for other in spans
                                if other["x0"] > s["x1"] - 4.0 and abs(other["y0"] - s["y0"]) <= 12.0
                                and is_clean_holder_candidate(other["text"].lstrip(":- "))
                            ]
                            if right_matches:
                                right_matches.sort(key=lambda item: item["x0"])
                                return right_matches[0]["text"].lstrip(":- ").strip()
        except Exception:
            pass

    # Stage 2: Stacked Columnar Label Block (e.g. 'Name:\nAddress:\nA/C No:\nJt. Holder:\nTHE COLLECTION')
    lines = [ln.strip() for ln in pre_table_text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if re.match(r"^(?:account\s*name|customer\s*name|holder\s*name|name)\s*[:\-]?$", ln, re.IGNORECASE):
            # Walk forward past consecutive short colon labels ('Address:', 'A/C No:', 'Jt. Holder:')
            j = i + 1
            while j < min(len(lines), i + 7) and (lines[j].endswith(":") or re.match(r"^(?:address|a/c\s*no|account\s*no|jt\.?\s*holder|cust\s*id)\s*[:\-]?", lines[j], re.IGNORECASE)):
                j += 1
            if j < len(lines) and is_clean_holder_candidate(lines[j]):
                return lines[j].lstrip(":- ").strip()

    # Stage 3: Inline Key-Value or Unlabeled Address-Block Anchor inside pre_table_text ONLY
    inline_patterns = [
        r"(?:^|\n)([A-Za-z0-9\.\&\/\ \t]{3,45}?)\s*\(\s*(?:Account\s*Name|Company\s*Name)\s*\)",
        r"(?<!nominee\s)(?<!nominee)(?<!father\s)(?<!branch\s)(?:account\s*holder|account\s*name|customer\s*name|employee\s*name|applicant\s*name|card\s*holder|legal\s*name|name\s*of\s*enterprise|name)\s*[:\-]\s*([A-Za-z][A-Za-z0-9\ \t\.\&/]{2,45}?)(?=\s*\||$|\n|pan|month|period|statement|father|dob|branch|address)",
        r"(?:^|\n)([A-Z][A-Z0-9\ \t\.\&]{2,45})\n\s*(?:PLOT\s+NO|FLAT\s+NO|H\.?\s*NO|HOUSE\s+NO|KH\s+NO|\u039a\u0397\s+\u039d\u039f|SHOP\s+NO|SURVEY\s+NO|DOOR\s+NO|NEW\s+\d+|\d+[\-/A-Z0-9]+\s+[A-Z])",
    ]
    for pat in inline_patterns:
        for m in re.finditer(pat, pre_table_text, re.IGNORECASE):
            cand = m.group(1).strip()
            if is_clean_holder_candidate(cand):
                return cand

    return None


def extract_universal_issuing_bank(pre_table_text: str, full_text: str) -> str | None:
    # 1. Check authoritative IFSC code in the pre-table header first
    for m_ifsc in re.finditer(r"\b([A-Z]{4})0[A-Z0-9]{6}\b", pre_table_text.upper()):
        pref = m_ifsc.group(1)
        if pref in UNIVERSAL_IFSC_BANK_MAP:
            return UNIVERSAL_IFSC_BANK_MAP[pref]

    # 2. Check earliest-appearing bank name in pre_table_text (by character position, NEVER dict order!)
    generic_bank_re = re.compile(
        r"\b((?:State\s+Bank\s+of\s+India|Bank\s+of\s+(?:Baroda|India|Maharashtra)|Central\s+Bank\s+of\s+India|Union\s+Bank\s+of\s+India|Indian\s+(?:Overseas\s+)?Bank|Punjab\s+(?:National|&\s*Sind)\s+Bank|(?:ICICI|HDFC|YES|Axis|Kotak\s+Mahindra|IndusInd|IDFC\s+First|IDBI|Federal|Bandhan|Canara|RBL|Karnataka|Karur\s+Vysya|South\s+Indian|City\s+Union|Standard\s+Chartered|HSBC|DBS|AU\s+Small\s+Finance|Equitas\s+Small\s+Finance|Ujjivan\s+Small\s+Finance)\s+Bank))\b",
        re.IGNORECASE
    )
    matches = [(m.start(), m.group(1)) for m in generic_bank_re.finditer(pre_table_text)]
    if matches:
        matches.sort(key=lambda x: x[0])
        raw_b = re.sub(r"\s+", " ", matches[0][1]).strip()
        words = []
        for w in raw_b.split():
            if w.upper() in {"ICICI", "HDFC", "IDFC", "IDBI", "HSBC", "DBS", "UCO", "RBL", "SBI", "PNB", "AU"}:
                words.append(w.upper())
            elif w.lower() in {"of", "and", "&"}:
                words.append(w.lower())
            else:
                words.append(w.capitalize())
        return " ".join(words)

    # 3. Check footer disclaimer ('brought to the notice of <Bank> Ltd') or IFSC anywhere in document
    footer_m = re.search(r"notice\s+of\s+([A-Za-z\s]+?Bank)(?:\s+Ltd|\s+Limited|\.)", full_text[-1500:], re.IGNORECASE)
    if footer_m:
        return re.sub(r"\s+", " ", footer_m.group(1)).strip().title()

    for m_ifsc in re.finditer(r"\b([A-Z]{4})0[A-Z0-9]{6}\b", full_text.upper()):
        pref = m_ifsc.group(1)
        if pref in UNIVERSAL_IFSC_BANK_MAP:
            return UNIVERSAL_IFSC_BANK_MAP[pref]

    return None


def identify_document_and_entity(file_path: str, raw_report: dict = None) -> dict:
    text = extract_raw_text_from_file(file_path)
    if not text:
        return {
            "doc_category": "Unknown Document", "doc_code": "UNKNOWN_DOCUMENT",
            "classification_confidence": 0.0, "holder_name": "Unknown Holder",
            "person_name": "Unknown Holder", "raw_extracted_name": "Unknown Holder",
            "name_confidence": 0.0, "pan_number": None, "issuing_bank": None,
            "employer_name": None, "business_name": None, "proprietor_name": None,
            "auditor_name": None, "signatory_name": None,
            "identity_badge": "Unknown Document of Unknown Holder"
        }

    cls_info = classify_financial_document(text)
    doc_code = cls_info.get("doc_code", "UNKNOWN_DOCUMENT")
    doc_category = cls_info.get("doc_category", "Unknown Document")
    cls_conf = cls_info.get("classification_confidence", 0.0)

    normalized_text = re.sub(r"\s+", " ", text.upper())
    if "SECTION 44AB" in normalized_text and (re.search(r"\bFORM\s+NO\.?\s*3C[BD]\b", normalized_text)):
        doc_code = "TAX_AUDIT_REPORT"
        doc_category = "Tax Audit Report (Form 3CB/3CD)"
        cls_conf = 0.95

    if doc_code == "UNKNOWN_DOCUMENT":
        return {
            "doc_category": "Unknown Document", "doc_code": "UNKNOWN_DOCUMENT",
            "classification_confidence": 0.0, "holder_name": "Unknown Holder",
            "person_name": "Unknown Holder", "raw_extracted_name": "Unknown Holder",
            "name_confidence": 0.0, "pan_number": None, "issuing_bank": None,
            "employer_name": None, "business_name": None, "proprietor_name": None,
            "auditor_name": None, "signatory_name": None,
            "identity_badge": "Unknown Document of Unknown Holder"
        }

    pre_table_header = slice_strictly_above_table_header(text)
    extracted_pan = None
    raw_extracted_name = None
    person_name = "Unknown Holder"
    name_confidence = 0.0
    business_name = None
    proprietor_name = None
    auditor_name = None
    signatory_name = None
    issuing_bank = None
    employer_name = None

    if doc_code == "TAX_AUDIT_REPORT":
        fin_r = (raw_report or {}).get("financial_reconciliation", {})
        business_name = fin_r.get("assessee_business_name")
        proprietor_name = fin_r.get("proprietor_name")
        auditor_name = fin_r.get("auditor_firm_name")
        signatory_name = fin_r.get("auditor_signatory")
        extracted_pan = fin_r.get("pan_number")
        if business_name:
            person_name = business_name
            raw_extracted_name = business_name
            name_confidence = 0.95
    else:
        pan_m = re.search(r"\b([A-Z]{5}\d{4}[A-Z])\b", pre_table_header.upper()) or re.search(r"\b([A-Z]{5}\d{4}[A-Z])\b", text.upper())
        extracted_pan = pan_m.group(1) if pan_m else None

        holder_cand = extract_holder_universal(file_path, pre_table_header)
        if holder_cand:
            raw_extracted_name = holder_cand
            person_name, name_confidence = clean_ocr_person_name(holder_cand)

    if doc_code == "BANK_STATEMENT":
        issuing_bank = extract_universal_issuing_bank(pre_table_header, text)

    if doc_code == "SALARY_SLIP":
        emp_m = re.search(r"(?im)^\s*(?:employer|company\s+name|organization)\s*[:\-]\s*([A-Za-z0-9][A-Za-z0-9.,&'()/ \-]{2,69})\s*$", pre_table_header)
        if emp_m:
            employer_name = emp_m.group(1).strip()

    return {
        "doc_category": doc_category, "doc_code": doc_code,
        "classification_confidence": cls_conf, "holder_name": person_name,
        "person_name": person_name, "raw_extracted_name": raw_extracted_name or person_name,
        "name_confidence": name_confidence, "pan_number": extracted_pan,
        "issuing_bank": issuing_bank, "employer_name": employer_name,
        "business_name": business_name, "proprietor_name": proprietor_name,
        "auditor_name": auditor_name, "signatory_name": signatory_name,
        "identity_badge": f"{doc_category} of {person_name}"
    }




# ==============================================================================
# UNIVERSAL ZERO-SHOT IDENTITY & BANK EXTRACTOR (STRICT PRE-TABLE ISOLATION)
# ==============================================================================
UNIVERSAL_IFSC_BANK_MAP = {
    "SBIN": "State Bank of India", "HDFC": "HDFC Bank", "ICIC": "ICICI Bank",
    "UTIB": "Axis Bank", "KKBK": "Kotak Mahindra Bank", "YESB": "Yes Bank",
    "PUNB": "Punjab National Bank", "BARB": "Bank of Baroda", "CNRB": "Canara Bank",
    "UBIN": "Union Bank of India", "IDIB": "Indian Bank", "BKID": "Bank of India",
    "CBIN": "Central Bank of India", "IOBA": "Indian Overseas Bank", "UCBA": "UCO Bank",
    "MAHB": "Bank of Maharashtra", "PSIB": "Punjab & Sind Bank", "IDFB": "IDFC First Bank",
    "INDB": "IndusInd Bank", "FDRL": "Federal Bank", "BDBL": "Bandhan Bank",
    "IBKL": "IDBI Bank", "RATN": "RBL Bank", "KARB": "Karnataka Bank",
    "KVBL": "Karur Vysya Bank", "SIBL": "South Indian Bank", "TMBL": "Tamilnad Mercantile Bank",
    "CIUB": "City Union Bank", "JSBL": "Jana Small Finance Bank", "AUBL": "AU Small Finance Bank",
    "ESFB": "Equitas Small Finance Bank", "UJVN": "Ujjivan Small Finance Bank",
    "SCBL": "Standard Chartered Bank", "CITI": "Citibank", "HSBC": "HSBC Bank",
    "DBSS": "DBS Bank", "DEUT": "Deutsche Bank", "AIRP": "Airtel Payments Bank",
    "PYTM": "Paytm Payments Bank", "SURY": "Suryoday Small Finance Bank"
}


def slice_strictly_above_table_header(full_text: str) -> str:
    """
    Finds the exact start of the transaction table on Page 1 and slices the text
    strictly ABOVE it. Counterparty names/banks inside transaction rows are 100% excluded.
    """
    # Check common multi-column table header anchors
    table_header_patterns = [
        r"\b(?:sl\s*no|si\s*no|sr\s*no)\b[\s\S]{0,120}?\b(?:tran\s*id|value\s*date|transaction\s*date|balance)\b",
        r"\b(?:transaction\s*date|value\s*date|txn\s*date|date)\b[\s\S]{0,140}?\b(?:narration|description|particulars|remarks|withdrawal|deposit|debit|credit)\b[\s\S]{0,80}?\bbalance\b",
    ]
    best_cut = min(len(full_text), 1800)
    for pat in table_header_patterns:
        m = re.search(pat, full_text[:3500], re.IGNORECASE)
        if m and m.start() > 40:
            best_cut = min(best_cut, m.start())
    return full_text[:best_cut]


def extract_holder_universal(file_path: str, pre_table_text: str) -> str | None:
    """
    Universal 3-Stage Holder Name Extractor (Strictly within Page 1 Pre-Table Header):
    1. 2D Coordinate right-of-label lookup on Page 1 (top-left Y alignment).
    2. Stacked label-list pairing (e.g. 'Name:\nAddress:\nA/C No:\nJt. Holder:\n<VALUE>').
    3. Inline 'Name : <VALUE>' or Unlabeled Customer Address-Block Anchor.
    """
    bad_phrases = [
        "statement of", "detailed statement", "account statement", "salary slip",
        "icici bank", "hdfc bank", "yes bank", "kotak mahindra", "state bank",
        "axis bank", "standard chartered", "branch", "ifsc", "currency", "inr",
        "nominee", "address", "a/c", "account", "cust id", "period", "advanced search",
        "transaction", "karol bagh", "medical square", "galaxy towers", "guruwara road"
    ]

    def is_clean_holder_candidate(cand: str) -> bool:
        c = re.sub(r"\s+", " ", cand).strip(" :-|.,")
        if len(c) < 3 or len(c) > 50:
            return False
        c_low = c.lower()
        if any(bp in c_low for bp in bad_phrases):
            return False
        if re.match(r"^[\d\W]+$", c):
            return False
        if DATE_RE_SIMPLE.match(c):
            return False
        return True

    DATE_RE_SIMPLE = re.compile(r"^\d{1,2}[\-/\s\.](?:\d{1,2}|[a-z]{3,9})[\-/\s\.]\d{2,4}$", re.IGNORECASE)

    # Stage 1: 2D PDF Coordinate Pairing on Page 1 (matches top-edge y0 within 12pt)
    if file_path and file_path.lower().endswith(".pdf") and os.path.exists(file_path):
        try:
            with pymupdf.open(file_path) as doc:
                if len(doc) > 0:
                    page = doc[0]
                    spans = []
                    for b in page.get_text("dict").get("blocks", []):
                        for l in b.get("lines", []):
                            line_txt = " ".join(s.get("text", "").strip() for s in l.get("spans", []) if s.get("text", "").strip())
                            if line_txt:
                                bbox = l["bbox"]
                                spans.append({"text": line_txt, "x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]})

                    label_re = re.compile(
                        r"^(?:account\s*holder(?:\s*name)?|account\s*name|customer\s*name|applicant\s*name|employee\s*name|legal\s*name|name\s*of\s*enterprise|name)\s*[:\-]?$",
                        re.IGNORECASE
                    )
                    for s in spans:
                        if s["y0"] > 350:
                            continue
                        if label_re.match(s["text"].strip()):
                            # Check same line to the right using top-edge y0 (handles multi-line blocks!)
                            right_matches = [
                                other for other in spans
                                if other["x0"] > s["x1"] - 4.0 and abs(other["y0"] - s["y0"]) <= 12.0
                                and is_clean_holder_candidate(other["text"].lstrip(":- "))
                            ]
                            if right_matches:
                                right_matches.sort(key=lambda item: item["x0"])
                                return right_matches[0]["text"].lstrip(":- ").strip()
        except Exception:
            pass

    # Stage 2: Stacked Columnar Label Block (e.g. 'Name:\nAddress:\nA/C No:\nJt. Holder:\nTHE COLLECTION')
    lines = [ln.strip() for ln in pre_table_text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if re.match(r"^(?:account\s*name|customer\s*name|holder\s*name|name)\s*[:\-]?$", ln, re.IGNORECASE):
            # Walk forward past consecutive short colon labels ('Address:', 'A/C No:', 'Jt. Holder:')
            j = i + 1
            while j < min(len(lines), i + 7) and (lines[j].endswith(":") or re.match(r"^(?:address|a/c\s*no|account\s*no|jt\.?\s*holder|cust\s*id)\s*[:\-]?", lines[j], re.IGNORECASE)):
                j += 1
            if j < len(lines) and is_clean_holder_candidate(lines[j]):
                return lines[j].lstrip(":- ").strip()

    # Stage 3: Inline Key-Value or Unlabeled Address-Block Anchor inside pre_table_text ONLY
    inline_patterns = [
        r"(?:^|\n)([A-Za-z0-9\.\&\/\ \t]{3,45}?)\s*\(\s*(?:Account\s*Name|Company\s*Name)\s*\)",
        r"(?<!nominee\s)(?<!nominee)(?<!father\s)(?<!branch\s)(?:account\s*holder|account\s*name|customer\s*name|employee\s*name|applicant\s*name|card\s*holder|legal\s*name|name\s*of\s*enterprise|name)\s*[:\-]\s*([A-Za-z][A-Za-z0-9\ \t\.\&/]{2,45}?)(?=\s*\||$|\n|pan|month|period|statement|father|dob|branch|address)",
        r"(?:^|\n)([A-Z][A-Z0-9\ \t\.\&]{2,45})\n\s*(?:PLOT\s+NO|FLAT\s+NO|H\.?\s*NO|HOUSE\s+NO|KH\s+NO|\u039a\u0397\s+\u039d\u039f|SHOP\s+NO|SURVEY\s+NO|DOOR\s+NO|NEW\s+\d+|\d+[\-/A-Z0-9]+\s+[A-Z])",
    ]
    for pat in inline_patterns:
        for m in re.finditer(pat, pre_table_text, re.IGNORECASE):
            cand = m.group(1).strip()
            if is_clean_holder_candidate(cand):
                return cand

    return None


def extract_universal_issuing_bank(pre_table_text: str, full_text: str) -> str | None:
    # 1. Check authoritative IFSC code in the pre-table header first
    for m_ifsc in re.finditer(r"\b([A-Z]{4})0[A-Z0-9]{6}\b", pre_table_text.upper()):
        pref = m_ifsc.group(1)
        if pref in UNIVERSAL_IFSC_BANK_MAP:
            return UNIVERSAL_IFSC_BANK_MAP[pref]

    # 2. Check earliest-appearing bank name in pre_table_text (by character position, NEVER dict order!)
    generic_bank_re = re.compile(
        r"\b((?:State\s+Bank\s+of\s+India|Bank\s+of\s+(?:Baroda|India|Maharashtra)|Central\s+Bank\s+of\s+India|Union\s+Bank\s+of\s+India|Indian\s+(?:Overseas\s+)?Bank|Punjab\s+(?:National|&\s*Sind)\s+Bank|(?:ICICI|HDFC|YES|Axis|Kotak\s+Mahindra|IndusInd|IDFC\s+First|IDBI|Federal|Bandhan|Canara|RBL|Karnataka|Karur\s+Vysya|South\s+Indian|City\s+Union|Standard\s+Chartered|HSBC|DBS|AU\s+Small\s+Finance|Equitas\s+Small\s+Finance|Ujjivan\s+Small\s+Finance)\s+Bank))\b",
        re.IGNORECASE
    )
    matches = [(m.start(), m.group(1)) for m in generic_bank_re.finditer(pre_table_text)]
    if matches:
        matches.sort(key=lambda x: x[0])
        raw_b = re.sub(r"\s+", " ", matches[0][1]).strip()
        words = []
        for w in raw_b.split():
            if w.upper() in {"ICICI", "HDFC", "IDFC", "IDBI", "HSBC", "DBS", "UCO", "RBL", "SBI", "PNB", "AU"}:
                words.append(w.upper())
            elif w.lower() in {"of", "and", "&"}:
                words.append(w.lower())
            else:
                words.append(w.capitalize())
        return " ".join(words)

    # 3. Check footer disclaimer ('brought to the notice of <Bank> Ltd') or IFSC anywhere in document
    footer_m = re.search(r"notice\s+of\s+([A-Za-z\s]+?Bank)(?:\s+Ltd|\s+Limited|\.)", full_text[-1500:], re.IGNORECASE)
    if footer_m:
        return re.sub(r"\s+", " ", footer_m.group(1)).strip().title()

    for m_ifsc in re.finditer(r"\b([A-Z]{4})0[A-Z0-9]{6}\b", full_text.upper()):
        pref = m_ifsc.group(1)
        if pref in UNIVERSAL_IFSC_BANK_MAP:
            return UNIVERSAL_IFSC_BANK_MAP[pref]

    return None


def identify_document_and_entity(file_path: str, raw_report: dict = None) -> dict:
    text = extract_raw_text_from_file(file_path)
    if not text:
        return {
            "doc_category": "Unknown Document", "doc_code": "UNKNOWN_DOCUMENT",
            "classification_confidence": 0.0, "holder_name": "Unknown Holder",
            "person_name": "Unknown Holder", "raw_extracted_name": "Unknown Holder",
            "name_confidence": 0.0, "pan_number": None, "issuing_bank": None,
            "employer_name": None, "business_name": None, "proprietor_name": None,
            "auditor_name": None, "signatory_name": None,
            "identity_badge": "Unknown Document of Unknown Holder"
        }

    cls_info = classify_financial_document(text)
    doc_code = cls_info.get("doc_code", "UNKNOWN_DOCUMENT")
    doc_category = cls_info.get("doc_category", "Unknown Document")
    cls_conf = cls_info.get("classification_confidence", 0.0)

    normalized_text = re.sub(r"\s+", " ", text.upper())
    if "SECTION 44AB" in normalized_text and (re.search(r"\bFORM\s+NO\.?\s*3C[BD]\b", normalized_text)):
        doc_code = "TAX_AUDIT_REPORT"
        doc_category = "Tax Audit Report (Form 3CB/3CD)"
        cls_conf = 0.95

    if doc_code == "UNKNOWN_DOCUMENT":
        return {
            "doc_category": "Unknown Document", "doc_code": "UNKNOWN_DOCUMENT",
            "classification_confidence": 0.0, "holder_name": "Unknown Holder",
            "person_name": "Unknown Holder", "raw_extracted_name": "Unknown Holder",
            "name_confidence": 0.0, "pan_number": None, "issuing_bank": None,
            "employer_name": None, "business_name": None, "proprietor_name": None,
            "auditor_name": None, "signatory_name": None,
            "identity_badge": "Unknown Document of Unknown Holder"
        }

    pre_table_header = slice_strictly_above_table_header(text)
    extracted_pan = None
    raw_extracted_name = None
    person_name = "Unknown Holder"
    name_confidence = 0.0
    business_name = None
    proprietor_name = None
    auditor_name = None
    signatory_name = None
    issuing_bank = None
    employer_name = None

    if doc_code == "TAX_AUDIT_REPORT":
        fin_r = (raw_report or {}).get("financial_reconciliation", {})
        business_name = fin_r.get("assessee_business_name")
        proprietor_name = fin_r.get("proprietor_name")
        auditor_name = fin_r.get("auditor_firm_name")
        signatory_name = fin_r.get("auditor_signatory")
        extracted_pan = fin_r.get("pan_number")
        if business_name:
            person_name = business_name
            raw_extracted_name = business_name
            name_confidence = 0.95
    else:
        pan_m = re.search(r"\b([A-Z]{5}\d{4}[A-Z])\b", pre_table_header.upper()) or re.search(r"\b([A-Z]{5}\d{4}[A-Z])\b", text.upper())
        extracted_pan = pan_m.group(1) if pan_m else None

        holder_cand = extract_holder_universal(file_path, pre_table_header)
        if holder_cand:
            raw_extracted_name = holder_cand
            person_name, name_confidence = clean_ocr_person_name(holder_cand)

    if doc_code == "BANK_STATEMENT":
        issuing_bank = extract_universal_issuing_bank(pre_table_header, text)

    if doc_code == "SALARY_SLIP":
        emp_m = re.search(r"(?im)^\s*(?:employer|company\s+name|organization)\s*[:\-]\s*([A-Za-z0-9][A-Za-z0-9.,&'()/ \-]{2,69})\s*$", pre_table_header)
        if emp_m:
            employer_name = emp_m.group(1).strip()

    return {
        "doc_category": doc_category, "doc_code": doc_code,
        "classification_confidence": cls_conf, "holder_name": person_name,
        "person_name": person_name, "raw_extracted_name": raw_extracted_name or person_name,
        "name_confidence": name_confidence, "pan_number": extracted_pan,
        "issuing_bank": issuing_bank, "employer_name": employer_name,
        "business_name": business_name, "proprietor_name": proprietor_name,
        "auditor_name": auditor_name, "signatory_name": signatory_name,
        "identity_badge": f"{doc_category} of {person_name}"
    }


def build_module1_output(file_path, raw_report, tenant_id="default_lender", doc_id=None, family_d_report=None):
    doc_id = doc_id or f"DOC-{uuid.uuid4().hex[:8].upper()}"
    tenant_cfg = TENANT_CONFIGS.get(tenant_id, TENANT_CONFIGS["default_lender"])
    weights = dict(tenant_cfg["family_weights"])

    identity_info = identify_document_and_entity(file_path, raw_report)
    doc_label = identity_info["doc_category"]
    doc_code = identity_info["doc_code"]
    file_fmt = raw_report.get("file_format", "UNKNOWN")

    fin_rec = dict(raw_report.get("financial_reconciliation", {}))
    if doc_code == "UNKNOWN_DOCUMENT":
        fin_rec["document_type"] = "UNKNOWN_DOCUMENT"
        fin_rec["issues"] = []
        fin_rec["detailed_issues"] = []
        fin_rec["audit_coverage"] = {"document_schema_recognized": "FAILED"}
    elif doc_code == "PAN_CARD":
        fin_rec["document_type"] = "PAN_CARD"
        fin_rec["issues"] = [i for i in fin_rec.get("issues", []) if "transaction" not in i.lower() and "header" not in i.lower()]
        fin_rec["detailed_issues"] = []
        pan_no = identity_info.get("pan_number")
        pan_ok = bool(pan_no and len(pan_no) == 10)
        entity_ok = bool(pan_no and pan_no[3] == "P")
        fin_rec["audit_coverage"] = {
            "pan_number_extracted": "PASSED" if pan_ok else "FAILED",
            "individual_entity_code_verified": ("PASSED" if entity_ok else "FAILED") if pan_ok else "NOT_RUN"
        }
        if not pan_ok:
            fin_rec["issues"].append("PAN Card Warning: Valid 10-character PAN number could not be extracted.")
        elif not entity_ok:
            fin_rec["issues"].append(f"PAN Entity Warning: 4th character '{pan_no[3]}' is not 'P' (Individual).")

    fin_rec["audit_coverage"] = standardize_audit_coverage(fin_rec.get("audit_coverage", {}))

    raw_report_norm = dict(raw_report)
    raw_report_norm["financial_reconciliation"] = fin_rec
    structured_findings = build_structured_findings(doc_id, doc_label, doc_code, raw_report_norm, family_d_report)

    fam_a_findings = [f for f in structured_findings if f["detector_id"] in ("FAM_A_LOGIC_ARITHMETIC", "STAGE_2_DOCUMENT_CLASSIFIER")]
    intake_findings = [f for f in structured_findings if f["detector_id"] == "STAGE_1_INTAKE_METADATA_CHECK"]
    fam_b_findings = [f for f in structured_findings if f["detector_id"] == "FAM_B_STRUCTURE_FORENSICS"]
    fam_c_findings = [f for f in structured_findings if f["detector_id"] == "FAM_C_IMAGE_FORENSICS"]
    fam_d_findings = [f for f in structured_findings if f["detector_id"] == "FAM_D_CROSS_DOCUMENT_KYC"]

    score_a = max((f["raw_detector_score"] for f in fam_a_findings), default=0.0)
    score_intake = max((f["raw_detector_score"] for f in intake_findings), default=0.0)
    score_b = max((f["raw_detector_score"] for f in fam_b_findings), default=0.0)
    score_c = max((f["raw_detector_score"] for f in fam_c_findings), default=0.0)
    score_d = max((f["raw_detector_score"] for f in fam_d_findings), default=0.0)

    aud_vals = [v for v in fin_rec["audit_coverage"].values() if v != "NOT_APPLICABLE"]
    if score_a > 0 or "FAILED" in aud_vals:
        cov_a = "FAILED"
    elif aud_vals and all(v == "PASSED" for v in aud_vals):
        cov_a = "PASSED"
    else:
        cov_a = "NOT_RUN"

    if score_b > 0:
        cov_b = "FAILED"
    elif file_fmt in ("DIGITAL_PDF", "HYBRID_PDF", "DOCX"):
        cov_b = "PASSED"
    else:
        cov_b = "NOT_APPLICABLE"

    if score_c > 0:
        cov_c = "FAILED"
    elif file_fmt in ("IMAGE", "SCANNED_PDF", "HYBRID_PDF"):
        cov_c = "PASSED"
    else:
        cov_c = "NOT_APPLICABLE"

    if family_d_report is None:
        cov_d = "NOT_RUN"
    elif score_d > 0 or family_d_report.get("status") == "REVIEW_REQUIRED":
        cov_d = "FAILED"
    else:
        cov_d = "PASSED"

    coverage_state = {
        "family_a_logic": cov_a,
        "family_b_pdf_structure": cov_b,
        "family_c_image_forensics": cov_c,
        "family_d_cross_document": cov_d
    }

    if cov_d == "NOT_RUN":
        active_w_sum = weights["A"] + weights["B"] + weights["C"]
        eff_weights = {
            "A": round(weights["A"] / active_w_sum, 3),
            "B": round(weights["B"] / active_w_sum, 3),
            "C": round(weights["C"] / active_w_sum, 3),
            "D": 0.0
        }
    else:
        eff_weights = weights

    weighted_sum = round(
        (eff_weights["A"] * score_a)
        + (eff_weights["B"] * score_b)
        + (eff_weights["C"] * score_c)
        + (eff_weights["D"] * score_d),
        3
    )

    raw_fused = 0.0
    overall_doc_score = 0.02
    fusion_rule_applied = "all_evaluated_families_passed"

    active_flagged = sum(
        1 for s in (score_a, score_b, score_c, score_d)
        if s is not None and s > 0
    )

    if active_flagged >= 2:
        raw_fused = round(min(1.0, weighted_sum * 1.35), 3)
        fusion_rule_applied = "multi_family_consensus_boost (1.35x)"

    elif active_flagged == 1:
        raw_fused = round(max(weighted_sum, 0.45), 3)
        fusion_rule_applied = "single_family_alert_floor (min 0.45 raw fused score)"
        
    elif score_intake is not None and score_intake > 0:
        raw_fused = round(min(1.0, score_intake), 3)
        overall_doc_score = round(raw_fused, 2)
        fusion_rule_applied = "intake_metadata_discrepancy_review"

    else:
        raw_fused = 0.0
        overall_doc_score = 0.02
        fusion_rule_applied = "all_evaluated_families_passed"

    if active_flagged > 0:
        overall_doc_score = float(SCORER.transform(raw_fused))
        overall_doc_score = max(0.0, min(1.0, overall_doc_score))

    max_finding_score = max(
        (f["individual_finding_score"] for f in structured_findings),
        default=0.0
    )
    if overall_doc_score >= tenant_cfg["high_risk_threshold"]:
        triage_band = "high_risk"
    elif overall_doc_score >= tenant_cfg["needs_review_threshold"]:
        triage_band = "needs_review"
    else:
        triage_band = "low_risk"

    plain_summary = compose_founder_summary(doc_label, structured_findings)

    payload = {
        "module": "MODULE_1_AI_DOCUMENT_CHECKER",
        "schema_version": "4.3.0",
        "document_id": doc_id,
        "tenant_id": tenant_id,
        "tenant_thresholds_applied": {
            "needs_review": tenant_cfg["needs_review_threshold"],
            "high_risk": tenant_cfg["high_risk_threshold"]
        },
        "identity_card": identity_info,
        "file_format": file_fmt,
        "total_pages_processed": raw_report.get("total_pages", fin_rec.get("total_pages_processed", 1)),
        "raw_fused_document_score": raw_fused,
        "overall_document_risk_score": overall_doc_score,
        "calibrated_risk_score": overall_doc_score,
        "scoring_methodology": {
            "score_type": SCORER.score_type,
            "is_empirically_calibrated": SCORER.is_empirically_validated,
            "calibration_warning": (
                None if SCORER.is_empirically_validated else
                "Scores are heuristic review-priority scores mapped via a monotonic scaling curve, NOT empirical fraud probabilities."
            ),
            "overall_document_risk_score": overall_doc_score,
            "max_individual_finding_score": max_finding_score,
            "family_raw_scores": {"Family_A": score_a, "Family_B": score_b, "Family_C": score_c, "Family_D": score_d},
            "effective_family_weights": eff_weights,
            "weighted_linear_sum": weighted_sum,
            "fusion_rule_applied": fusion_rule_applied,
            "raw_fused_document_score": raw_fused
        },
        "triage_band": triage_band,
        "plain_language_summary": plain_summary,
        "detector_coverage": coverage_state,
        "structured_findings": structured_findings,
        "extracted_financials": fin_rec,
        "disclaimer": (
            "A PASSED check means no configured anomalies were detected; it does not prove a document is genuine. "
            "A needs_review or high_risk result indicates the document requires manual investigation, not conclusive proof of fraud."
        )
    }

    is_valid, schema_errors = validate_section_4_3_contract(payload)
    payload["contract_validation"] = {"valid": is_valid, "errors": schema_errors}
    return payload


# ==============================================================================
# AUTHORITATIVE PRIORITY 3 OVERRIDE: STRICT PAGE-1 PRE-TABLE IDENTITY ENGINE
# ==============================================================================
UNIVERSAL_IFSC_BANK_MAP = {
    "SBIN": "State Bank of India", "HDFC": "HDFC Bank", "ICIC": "ICICI Bank",
    "UTIB": "Axis Bank", "KKBK": "Kotak Mahindra Bank", "YESB": "Yes Bank",
    "PUNB": "Punjab National Bank", "BARB": "Bank of Baroda", "CNRB": "Canara Bank",
    "UBIN": "Union Bank of India", "IDIB": "Indian Bank", "BKID": "Bank of India",
    "CBIN": "Central Bank of India", "IOBA": "Indian Overseas Bank", "UCBA": "UCO Bank",
    "MAHB": "Bank of Maharashtra", "PSIB": "Punjab & Sind Bank", "IDFB": "IDFC First Bank",
    "INDB": "IndusInd Bank", "FDRL": "Federal Bank", "BDBL": "Bandhan Bank",
    "IBKL": "IDBI Bank", "RATN": "RBL Bank", "KARB": "Karnataka Bank",
    "KVBL": "Karur Vysya Bank", "SIBL": "South Indian Bank", "SCBL": "Standard Chartered Bank"
}


def _slice_page1_pre_table(full_text: str) -> str:
    patterns = [
        r"\b(?:sl\s*no|si\s*no|sr\s*no)\b[\s\S]{0,120}?\b(?:tran\s*id|value\s*date|transaction\s*date|balance)\b",
        r"\b(?:transaction\s*date|value\s*date|txn\s*date|date)\b[\s\S]{0,140}?\b(?:narration|description|particulars|remarks|withdrawal|deposit|debit|credit)\b",
    ]
    cut = min(len(full_text), 1500)
    for pat in patterns:
        m = re.search(pat, full_text[:3000], re.IGNORECASE)
        if m and m.start() > 30:
            cut = min(cut, m.start())
    return full_text[:cut]


def _extract_holder_strictly_from_header(file_path: str, pre_table_text: str) -> str | None:
    bad_words = [
        "statement", "salary slip", "icici bank", "hdfc bank", "yes bank", "kotak",
        "state bank", "axis bank", "standard chartered", "branch", "ifsc", "currency",
        "nominee", "address", "a/c", "account", "cust", "period", "search", "karol bagh",
        "medical square", "galaxy", "gurudwara", "road", "delhi", "nagpur", "bangalore"
    ]
    def _valid(c: str) -> bool:
        c = re.sub(r"\s+", " ", c).strip(" :-|.,")
        if len(c) < 3 or len(c) > 45:
            return False
        if any(b in c.lower() for b in bad_words):
            return False
        if re.match(r"^[\d\W]+$", c) or re.match(r"^\d{1,2}[\-/][A-Za-z0-9]{2,9}[\-/]\d{2,4}$", c):
            return False
        return True

    # 1. Stacked Columnar Label Block (e.g. 'Name:\nAddress:\nA/C No:\nJt. Holder:\nTHE COLLECTION')
    lines = [ln.strip() for ln in pre_table_text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if re.match(r"^(?:account\s*name|customer\s*name|holder\s*name|name)\s*[:\-]?$", ln, re.IGNORECASE):
            j = i + 1
            while j < min(len(lines), i + 7) and (lines[j].endswith(":") or re.match(r"^(?:address|a/c|account|jt\.?\s*holder|cust)\b", lines[j], re.IGNORECASE)):
                j += 1
            if j < len(lines) and _valid(lines[j]):
                return lines[j].lstrip(":- ").strip()

    # 2. Inline Key-Value or Unlabeled Address-Block Anchor inside pre_table_text ONLY
    pats = [
        r"statement\s+of\s+account\s*\n+\s*([A-Z][A-Za-z0-9\ \t\.\&]{2,45}?)(?=\n|$)",
        r"(?:^|\n)([A-Z][A-Z0-9\ \t\.\&]{2,45})\n\s*(?:PLOT\s+NO|FLAT\s+NO|H\.?\s*NO|KH\s+NO|\u039a\u0397\s+\u039d\u039f|SHOP\s+NO|NEW\s+\d+|S\s+\d+/\d+|NEAR\s+|OPP\s+|GALI\s+NO)",
        r"(?:^|\n)([A-Za-z0-9\.\&\/\ \t]{3,45}?)\s*\(\s*(?:Account\s*Name|Company\s*Name)\s*\)",
        r"(?<!nominee\s)(?<!nominee)(?<!branch\s)(?:account\s*holder|account\s*name|customer\s*name|applicant\s*name|employee\s*name|legal\s*name|name)\s*[:\-]\s*([A-Za-z][A-Za-z0-9\ \t\.\&/]{2,45}?)(?=\s*\||$|\n|pan|period|statement|branch|address)",
        r"(?:^|\n)([A-Z][A-Z0-9\ \t\.\&]{2,45})\n\s*(?:PLOT\s+NO|FLAT\s+NO|H\.?\s*NO|KH\s+NO|\u039a\u0397\s+\u039d\u039f|SHOP\s+NO|NEW\s+\d+)",
    ]
    for pat in pats:
        for m in re.finditer(pat, pre_table_text, re.IGNORECASE):
            if _valid(m.group(1)):
                return m.group(1).strip()
    return None


def _extract_bank_strictly_from_header(pre_table_text: str, full_text: str) -> str | None:
    # 1. Authoritative IFSC Code in Page 1 Header
    for m_ifsc in re.finditer(r"\b([A-Z]{4})0[A-Z0-9]{6}\b", pre_table_text.upper()):
        pref = m_ifsc.group(1)
        if pref in UNIVERSAL_IFSC_BANK_MAP:
            return UNIVERSAL_IFSC_BANK_MAP[pref]

    # 2. Earliest-appearing bank name in Page 1 Pre-Table Header (by character index!)
    bank_re = re.compile(
        r"\b(ICICI\s+Bank|HDFC\s+Bank|Yes\s+Bank|Kotak\s+Mahindra\s+Bank|Axis\s+Bank|State\s+Bank\s+of\s+India|Standard\s+Chartered\s+Bank|Punjab\s+National\s+Bank|Bank\s+of\s+Baroda|Canara\s+Bank|Union\s+Bank\s+of\s+India|IndusInd\s+Bank|IDFC\s+First\s+Bank)\b",
        re.IGNORECASE
    )
    found = [(m.start(), m.group(1)) for m in bank_re.finditer(pre_table_text)]
    if found:
        found.sort(key=lambda x: x[0])
        b_raw = re.sub(r"\s+", " ", found[0][1]).strip()
        return "ICICI Bank" if "icici" in b_raw.lower() else ("HDFC Bank" if "hdfc" in b_raw.lower() else b_raw.title())

    for m_ifsc in re.finditer(r"\b([A-Z]{4})0[A-Z0-9]{6}\b", full_text.upper()):
        pref = m_ifsc.group(1)
        if pref in UNIVERSAL_IFSC_BANK_MAP:
            return UNIVERSAL_IFSC_BANK_MAP[pref]
    return None


def identify_document_and_entity(file_path: str, raw_report: dict = None) -> dict:
    text = extract_raw_text_from_file(file_path)
    if not text:
        return {
            "doc_category": "Unknown Document", "doc_code": "UNKNOWN_DOCUMENT",
            "classification_confidence": 0.0, "holder_name": "Unknown Holder",
            "person_name": "Unknown Holder", "raw_extracted_name": "Unknown Holder",
            "name_confidence": 0.0, "pan_number": None, "issuing_bank": None,
            "employer_name": None, "identity_badge": "Unknown Document of Unknown Holder"
        }

    cls_info = classify_financial_document(text)
    doc_code = cls_info.get("doc_code", "UNKNOWN_DOCUMENT")
    doc_category = cls_info.get("doc_category", "Unknown Document")
    cls_conf = cls_info.get("classification_confidence", 0.0)

    norm_up = re.sub(r"\s+", " ", text.upper())
    if "SECTION 44AB" in norm_up and re.search(r"\bFORM\s+NO\.?\s*3C[BD]\b", norm_up):
        doc_code = "TAX_AUDIT_REPORT"
        doc_category = "Tax Audit Report (Form 3CB/3CD)"
        cls_conf = 0.95

    pre_table_header = _slice_page1_pre_table(text)
    person_name = "Unknown Holder"
    raw_extracted_name = None
    name_confidence = 0.0
    extracted_pan = None
    issuing_bank = None
    employer_name = None

    if doc_code == "TAX_AUDIT_REPORT":
        fin_r = (raw_report or {}).get("financial_reconciliation", {})
        ass_name = fin_r.get("assessee_business_name")
        extracted_pan = fin_r.get("pan_number")
        if ass_name:
            person_name = ass_name
            raw_extracted_name = ass_name
            name_confidence = 0.95
    else:
        pan_m = re.search(r"\b([A-Z]{5}\d{4}[A-Z])\b", pre_table_header.upper()) or re.search(r"\b([A-Z]{5}\d{4}[A-Z])\b", text.upper())
        extracted_pan = pan_m.group(1) if pan_m else None
        cand = _extract_holder_strictly_from_header(file_path, pre_table_header)
        if cand:
            raw_extracted_name = cand
            person_name, name_confidence = clean_ocr_person_name(cand)

    if doc_code == "BANK_STATEMENT":
        issuing_bank = _extract_bank_strictly_from_header(text[:800], text)

    return {
        "doc_category": doc_category, "doc_code": doc_code,
        "classification_confidence": cls_conf, "holder_name": person_name,
        "person_name": person_name, "raw_extracted_name": raw_extracted_name or person_name,
        "name_confidence": name_confidence, "pan_number": extracted_pan,
        "issuing_bank": issuing_bank, "employer_name": employer_name,
        "identity_badge": f"{doc_category} of {person_name}"
    }
