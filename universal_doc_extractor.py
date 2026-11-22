import base64
import gc
import html
import io
import json
import os
import re
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf
from PIL import Image
from ocr_compat import run_ocr_normalized, _OCR_INSTANCE, get_ocr_engine


KNOWN_BANKS = [
    "INDIAN BANK", "STATE BANK OF INDIA", "SBI", "HDFC BANK", "ICICI BANK",
    "AXIS BANK", "KOTAK MAHINDRA BANK", "BANK OF BARODA", "PUNJAB NATIONAL BANK",
    "PNB", "CANARA BANK", "UNION BANK OF INDIA", "BANK OF INDIA", "IDBI BANK",
    "INDUSIND BANK", "YES BANK", "IDFC FIRST BANK", "STANDARD CHARTERED",
]

FIELD_LABELS = {
    "account_holder": ["account holder", "account holder name", "name of account holder", "customer name"],
    "account_number": ["account number", "account no", "a/c no", "acct no"],
    "ifsc_code": ["ifsc", "ifsc code", "rtgs/neft ifsc"],
    "pan": ["pan", "pan no", "pan number", "pan card"],
    "bank_name": ["bank name", "bank"],
    "branch": ["branch", "branch name"],
    "statement_period": ["statement period", "statement date", "period", "transaction date from"],
}

GLM_OCR_PROMPT = (
    "You are an expert document OCR engine. Extract ALL text and tables from this image.\n\n"
    "CRITICAL RULES:\n"
    "1. Preserve EXACT text, numbers, and dates. Do not paraphrase.\n"
    "2. For tables: Use proper Markdown format with `|` separators and `|---|` header rows.\n"
    "3. Maintain original reading order.\n"
    "4. Output ONLY the extracted content in clean Markdown. No introductory text."
)


# 1. SMART DOCUMENT TYPE DETECTOR (FIXED FOR SCANNED PDFs)
def detect_document_mode(file_path: str) -> Dict[str, Any]:
    """Determines whether a file should use Native (DocVortex) or OCR (RapidOCR/GLM-OCR)."""
    ext = Path(file_path).suffix.lower()
    info = {
        "filename": Path(file_path).name,
        "extension": ext,
        "is_image": False,
        "is_native_pdf": False,
        "is_scanned_pdf": False,
        "is_spreadsheet": False,
        "requires_ocr": False,
        "recommended_engine": "docvortex",
        "avg_chars_per_page": 0.0,
    }

    if ext in {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}:
        info.update({"is_image": True, "requires_ocr": True, "recommended_engine": "ocr"})
        return info

    if ext in {".xlsx", ".xls", ".xlsm", ".csv", ".tsv", ".docx"}:
        info.update({"is_spreadsheet": ext != ".docx", "requires_ocr": False, "recommended_engine": "docvortex"})
        return info

    if ext == ".pdf":
        try:
            doc = pymupdf.open(file_path)
            sample_pages = min(3, len(doc))
            total_chars = sum(len((doc[i].get_text() or "").strip()) for i in range(sample_pages))
            doc.close()
            avg_chars = total_chars / max(1, sample_pages)
            info["avg_chars_per_page"] = round(avg_chars, 1)

            if avg_chars < 100:
                info.update({"is_scanned_pdf": True, "requires_ocr": True, "recommended_engine": "ocr"})
            else:
                info.update({"is_native_pdf": True, "requires_ocr": False, "recommended_engine": "docvortex"})
        except Exception:
            info.update({"is_scanned_pdf": True, "requires_ocr": True, "recommended_engine": "ocr"})

    return info

# 2. PAGE RENDERING & OCR ENGINES (RapidOCR + Optional Local Ollama GLM-OCR)
def _render_file_to_images(file_path: str, tmp_dir: Path, dpi: int = 200) -> List[Dict[str, Any]]:
    ext = Path(file_path).suffix.lower()
    pages: List[Dict[str, Any]] = []

    if ext == ".pdf":
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(file_path)
        scale = dpi / 72.0
        for idx in range(len(pdf)):
            pil_img = pdf[idx].render(scale=scale).to_pil()
            out_path = tmp_dir / f"page_{idx + 1:03d}.png"
            pil_img.save(out_path, "PNG", optimize=True)
            pages.append({"page_number": idx + 1, "path": str(out_path)})
            pil_img.close()
        pdf.close()
    elif ext in {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}:
        img = Image.open(file_path)
        out_path = tmp_dir / "page_001.png"
        img.convert("RGB").save(out_path, "PNG")
        img.close()
        pages.append({"page_number": 1, "path": str(out_path)})
    else:
        raise ValueError(f"Unsupported file type for OCR: {ext}")

    return pages


def _rapidocr_image_to_markdown(image_path: str) -> str:
    img = Image.open(image_path)
    process_path = image_path
    temp_resized = None

    if img.width > 1500:
        new_h = int(img.height * (1500 / img.width))
        resized = img.resize((1500, new_h), Image.Resampling.LANCZOS)
        temp_resized = str(Path(image_path).with_name(f"resized_{Path(image_path).name}"))
        resized.save(temp_resized, "PNG", optimize=True)
        resized.close()
        process_path = temp_resized
    img.close()

    try:
        result = run_ocr_normalized(process_path)
    finally:
        if temp_resized and os.path.exists(temp_resized):
            os.remove(temp_resized)

    if not result:
        return ""

    y_groups = defaultdict(list)
    for box, text, conf in result:
        y_key = int(box[0][1] // 12) * 12
        y_groups[y_key].append({"x_left": box[0][0], "text": str(text).strip()})

    lines = []
    for _, items in sorted(y_groups.items(), key=lambda x: x[0]):
        items.sort(key=lambda i: i["x_left"])
        if len(items) >= 3:
            lines.append("| " + " | ".join(i["text"] for i in items) + " |")
        else:
            lines.append(" ".join(i["text"] for i in items))

    return "\n".join(lines)


def _glm_ocr_image_to_markdown(image_path: str, model: str = "glm-ocr:q8_0") -> str:
    import ollama

    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.width > 1000:
        new_h = int(img.height * (1000 / img.width))
        img = img.resize((1000, new_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75, optimize=True)
    img.close()
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": GLM_OCR_PROMPT, "images": [b64]}],
        options={"temperature": 0.1, "num_predict": 4000, "num_ctx": 8192},
    )
    raw = response["message"]["content"]
    cleaned = re.sub(r"^```(?:markdown)?\s*\n", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"\n```\s*$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def _run_ocr_pipeline(file_path: str, use_glm_ocr: bool = False, glm_model: str = "glm-ocr:q8_0") -> Tuple[str, str]:
    with tempfile.TemporaryDirectory(prefix="doc_ocr_") as tmp:
        dpi = 150 if use_glm_ocr else 200
        pages = _render_file_to_images(file_path, Path(tmp), dpi=dpi)
        parts = []
        engine_used = f"OLLAMA_{glm_model}" if use_glm_ocr else "RAPIDOCR_ONNX"

        for p in pages:
            page_md = ""
            if use_glm_ocr:
                try:
                    page_md = _glm_ocr_image_to_markdown(p["path"], model=glm_model)
                except Exception:
                    # Graceful fallback to RapidOCR if Ollama server or glm-ocr model isn't running
                    page_md = _rapidocr_image_to_markdown(p["path"])
                    engine_used = "RAPIDOCR_ONNX_FALLBACK"
            else:
                page_md = _rapidocr_image_to_markdown(p["path"])

            parts.append(f"--- PAGE {p['page_number']} ---\n{page_md}")
            gc.collect()

    return "\n\n".join(parts), engine_used


# 3. NATIVE PIPELINE (DocVortex) WITH AUTOMATIC OCR FALLBACK
def _run_docvortex_pipeline(file_path: str) -> Tuple[str, Optional[dict]]:
    import docvortex

    with tempfile.TemporaryDirectory(prefix="doc_vortex_") as tmp:
        tmp_dir = Path(tmp)
        result = docvortex.parse(file_path, parse_mode="auto")

        md_path = tmp_dir / "document.md"
        result.export(str(md_path), output_format="markdown")
        markdown_content = md_path.read_text(encoding="utf-8")

        json_content = None
        try:
            json_path = tmp_dir / "document.json"
            result.export(str(json_path), output_format="structured_content")
            json_content = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            json_content = None
    markdown_content = _clean_docvortex_markdown(markdown_content)
    return markdown_content, json_content

def _clean_docvortex_markdown(md_text: str) -> str:
    """Strips DocVortex HTML image placeholder blocks and noisy bold punctuation."""
    if not md_text:
        return ""
    cleaned = re.sub(r"<details>[\s\S]*?</details>", "", md_text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\*\*([:,\-/\.]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

# 4. ROBUST MARKDOWN + HTML TABLE EXTRACTION (POLARS) & METADATA SEPARATION
def _split_md_row(line: str) -> List[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.replace(r"\|", "|").strip() for c in re.split(r"(?<!\\)\|", line)]


def _is_md_table_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and s.count("|") >= 2


def _is_separator_row(line: str) -> bool:
    cells = _split_md_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c.strip()) for c in cells)


def _html_table_to_markdown(html_table: str) -> str:
    html_table = html.unescape(html_table)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html_table, re.DOTALL | re.IGNORECASE)
    if not rows:
        return ""
    md_rows = []
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL | re.IGNORECASE)
        cleaned = [
            " ".join(re.sub(r"<[^>]+>", " ", c).split()).strip().replace("|", r"\|")
            for c in cells
        ]
        if cleaned:
            md_rows.append("| " + " | ".join(cleaned) + " |")
    if len(md_rows) < 2:
        return "\n".join(md_rows)
    num_cols = md_rows[0].count("|") - 1
    sep = "| " + " | ".join(["---"] * num_cols) + " |"
    return "\n".join([md_rows[0], sep] + md_rows[1:])


def _normalize_html_tables_in_markdown(markdown_text: str) -> str:
    return re.sub(
        r"<table[\s\S]*?</table>",
        lambda m: "\n" + _html_table_to_markdown(m.group(0)) + "\n",
        markdown_text,
        flags=re.IGNORECASE,
    )


def extract_tables_with_polars(markdown_text: str) -> List[Dict[str, Any]]:
    import polars as pl

    norm_md = _normalize_html_tables_in_markdown(markdown_text)
    lines = norm_md.splitlines()

    # Pass 1: standard separator-backed Markdown tables
    blocks: List[Dict[str, Any]] = []
    current: List[str] = []
    start_line = None

    for line_no, line in enumerate(lines, start=1):
        if _is_md_table_line(line):
            if not current:
                start_line = line_no
            current.append(line)
        else:
            if len(current) >= 2 and any(_is_separator_row(x) for x in current):
                blocks.append({"start_line": start_line, "end_line": line_no - 1, "lines": current})
            current, start_line = [], None

    if len(current) >= 2 and any(_is_separator_row(x) for x in current):
        blocks.append({"start_line": start_line, "end_line": len(lines), "lines": current})

    # Pass 2: fallback for pipe-delimited blocks without separator rows (from Notebook 2)
    if not blocks:
        current, start_line = [], None
        for line_no, line in enumerate(lines, start=1):
            if _is_md_table_line(line):
                if not current:
                    start_line = line_no
                current.append(line)
            else:
                if len(current) >= 2:
                    blocks.append({"start_line": start_line, "end_line": line_no - 1, "lines": current})
                current, start_line = [], None
        if len(current) >= 2:
            blocks.append({"start_line": start_line, "end_line": len(lines), "lines": current})

    all_tables: List[Dict[str, Any]] = []
    for t_idx, block in enumerate(blocks, start=1):
        t_lines = block["lines"]
        sep_indices = [i for i, ln in enumerate(t_lines) if _is_separator_row(ln)]
        if sep_indices and sep_indices[0] > 0:
            header_line = t_lines[sep_indices[0] - 1]
            data_lines = t_lines[sep_indices[0] + 1 :]
        else:
            header_line = t_lines[0]
            data_lines = t_lines[1:]

        raw_headers = _split_md_row(header_line)
        headers, seen = [], {}
        for i, h in enumerate(raw_headers, start=1):
            base = h.strip() if h and h.strip() else f"column_{i}"
            if base not in seen:
                seen[base] = 0
                headers.append(base)
            else:
                seen[base] += 1
                headers.append(f"{base}_{seen[base]}")

        rows = []
        for ln in data_lines:
            cells = _split_md_row(ln)
            if len(cells) < len(headers):
                cells += [None] * (len(headers) - len(cells))
            elif len(cells) > len(headers):
                cells = cells[: len(headers)]
            cleaned = [None if c is None or str(c).strip() == "" else str(c).strip() for c in cells]
            rows.append(dict(zip(headers, cleaned)))

        if not rows:
            continue

        df = pl.DataFrame(rows, schema={h: pl.Utf8 for h in headers})
        table_dicts = df.to_dicts()
        all_tables.append({
            "table_number": t_idx,
            "row_count": len(table_dicts),
            "column_count": len(headers),
            "columns": headers,
            "data": table_dicts,
        })

    return all_tables


def _clean_meta_text(value: Any) -> str:
    if value is None:
        return ""
    s = html.unescape(str(value))
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("**", "").replace("__", "").replace("|", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"^[\s:\-–=]+", "", s).strip()

def _extract_pdf_page1_visual_lines(file_path: Optional[str]) -> str: 
    """Reconstructs Page 1 horizontal lines by Y-coordinate so merged DocVortex cells don't break Key:Value rows."""
    if not file_path or Path(file_path).suffix.lower() != ".pdf":
        return ""
    try:
        rows = defaultdict(list)
        with pymupdf.open(file_path) as doc:
            if len(doc) == 0:
                return ""
            page_dict = doc[0].get_text("dict")
            for blk in page_dict.get("blocks", []):
                for line in blk.get("lines", []):
                    for span in line.get("spans", []):
                        txt = (span.get("text") or "").strip()
                        if not txt:
                            continue
                        x0, y0 = span["bbox"][0], span["bbox"][1]
                        y_key = int(round(y0 / 4.0)) * 4
                        rows[y_key].append((x0, txt))
        return "\n".join(
            " ".join(t[1] for t in sorted(rows[y], key=lambda item: item[0]))
            for y in sorted(rows.keys())
        )
    except Exception:
        return ""

def extract_financial_metadata(
    markdown_text: str,
    tables_data: List[Dict[str, Any]],
    file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Extracts bank/KYC metadata while blocking false positives and surviving merged table cells."""
    candidates = defaultdict(list)

    def _validate(field: str, val: str) -> Optional[str]:
        v = _clean_meta_text(val)
        if not v:
            return None
        up = v.upper()

        if field == "ifsc_code":
            m = re.search(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", up)
            return m.group(0) if m else None
        if field == "pan":
            m = re.search(r"\b[A-Z]{3}[PCHFATBLJG][A-Z][0-9]{4}[A-Z]\b", up)
            return m.group(0) if m else None
        if field == "account_number":
            m = re.search(r"\b\d{8,22}\b", v)
            return m.group(0) if m else None
        if field == "bank_name":
            if re.fullmatch(r"[\d,.\-+\s]+", v) or len(re.findall(r"[A-Za-z]", v)) < 3:
                return None
            for bank in KNOWN_BANKS:
                if bank in up:
                    return bank
            return v if 3 <= len(v) <= 60 else None
        if field == "branch":
            if any(bad in v.lower() for bad in ("account type", "account number", "currency", "statement date")):
                return None
            return v if 2 <= len(re.findall(r"[A-Za-z]", v)) and len(v) <= 60 else None
        if field == "account_holder":
            if any(w in v.lower() for w in ("withdrawal", "deposit", "neft", "rtgs", "imps", "upi", "balance")):
                return None
            return v if 2 <= len(re.findall(r"[A-Za-z]", v)) and len(v) <= 100 else None
        return v

    def _add(field: str, value: Any, conf: float):
        val = _validate(field, str(value))
        if val:
            candidates[field].append((conf, val))

    # Prefer geometric Page 1 lines for PDFs so merged table cells stay cleanly separated
    visual_p1 = _extract_pdf_page1_visual_lines(file_path)
    norm_md = _normalize_html_tables_in_markdown(markdown_text or "")
    combined_source = f"{visual_p1}\n{norm_md}" if visual_p1 else norm_md

    header_part = re.split(
        r"(?i)\b(?:balance\s+brought\s+forward|date\s+description\s+withdrawal)\b",
        combined_source,
        maxsplit=1,
    )[0]

    for line in header_part.splitlines():
        lc = _clean_meta_text(line)
        if not lc:
            continue
        up = lc.upper()
        for m in re.finditer(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", up):
            _add("ifsc_code", m.group(0), 0.90)
        for m in re.finditer(r"\b[A-Z]{3}[PCHFATBLJG][A-Z][0-9]{4}[A-Z]\b", up):
            _add("pan", m.group(0), 0.90)
        for bank in KNOWN_BANKS:
            if bank in up:
                _add("bank_name", bank, 0.85)
        if ":" in lc:
            k, v = lc.split(":", 1)
            k_low = re.sub(r"[^a-z0-9]+", " ", k.lower()).strip()
            matched = False
            for field, labels in FIELD_LABELS.items():
                if any(re.search(rf"\b{re.escape(lbl)}$", k_low) for lbl in labels):
                    if _add(field, v, 0.88 if visual_p1 else 0.82):
                        matched = True
                        break
            if not matched:
                for field, labels in FIELD_LABELS.items():
                    if any(re.search(rf"\b{re.escape(lbl)}\b", k_low) for lbl in labels):
                        if _add(field, v, 0.84 if visual_p1 else 0.80):
                            break

    final_meta = {}
    for field, items in candidates.items():
        items.sort(key=lambda x: x[0], reverse=True)
        final_meta[field] = items[0][1]

    return final_meta


def verify_extraction_against_markdown(tables_data: List[Dict[str, Any]], markdown_text: str) -> Dict[str, Any]:
    """Bulletproof verification from Notebook 2: confirms extracted JSON cells exist in Markdown."""
    md_norm = re.sub(r"[^A-Z0-9]", "", (markdown_text or "").upper())
    total_rows = 0
    verified_rows = 0

    for tbl in tables_data:
        cols = tbl.get("columns", [])
        rows = tbl.get("data", [])
        total_rows += len(rows)
        for row in rows:
            checked = 0
            ok = True
            for col in cols:
                val = row.get(col)
                if val is None:
                    continue
                vn = re.sub(r"[^A-Z0-9]", "", str(val).upper())
                if len(vn) < 3:
                    continue
                needle = vn[:15] if len(vn) > 20 else vn
                checked += 1
                if needle not in md_norm:
                    ok = False
                    break
            if checked > 0 and ok:
                verified_rows += 1

    pass_rate = round((verified_rows / max(1, total_rows)) * 100, 2) if total_rows else 100.0
    return {
        "total_tables": len(tables_data),
        "total_rows": total_rows,
        "verified_rows": verified_rows,
        "row_pass_rate": pass_rate,
    }

# 5. MAIN ENTRY POINT FOR YOUR LOAN ASSISTANT APP
def extract_universal_document(file_path: str, use_glm_ocr: bool = False) -> Dict[str, Any]:
    """
    Unified entry point combining Notebook 1 (DocVortex + RapidOCR/GLM-OCR)
    and Notebook 2 (Polars table parser + verification).
    """
    t0 = time.time()
    mode = detect_document_mode(file_path)
    json_content = None
    engine_used = "DOCVORTEX_NATIVE"

    if mode["requires_ocr"]:
        markdown_content, engine_used = _run_ocr_pipeline(file_path, use_glm_ocr=use_glm_ocr)
    else:
        try:
            markdown_content, json_content = _run_docvortex_pipeline(file_path)
        except Exception as exc:
            # Automatically recover from DocVortex 'pdf_ocr_unsupported' on scanned PDFs
            if "pdf_ocr_unsupported" in str(exc).lower() or "ocr" in str(exc).lower():
                markdown_content, engine_used = _run_ocr_pipeline(file_path, use_glm_ocr=use_glm_ocr)
            else:
                raise

    tables_data = extract_tables_with_polars(markdown_content)
    financial_metadata = extract_financial_metadata(markdown_content, tables_data, file_path=file_path)
    verification = verify_extraction_against_markdown(tables_data, markdown_content)

    return {
        "source_file": Path(file_path).name,
        "extraction_engine": engine_used,
        "processing_time_sec": round(time.time() - t0, 2),
        "detection": mode,
        "markdown_content": markdown_content,
        "financial_metadata": financial_metadata,
        "tables": tables_data,
        "structured_json": json_content,
        "verification": verification,
        "timestamp": datetime.now().isoformat(),
    }