import os
import re
from typing import Dict, List, Any, Tuple
import numpy as np
import pymupdf
import docx

DATE_TOKEN_RE = re.compile(
    r"^(?:(0?[1-9]|[12]\d|3[01])[-/\s\.]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|[A-Za-z]{3,9})[-/\s\.,]*\d{2,4}"
    r"|(0?[1-9]|[12]\d|3[01])(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[-/\s\.]*\d{2,4}"
    r"|(0?[1-9]|[12]\d|3[01])[-/\.](0?[1-9]|1[0-2])[-/\.]\d{2,4}"
    r"|\d{4}[-/\.](0?[1-9]|1[0-2])[-/\.](0?[1-9]|[12]\d|3[01]))\b",
    re.IGNORECASE,
)

DECIMAL_NUMBER_RE = re.compile(
    r"^[₹$\s]*[+-]?(?:\d{1,3}(?:,\d{2,3})+|\d+)[\.,:\-]\d{2}(?:\s*\(?[CD]r\.?\)?)?$",
    re.IGNORECASE,
)

SUMMARY_KEYWORD_RE = re.compile(
    r"\b(?:statement\s+summary|transaction\s+total|opening\s+balance|closing\s+bal(?:ance)?|dr\s+count|cr\s+count|total\s+withdrawal|total\s+deposit|total\s+debit|total\s+credit)\b",
    re.IGNORECASE,
)

HEADER_CANDIDATE_RE = re.compile(
    r"\b(?:date|transactio|narration|particulars|description|remarks|instrument|chq|cheque|ref|withdrawal|withdrawals|withdrawl|deposit|deposits|debit|debits|credit|credits|amount|amt|balance)\b",
    re.IGNORECASE,
)


def _normalize_font_name(font_name: str) -> str:
    if not font_name:
        return "Unknown"
    if font_name.startswith("Type3"):
        return "Type3"
    if "+" in font_name:
        return font_name.split("+", 1)[1]
    return font_name


def _transform_rotated_bbox(
    rx0: float, ry0: float, rx1: float, ry1: float,
    raw_w: float, raw_h: float, line_dir: Tuple[float, float]
) -> Tuple[float, float, float, float, float, float]:
    d_cos, d_sin = float(line_dir[0]), float(line_dir[1])
    if abs(d_cos) < 0.5 and d_sin < -0.5:
        vx0 = max(0.0, raw_h - ry1)
        vx1 = max(vx0 + 1.0, raw_h - ry0)
        vy0 = rx0
        vy1 = rx1
        return vx0, vy0, vx1, vy1, raw_h, raw_w
    elif abs(d_cos) < 0.5 and d_sin > 0.5:
        vx0 = ry0
        vx1 = ry1
        vy0 = max(0.0, raw_w - rx1)
        vy1 = max(vy0 + 1.0, raw_w - rx0)
        return vx0, vy0, vx1, vy1, raw_h, raw_w
    return rx0, ry0, rx1, ry1, raw_w, raw_h


def _group_spans_into_lines(spans: List[Dict[str, Any]], y_tol_norm: float = 0.0055) -> List[List[Dict[str, Any]]]:
    sorted_spans = sorted(spans, key=lambda s: (s["cy_norm"], s["cx_norm"]))
    lines: List[List[Dict[str, Any]]] = []

    for sp in sorted_spans:
        placed = False
        for line in lines:
            avg_cy = sum(item["cy_norm"] for item in line) / len(line)
            if abs(sp["cy_norm"] - avg_cy) <= y_tol_norm:
                line.append(sp)
                placed = True
                break
        if not placed:
            lines.append([sp])

    for line in lines:
        line.sort(key=lambda s: s["bbox_points"][0])
    return lines


def _compute_numeric_x_clusters(
    spans: List[Dict[str, Any]],
    date_row_count: int = 0,
    cluster_tol_norm: float = 0.045,
) -> List[Dict[str, float]]:
    num_spans = [s for s in spans if DECIMAL_NUMBER_RE.match(s["text"])]
    if not num_spans:
        return []

    clusters: List[List[Dict[str, Any]]] = []
    for sp in sorted(num_spans, key=lambda s: s["bbox_norm"][2]):
        x1_norm = sp["bbox_norm"][2]
        placed = False
        for cl in clusters:
            avg_x1 = sum(item["bbox_norm"][2] for item in cl) / len(cl)
            if abs(x1_norm - avg_x1) <= cluster_tol_norm:
                cl.append(sp)
                placed = True
                break
        if not placed:
            clusters.append([sp])

    min_cluster_size = 1 if (date_row_count <= 4 or len(num_spans) <= 8) else 2
    valid_clusters = []
    for cl in clusters:
        if len(cl) >= min_cluster_size:
            x0_med = float(np.median([i["bbox_norm"][0] for i in cl]))
            x1_med = float(np.median([i["bbox_norm"][2] for i in cl]))
            cx_med = float(np.median([i["cx_norm"] for i in cl]))
            valid_clusters.append({
                "count": len(cl),
                "x_range_normalized": [round(x0_med, 4), round(x1_med, 4)],
                "cx_normalized": round(cx_med, 4),
                "x1_normalized": round(x1_med, 4),
            })

    valid_clusters.sort(key=lambda c: c["x1_normalized"])
    return valid_clusters


def _is_true_table_header_window(win_low: str, win_spans: List[Dict[str, Any]], date_rows_seen_so_far: int) -> bool:
    if date_rows_seen_so_far > 0:
        return False
    if any(
        bad in win_low
        for bad in (
            "opening balance", "statement summary", "dr count", "closing balance includes",
            "earmarked for hold", "contents of this statement", "within 30 days", "registered office",
            "from date", "to date"
        )
    ):
        return False
    if any(DATE_TOKEN_RE.match(s["text"].strip()) or DECIMAL_NUMBER_RE.match(s["text"].strip()) for s in win_spans):
        return False
    kw_hits = len(set(m.group(0).lower().rstrip("s") for m in HEADER_CANDIDATE_RE.finditer(win_low)))
    return kw_hits >= 3 and ("balance" in win_low or "amount" in win_low or "amt" in win_low)


def _scan_docx_structure(file_path: str) -> Dict[str, Any]:
    d = docx.Document(file_path)
    page_w = 595.0
    page_h = 842.0
    page_spans: List[Dict[str, Any]] = []
    y_cursor = 24.0

    for p in d.paragraphs:
        t = p.text.strip()
        if t:
            x0, y0, x1, y1 = 36.0, y_cursor, 550.0, y_cursor + 12.0
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            page_spans.append({
                "page": 1,
                "text": t,
                "bbox_points": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                "bbox_norm": [round(x0 / page_w, 4), round(y0 / page_h, 4), round(x1 / page_w, 4), round(y1 / page_h, 4)],
                "cx_points": round(cx, 2),
                "cy_points": round(cy, 2),
                "cx_norm": round(cx / page_w, 4),
                "cy_norm": round(cy / page_h, 4),
                "font": "DOCX_Native",
                "size": 10.0,
                "color": 0,
            })
            y_cursor += 16.0

    for table in d.tables:
        for row in table.rows:
            num_cells = max(1, len(row.cells))
            col_w = round((page_w - 60.0) / num_cells, 2)
            for c_idx, cell in enumerate(row.cells):
                ct = " ".join(cell.text.strip().splitlines()).strip()
                if ct:
                    x0 = round(30.0 + c_idx * col_w, 2)
                    x1 = round(x0 + col_w - 6.0, 2)
                    y0, y1 = round(y_cursor, 2), round(y_cursor + 12.0, 2)
                    cx, cy = round((x0 + x1) / 2.0, 2), round((y0 + y1) / 2.0, 2)
                    page_spans.append({
                        "page": 1,
                        "text": ct,
                        "bbox_points": [x0, y0, x1, y1],
                        "bbox_norm": [round(x0 / page_w, 4), round(y0 / page_h, 4), round(x1 / page_w, 4), round(y1 / page_h, 4)],
                        "cx_points": cx,
                        "cy_points": cy,
                        "cx_norm": round(cx / page_w, 4),
                        "cy_norm": round(cy / page_h, 4),
                        "font": "DOCX_Table",
                        "size": 10.0,
                        "color": 0,
                    })
            y_cursor += 18.0

    lines = _group_spans_into_lines(page_spans, y_tol_norm=0.004)
    header_line_idx = None
    date_row_count = 0
    summary_lines: List[str] = []

    for l_idx, ln in enumerate(lines):
        joined_ln = " ".join(s["text"] for s in ln)
        joined_low = joined_ln.lower()
        if SUMMARY_KEYWORD_RE.search(joined_ln):
            summary_lines.append(joined_ln)
        if any(DATE_TOKEN_RE.match(s["text"]) and s["cx_norm"] <= 0.35 for s in ln):
            date_row_count += 1
        if header_line_idx is None:
            if _is_true_table_header_window(joined_low, ln, date_row_count):
                header_line_idx = l_idx

    numeric_clusters = _compute_numeric_x_clusters(page_spans, date_row_count=date_row_count)
    cand_bal_cnt = numeric_clusters[-1]["count"] if numeric_clusters else date_row_count

    return {
        "file_path": file_path,
        "file_format": "DOCX",
        "total_pages": 1,
        "digital_pages": 1,
        "scanned_pages": 0,
        "total_candidate_balances": cand_bal_cnt,
        "pages": [{
            "page": 1,
            "width_points": page_w,
            "height_points": max(page_h, round(y_cursor + 20.0, 2)),
            "rotation": 0,
            "spans": page_spans,
            "lines": lines,
            "header_line_idx": header_line_idx,
            "date_row_count": date_row_count,
            "candidate_balance_count": cand_bal_cnt,
            "numeric_x_clusters": numeric_clusters,
            "has_summary_keywords": len(summary_lines) > 0,
            "summary_lines": summary_lines,
        }],
    }


def scan_pdf_structure(file_path: str, ocr_engine=None) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")
    if file_path.lower().endswith(".docx"):
        return _scan_docx_structure(file_path)

    pages_data: List[Dict[str, Any]] = []
    scanned_page_count = 0
    digital_page_count = 0
    doc_candidate_balances = 0

    with pymupdf.open(file_path) as doc:
        for p_idx, page in enumerate(doc):
            page_num = p_idx + 1
            rect = page.rect
            raw_w = max(1.0, float(rect.width))
            raw_h = max(1.0, float(rect.height))
            eff_page_w, eff_page_h = raw_w, raw_h
            rotation = int(page.rotation)

            page_spans: List[Dict[str, Any]] = []
            raw_text = page.get_text("text").strip()

            if len(raw_text) > 30:
                digital_page_count += 1
                for block in page.get_text("dict").get("blocks", []):
                    for line in block.get("lines", []):
                        line_dir = line.get("dir", (1.0, 0.0))
                        for span in line.get("spans", []):
                            raw_txt = span.get("text", "").strip()
                            if not raw_txt:
                                continue
                            rx0, ry0, rx1, ry1 = [float(v) for v in span["bbox"]]
                            bx0, by0, bx1, by1, eff_page_w, eff_page_h = _transform_rotated_bbox(
                                rx0, ry0, rx1, ry1, raw_w, raw_h, line_dir
                            )
                            sub_lines = [s_ln.strip() for s_ln in raw_txt.splitlines() if s_ln.strip()]
                            if not sub_lines:
                                continue
                            line_h = max(4.0, (by1 - by0) / len(sub_lines))
                            for s_idx, txt in enumerate(sub_lines):
                                x0 = round(bx0, 2)
                                x1 = round(bx1, 2)
                                y0 = round(by0 + s_idx * line_h, 2)
                                y1 = round(y0 + line_h, 2)
                                cx = round((x0 + x1) / 2.0, 2)
                                cy = round((y0 + y1) / 2.0, 2)
                                page_spans.append({
                                    "page": page_num,
                                    "text": txt,
                                    "bbox_points": [x0, y0, x1, y1],
                                    "bbox_norm": [
                                        round(x0 / eff_page_w, 4),
                                        round(y0 / eff_page_h, 4),
                                        round(x1 / eff_page_w, 4),
                                        round(y1 / eff_page_h, 4),
                                    ],
                                    "cx_points": cx,
                                    "cy_points": cy,
                                    "cx_norm": round(cx / eff_page_w, 4),
                                    "cy_norm": round(cy / eff_page_h, 4),
                                    "font": _normalize_font_name(span.get("font", "")),
                                    "size": round(float(span.get("size", 0.0)), 1),
                                    "color": span.get("color", 0),
                                })
            elif ocr_engine is not None:
                scanned_page_count += 1
                pix = page.get_pixmap(dpi=150)
                img_w = max(1.0, float(pix.width))
                img_h = max(1.0, float(pix.height))
                tmp_img = f"{file_path}_p{page_num}_scan.png"
                pix.save(tmp_img)
                try:
                    ocr_res, _ = ocr_engine(tmp_img)
                    if ocr_res:
                        for box, txt_ocr, conf in ocr_res:
                            txt = txt_ocr.strip()
                            if not txt:
                                continue
                            xs = [pt[0] for pt in box]
                            ys = [pt[1] for pt in box]
                            nx0, ny0 = min(xs) / img_w, min(ys) / img_h
                            nx1, ny1 = max(xs) / img_w, max(ys) / img_h
                            x0, y0 = round(nx0 * eff_page_w, 2), round(ny0 * eff_page_h, 2)
                            x1, y1 = round(nx1 * eff_page_w, 2), round(ny1 * eff_page_h, 2)
                            cx, cy = round((x0 + x1) / 2.0, 2), round((y0 + y1) / 2.0, 2)

                            page_spans.append({
                                "page": page_num,
                                "text": txt,
                                "bbox_points": [x0, y0, x1, y1],
                                "bbox_norm": [round(nx0, 4), round(ny0, 4), round(nx1, 4), round(ny1, 4)],
                                "cx_points": cx,
                                "cy_points": cy,
                                "cx_norm": round((nx0 + nx1) / 2.0, 4),
                                "cy_norm": round((ny0 + ny1) / 2.0, 4),
                                "font": "OCR",
                                "size": 10.0,
                                "color": 0,
                            })
                finally:
                    if os.path.exists(tmp_img):
                        os.remove(tmp_img)

            lines = _group_spans_into_lines(page_spans)
            page_spans.sort(key=lambda s: (s["cy_points"], s["cx_points"]))
            skip_sp = set()
            stitched_spans = []
            for idx_s, sp in enumerate(page_spans):
                if idx_s in skip_sp:
                    continue
                t1 = sp["text"].strip()
                # 1. Stitch wrapped date tails ('10/Apr/20' + '24' or '01/May/2' + '024')
                if sp["cx_norm"] <= 0.35 and re.match(r"^\d{1,2}[/\-][A-Za-z]{3}[/\-]20?$", t1, re.I):
                    for j_s in range(idx_s + 1, min(len(page_spans), idx_s + 12)):
                        if j_s in skip_sp:
                            continue
                        sp2 = page_spans[j_s]
                        if 0 < (sp2["cy_points"] - sp["cy_points"]) <= 16.0 and abs(sp2["cx_points"] - sp["cx_points"]) <= 24.0:
                            t2 = sp2["text"].strip()
                            if re.match(r"^\d{2,3}$", t2) and len(t1.split("/")[-1]) + len(t2) == 4:
                                sp["text"] = t1 + t2
                                skip_sp.add(j_s)
                                break
                # 2. Stitch wrapped monetary paise ('9,32,607.' + '00' or '10,11,062' + '.00')
                elif sp["cx_norm"] >= 0.55 and re.match(r"^[+-]?(?:\d{1,3}(?:,\d{2,3})+|\d{3,})\.?$", t1):
                    for j_s in range(idx_s + 1, min(len(page_spans), idx_s + 12)):
                        if j_s in skip_sp:
                            continue
                        sp2 = page_spans[j_s]
                        if 0 < (sp2["cy_points"] - sp["cy_points"]) <= 16.0 and abs(sp2["cx_points"] - sp["cx_points"]) <= 28.0:
                            t2 = sp2["text"].strip()
                            if (t1.endswith(".") and re.match(r"^\d{2}$", t2)) or ("," in t1 and re.match(r"^\.\d{2}$", t2)):
                                sp["text"] = t1.rstrip(".") + "." + t2.lstrip(".")
                                skip_sp.add(j_s)
                                break
                stitched_spans.append(sp)
            page_spans = stitched_spans
            header_line_idx = None
            date_row_count = 0
            summary_lines: List[str] = []

            for l_idx, ln in enumerate(lines):
                joined_ln = " ".join(s["text"] for s in ln)
                joined_low = joined_ln.lower()
                if SUMMARY_KEYWORD_RE.search(joined_ln):
                    summary_lines.append(joined_ln)

                if header_line_idx is None and date_row_count == 0:
                    win_spans = list(ln)
                    for k_off in range(1, 5):
                        if l_idx + k_off < len(lines):
                            nxt = lines[l_idx + k_off]
                            if any(DECIMAL_NUMBER_RE.match(s["text"]) for s in nxt):
                                break
                            win_spans.extend(nxt)
                    win_low = " ".join(s["text"].lower() for s in win_spans)
                    if _is_true_table_header_window(win_low, win_spans, date_row_count):
                        header_line_idx = l_idx

                if any(
                    (s["cx_norm"] <= 0.35 or s["bbox_norm"][0] <= 0.28)
                    and DATE_TOKEN_RE.match(s["text"])
                    and not re.search(r"\d{2,4}\s*to\s*\d{1,2}", s["text"], re.I)
                    for s in ln
                ):
                    if not any(
                        hk in joined_low
                        for hk in ("from date", "statement period", "for the period", "as on", "open date", "expiry date")
                    ):
                        if any(re.search(r"\d+\.\d{2}", s["text"]) for s in ln) or header_line_idx is not None:
                            date_row_count += 1

            numeric_clusters = _compute_numeric_x_clusters(page_spans, date_row_count=date_row_count)
            page_cand_balances = numeric_clusters[-1]["count"] if numeric_clusters else 0
            doc_candidate_balances += page_cand_balances

            pages_data.append({
                "page": page_num,
                "width_points": round(eff_page_w, 2),
                "height_points": round(eff_page_h, 2),
                "rotation": rotation,
                "spans": page_spans,
                "lines": lines,
                "header_line_idx": header_line_idx,
                "date_row_count": date_row_count,
                "candidate_balance_count": page_cand_balances,
                "numeric_x_clusters": numeric_clusters,
                "has_summary_keywords": len(summary_lines) > 0,
                "summary_lines": summary_lines,
            })

    file_format = (
        "HYBRID_PDF" if (scanned_page_count > 0 and digital_page_count > 0)
        else ("SCANNED_PDF" if scanned_page_count > 0 else "DIGITAL_PDF")
    )

    return {
        "file_path": file_path,
        "file_format": file_format,
        "total_pages": len(pages_data),
        "digital_pages": digital_page_count,
        "scanned_pages": scanned_page_count,
        "total_candidate_balances": doc_candidate_balances,
        "pages": pages_data,
    }