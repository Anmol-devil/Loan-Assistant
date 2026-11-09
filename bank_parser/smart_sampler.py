from typing import Dict, Any, List


def _format_line_with_dual_coords(line_spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Formats a single horizontal line with both point and normalized coordinates for the LLM."""
    y_pt = round(sum(s["cy_points"] for s in line_spans) / len(line_spans), 1)
    y_norm = round(sum(s["cy_norm"] for s in line_spans) / len(line_spans), 4)
    cells = []
    for s in line_spans:
        cells.append({
            "text": s["text"],
            "x_range_points": [s["bbox_points"][0], s["bbox_points"][2]],
            "x_range_normalized": [s["bbox_norm"][0], s["bbox_norm"][2]],
        })
    return {
        "y_points": y_pt,
        "y_normalized": y_norm,
        "spans": cells,
    }


def _clusters_differ(c1: List[Dict[str, float]], c2: List[Dict[str, float]], tol: float = 0.035) -> bool:
    """Returns True if two pages have different numeric column X-cluster layouts."""
    if not c1 or not c2:
        return False
    if len(c1) != len(c2):
        return True
    for a, b in zip(c1, c2):
        if abs(a["x1_normalized"] - b["x1_normalized"]) > tol:
            return True
    return False


def build_smart_llm_sample(scan_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    STAGE 1B: Selects high-signal structural samples across ALL pages (1..N)
    without dumping the entire multi-page PDF into the LLM context window.
    """
    pages = scan_data.get("pages", [])
    if not pages:
        return {}

    p1 = pages[0]
    p1_lines = p1["lines"]
    p1_hdr_idx = p1["header_line_idx"]

    top_cutoff = p1_hdr_idx if p1_hdr_idx is not None else min(25, len(p1_lines))
    page1_pre_table_text = "\n".join(
        " | ".join(s["text"] for s in ln) for ln in p1_lines[:top_cutoff]
    )
    page1_footer_text = "\n".join(
        " | ".join(s["text"] for s in ln) for ln in p1_lines[-18:]
    )

    table_start_sample = None
    baseline_clusters = None

    for p in pages:
        h_idx = p["header_line_idx"]
        if h_idx is not None or p["date_row_count"] >= 1:
            start_i = h_idx if h_idx is not None else 0
            sample_slice = p["lines"][max(0, start_i - 1) : min(len(p["lines"]), start_i + 15)]
            table_start_sample = {
                "page": p["page"],
                "page_dimensions_points": [p["width_points"], p["height_points"]],
                "header_line_detected": h_idx is not None,
                "numeric_x_clusters": p["numeric_x_clusters"],
                "sample_lines_with_coords": [_format_line_with_dual_coords(ln) for ln in sample_slice],
            }
            baseline_clusters = p["numeric_x_clusters"]
            break

    # Fallback to Page 1 if no page matched above
    if table_start_sample is None:
        sample_slice = p1_lines[: min(len(p1_lines), 20)]
        table_start_sample = {
            "page": p1["page"],
            "page_dimensions_points": [p1["width_points"], p1["height_points"]],
            "header_line_detected": False,
            "numeric_x_clusters": p1["numeric_x_clusters"],
            "sample_lines_with_coords": [_format_line_with_dual_coords(ln) for ln in sample_slice],
        }
        baseline_clusters = p1["numeric_x_clusters"]

    shifted_page_samples = []
    for p in pages[1:]:
        if p["date_row_count"] >= 1 and baseline_clusters:
            if _clusters_differ(baseline_clusters, p["numeric_x_clusters"]):
                h_i = p["header_line_idx"] or 0
                s_slice = p["lines"][h_i : min(len(p["lines"]), h_i + 6)]
                shifted_page_samples.append({
                    "page": p["page"],
                    "numeric_x_clusters": p["numeric_x_clusters"],
                    "sample_lines_with_coords": [_format_line_with_dual_coords(ln) for ln in s_slice],
                })
                if len(shifted_page_samples) >= 2:
                    break

    # 4. Collect Summary Table Blocks from ANY Page (Page 1, Middle Pages, or Last Page)
    summary_samples = []
    summary_kws = [
        "statement summary", "transaction total", "opening balance", "closing balance:",
        "closing bal", "dr count", "total withdrawals", "total deposits", "total debit", "total credit"
    ]
    for p in pages:
        if p.get("has_summary_keywords"):
            lines = p["lines"]
            matched_indices = [
                i for i, ln in enumerate(lines)
                if any(kw in " ".join(s["text"].lower() for s in ln) for kw in summary_kws)
                and "closing balance figure includes" not in " ".join(s["text"].lower() for s in ln)
                and "closing balance includes" not in " ".join(s["text"].lower() for s in ln)
            ]
            if matched_indices:
                first_m = matched_indices[0]
                last_m = matched_indices[-1]
                win_start = max(0, first_m - 2)
                win_end = min(len(lines), max(first_m + 12, last_m + 6))
                snippet_lines = [
                    " | ".join(s["text"] for s in ln)
                    for ln in lines[win_start:win_end]
                ]
                summary_samples.append({
                    "page": p["page"],
                    "summary_text_block": "\n".join(snippet_lines),
                })
                
    last_p = pages[-1]
    if not any(s["page"] == last_p["page"] for s in summary_samples):
        last_lines = [" | ".join(s["text"] for s in ln) for ln in last_p["lines"][-20:]]
        summary_samples.append({
            "page": last_p["page"],
            "summary_text_block": "\n".join(last_lines),
        })

    return {
        "total_pages": scan_data.get("total_pages", 1),
        "file_format": scan_data.get("file_format", "DIGITAL_PDF"),
        "page1_pre_table_header": page1_pre_table_text,
        "page1_bottom_footer": page1_footer_text,
        "primary_table_sample": table_start_sample,
        "shifted_continuation_samples": shifted_page_samples,
        "summary_blocks_across_pages": summary_samples,
    }