import re


PURE_CURRENCY_RE = re.compile(
    r"^[₹$\s]*[+-]?(?:(?:\d{1,3}(?:,\d{2,3})+|\d+)\.\d{2}|\.\d{2})\s*(?:\((?:Cr|Dr)\)|(?:Cr|Dr)\b)?$",
    re.IGNORECASE,
)

def normalize_base_font_family(font_name: str) -> str:
    if not font_name:
        return "unknown"
    f = font_name.strip()
    if "+" in f:
        f = f.split("+", 1)[1]
    f = f.lower()
    f = re.sub(r"[-,_]*(bolditalic|boldoblique|bold|italic|oblique|regular|roman|medium|light|mt|psmt|ps)$", "", f)
    f = re.sub(r"[-,_]*(bold|italic|regular)$", "", f)
    return f.strip() or "unknown"

def run_pdf_forensics(doc, pages_spans):
    flags = []
    meta = doc.metadata or {}
    suspicious_tools = [
        "photoshop", "canva", "illustrator", "gimp", "inkscape", "coreldraw", "nitro", "sejda", "ilovepdf", "smallpdf"
    ]
    for k in ("producer", "creator", "author"):
        val = str(meta.get(k, "")).lower()
        for tool in suspicious_tools:
            if tool in val:
                flags.append(f"Page 1 Metadata Anomaly: PDF {k} contains editing toolchain trace '{meta.get(k)}'.")
    if len(doc) > 0:
        p1_txt = doc[0].get_text("text") or ""
        # Only flag truly unfilled standalone placeholders (not Standard Chartered's "<Real Value> (Company Name)" labels)
        placeholder_hits = re.findall(
            r"(?:^|\n)\s*[\(\[]\s*(?:Company\s*Name|Account\s*Name|Customer\s*Name|Holder\s*Name|Address|Branch\s*Name|Bank\s*Name)\s*[\)\]]\s*(?=\n|$)"
            r"|\[\s*(?:Insert|Enter|Your)\s+[^\]]+\]"
            r"|\b(?:Lorem\s+Ipsum|Sample\s+Company|Test\s+Account|Dummy\s+Bank)\b",
            p1_txt,
            re.IGNORECASE,
        )
        if placeholder_hits:
            uniq_ph = ", ".join(dict.fromkeys(h.strip() for h in placeholder_hits))
            flags.append(
                f"Page 1 Template Placeholder Anomaly: Unfilled template placeholders ({uniq_ph}) detected in header."
            )
    for p_idx, page_spans in enumerate(pages_spans):
        page_num = p_idx + 1
        currency_spans = [s for s in page_spans if PURE_CURRENCY_RE.match(s["text"].strip())]

        for i in range(len(currency_spans)):
            for j in range(i + 1, len(currency_spans)):
                s1, s2 = currency_spans[i], currency_spans[j]
                b1, b2 = s1["bbox"], s2["bbox"]
                ix0, iy0 = max(b1[0], b2[0]), max(b1[1], b2[1])
                ix1, iy1 = min(b1[2], b2[2]), min(b1[3], b2[3])
                if ix1 > ix0 + 2.0 and iy1 > iy0 + 2.0 and s1["text"].strip() != s2["text"].strip():
                    flags.append(
                        f"Page {page_num} Text Overlay Anomaly: Monetary span '{s1['text']}' overlaps with "
                        f"'{s2['text']}' at [{b1[0]}, {b1[1]}, {b1[2]}, {b1[3]}]."
                    )

        if len(currency_spans) >= 4:
            family_counts = {}
            for s in currency_spans:
                base_fam = normalize_base_font_family(s.get("font", ""))
                family_counts[base_fam] = family_counts.get(base_fam, 0) + 1

            dominant_family, dom_count = max(family_counts.items(), key=lambda x: x[1])
            total_curr = len(currency_spans)

            if dom_count / total_curr >= 0.80 and len(family_counts) > 1:
                for s in currency_spans:
                    base_fam = normalize_base_font_family(s.get("font", ""))
                    if base_fam != dominant_family and (family_counts[base_fam] / total_curr) <= 0.15:
                        b = s["bbox"]
                        flags.append(
                            f"Page {page_num} Font Anomaly: Monetary amount '{s['text']}' uses font family "
                            f"'{s.get('font')}' ({base_fam}) while statement numeric font is '{dominant_family}' "
                            f"at [{b[0]}, {b[1]}, {b[2]}, {b[3]}]."
                        )
    return flags

def run_docx_forensics(d):
    flags = []
    props = d.core_properties
    if props.revision and int(props.revision) > 3:
        flags.append(f"DOCX Metadata Anomaly: High revision count ({props.revision}) detected.")
    return flags