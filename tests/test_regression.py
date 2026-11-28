import argparse
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List

import pymupdf

# Ensure project root is on sys.path when running from anywhere
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.doc_extractor import detect_and_extract_document

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baseline.json")

# Put your working test PDFs here (paths relative to project root)
SAMPLE_FILES: List[str] = [
    "Bank_statement_FY_2024-2025 Axis Bank.pdf",
    "INDUSIND BANK 3.pdf",
    # Add your ICICI, J&K Bank, PAN, and unlocked e-Aadhaar filenames below:
    # "ICICI_39_page.pdf",
    # "JK_Bank_Statement.pdf",
]


def _extract_metrics(file_path: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    res = detect_and_extract_document(file_path, use_llm=False)
    elapsed = round(time.perf_counter() - t0, 3)

    det = res.get("detection") or {}
    raw = res.get("raw_report") or {}
    m1 = res.get("module1_contract") or {}
    fin = raw.get("financial_reconciliation") or {}
    txs = fin.get("transactions") or []
    flags = raw.get("forensic_flags") or m1.get("findings") or []

    return {
        "status": res.get("status", "OK"),
        "doc_code": det.get("doc_code"),
        "layout_variant": det.get("layout_variant"),
        "triage_band": m1.get("triage_band"),
        "holder_name": fin.get("holder_name") or (m1.get("identity_card") or {}).get("holder_name"),
        "tx_count": len(txs),
        "opening_balance": fin.get("opening_balance_used") if fin.get("opening_balance_used") is not None else fin.get("opening_balance"),
        "closing_balance": fin.get("closing_balance"),
        "forensic_flag_count": len(flags),
        "elapsed_sec": elapsed,
    }


def _test_password_guard() -> None:
    tmp = os.path.join(tempfile.gettempdir(), "_regression_locked_test.pdf")
    try:
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), "Protected Bank Statement")
        doc.save(tmp, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="1234", owner_pw="admin")
        doc.close()

        t0 = time.perf_counter()
        res = detect_and_extract_document(tmp, use_llm=False)
        elapsed = round(time.perf_counter() - t0, 3)

        assert res.get("status") == "PASSWORD_PROTECTED", f"Expected PASSWORD_PROTECTED, got {res.get('status')}"
        assert elapsed < 1.5, f"Password guard too slow: {elapsed}s"
        print(f"[PASS] Password-protected PDF guard ({elapsed}s)")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save current outputs as golden baseline")
    args = parser.parse_args()

    _test_password_guard()

    existing_files = [
        fn for fn in SAMPLE_FILES if os.path.exists(os.path.join(ROOT_DIR, fn))
    ]
    if not existing_files:
        print("[WARN] No files from SAMPLE_FILES found in project root. Edit SAMPLE_FILES in tests/test_regression.py.")
        return

    current: Dict[str, Dict[str, Any]] = {}
    for fn in existing_files:
        full_path = os.path.join(ROOT_DIR, fn)
        metrics = _extract_metrics(full_path)
        current[fn] = metrics
        print(
            f"[RUN ] {fn} -> {metrics['doc_code']} | rows={metrics['tx_count']} | "
            f"open={metrics['opening_balance']} | close={metrics['closing_balance']} | "
            f"triage={metrics['triage_band']} | flags={metrics['forensic_flag_count']} | {metrics['elapsed_sec']}s"
        )

    if args.save:
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        print(f"\nSaved golden baseline to {BASELINE_PATH}")
        return

    if not os.path.exists(BASELINE_PATH):
        print("\n[FAIL] No baseline.json found. Run: python tests/test_regression.py --save")
        sys.exit(1)

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    failures = []
    strict_keys = (
        "status",
        "doc_code",
        "triage_band",
        "holder_name",
        "tx_count",
        "opening_balance",
        "closing_balance",
        "forensic_flag_count",
    )

    for fn, base in baseline.items():
        if fn not in current:
            failures.append(f"{fn}: missing from current run")
            continue
        now = current[fn]
        for k in strict_keys:
            if base.get(k) != now.get(k):
                failures.append(f"{fn} [{k}]: baseline={base.get(k)!r} vs now={now.get(k)!r}")

        # Performance regression guard: allow +25% + 0.5s OS jitter
        max_allowed = round(base["elapsed_sec"] * 1.25 + 0.5, 3)
        if now["elapsed_sec"] > max_allowed:
            failures.append(
                f"{fn} [SPEED]: slowed down from {base['elapsed_sec']}s to {now['elapsed_sec']}s (limit {max_allowed}s)"
            )

    if failures:
        print("\n[REGRESSION FAILED]")
        for msg in failures:
            print("  -", msg)
        sys.exit(1)

    print("\n[ALL REGRESSION & PERFORMANCE CHECKS PASSED]")


if __name__ == "__main__":
    main()