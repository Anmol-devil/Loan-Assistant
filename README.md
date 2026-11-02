# Loan Assistant — Module 1: AI Document Checker & Risk Assessment

An automated system for analyzing financial and KYC documents, detecting inconsistencies, and identifying potential document tampering.

## 1. Problem Statement

During loan processing, applicants submit multiple documents such as PAN cards, bank statements, salary slips, GST certificates, ITRs, and balance sheets.

Manually checking these documents is time-consuming. Financial values may not match, information may be inconsistent across documents, and documents may contain suspicious alterations.

**Our goal is to automate the initial document verification process, detect potential errors and tampering, and help reviewers identify documents that need manual investigation.**

## 2. What Are We Building?

Module 1 analyzes uploaded documents and:

- Identifies the document type and extracts important information.
- Checks financial calculations and transaction balances.
- Detects suspicious document structure, metadata, and image anomalies.
- Compares information across multiple documents.
- Generates structured findings with supporting evidence.
- Assigns risk-based triage bands: LOW RISK, NEEDS REVIEW, and HIGH RISK.

The system provides automated review signals, not definitive proof of fraud or genuineness.

## 3. System Architecture

The analysis is organized into four detection families:

| Family | Responsibility |
|---|---|
| **A — Logic & Arithmetic** | Checks financial calculations, balances, totals, and document consistency. |
| **B — Structural Forensics** | Examines PDF/DOCX metadata, fonts, text overlays, and structural anomalies. |
| **C — Image Forensics** | Examines scanned documents and images for potential visual manipulation. |
| **D — Cross-Document Verification** | Compares identity and financial information across submitted documents. |

### Processing Flow

**Document Upload → Document Identification → Data Extraction → Families A–D Verification → Risk Aggregation → Results & Evidence**

## 4. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development and verification logic |
| Streamlit | Web application interface |
| PyMuPDF | PDF text extraction and page analysis |
| DOCX processing tools | Word document extraction |
| Image-processing tools | Image analysis and forensic checks |
| Rule-based validation | Arithmetic and document verification |
| Git & GitHub | Version control and source-code management |

Dependencies are listed in `requirements.txt`.

## 5. Project Structure

| File | Responsibility |
|---|---|
| `app.py` | Streamlit user interface |
| `multiformated_reconciliation.py` | Document extraction and financial reconciliation |
| `corporate_kyc_engines.py` | KYC and financial-document verification |
| `cross_document_verifier.py` | Cross-document checks (Family D) |
| `forensic_aggregator.py` | Combines forensic analysis results |
| `module_aggregator.py` | Combines findings and generates risk assessment |

## 6. Setup & Run

**Requirements:** Python 3.10 or newer.

```bash
git clone https://github.com/Anmol-devil/Loan-Assistant.git
cd Loan-Assistant

python -m venv .venv
.venv\Scripts\activate

python -m pip install -r requirements.txt
streamlit run app.py
```

## 7. Important Notes

- Upload only documents you are authorized to process.
- Never commit API keys, credentials, or private customer documents to GitHub.
- A LOW RISK result means no configured anomalies were detected; it does not prove genuineness.
- All important findings should be reviewed with their supporting evidence.

**Project:** Loan Assistant | **Module:** 1 — AI Document Checker & Risk Assessment
