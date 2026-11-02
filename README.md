# Loan Assistant — Module 1: AI Document Checker & Risk Assessment

An automated financial and KYC document analysis system that extracts information, validates financial consistency, identifies potential document anomalies, and generates evidence-based review signals.

## 1. Problem Statement

Financial institutions process large volumes of financial and KYC documents during loan applications. These documents may contain missing information, inconsistent financial figures, altered transaction values, structural irregularities, or mismatches with other submitted documents.

Manually reviewing every document is time-consuming, repetitive, and vulnerable to human error. Furthermore, checking individual documents in isolation may not reveal inconsistencies between documents submitted by the same applicant.

**The problem we aim to solve is how to automate the initial analysis and verification of financial and KYC documents, identify inconsistencies and potential tampering, and help reviewers prioritize documents that require further investigation.**

## 2. Project Objective

Module 1 aims to build a document intelligence and verification system that:

- Automatically identifies supported financial and KYC document types.
- Extracts relevant information from PDF, DOCX, and image-based documents.
- Validates financial calculations, transaction records, and accounting relationships.
- Detects suspicious structural, metadata, and image-level anomalies.
- Cross-checks information across multiple uploaded documents.
- Generates structured findings with supporting evidence.
- Assigns document review-priority bands to support manual verification.

The system assists document reviewers; it does not independently establish whether a document is genuine or fraudulent.

## 3. Supported Document Types

The system is designed to analyze the following document categories:

| Category | Document |
|---|---|
| Identity | PAN Card |
| Banking | Bank Statement |
| Income | Salary Slip |
| Business Registration | GST Certificate |
| Business Registration | Udyam Certificate |
| Taxation | Income Tax Return (ITR) |
| Financial Statements | Balance Sheet |

**Supported upload formats:** PDF, PNG, JPG, JPEG, and DOCX.

Actual extraction and verification coverage depends on the document format and the checks implemented for each category.

## 4. Core Detection Architecture

The verification system organizes its analysis into four detection families.

### Family A — Arithmetic, Logic & Schema Validation

Checks the internal consistency of extracted document information.

Examples:
- Opening balance, credits, debits, and closing balance reconciliation.
- Bank statement running-balance verification.
- Transaction amount and ledger-total validation.
- Financial statement arithmetic checks.
- Missing or invalid mandatory fields.
- Document-specific logical and schema validation.

### Family B — Physical & Structural Forensics

Examines document structure and metadata for unusual characteristics.

Examples:
- PDF metadata inspection.
- Font and text-structure analysis.
- Text overlay and structural anomaly detection.
- DOCX metadata and revision information.
- Document-specific structural consistency checks.

Metadata anomalies are review signals and are not, by themselves, proof of tampering.

### Family C — Image Forensics

Examines raster images and scanned documents for potential visual manipulation.

Examples:
- Error Level Analysis (ELA), where applicable.
- Image-level inconsistency checks.
- Visual anomaly detection.
- Suspicious regions requiring manual inspection.

Some image-forensics techniques are not applicable to born-digital vector PDFs. The system should report these checks as `NOT_APPLICABLE` rather than treating them as passed.

### Family D — Cross-Document Verification

Compares information across multiple documents submitted together.

Examples:
- Applicant or business identity consistency.
- PAN and name consistency.
- Income information comparison.
- Business information consistency.
- Financial value and reporting-period comparisons.

This family requires multiple relevant documents. When the required documents are unavailable, its status should be `NOT_RUN` or `NOT_APPLICABLE`, as appropriate.

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core application and verification logic |
| Streamlit | Interactive web application and document-upload interface |
| PyMuPDF | PDF text extraction, page inspection, and layout analysis |
| DOCX processing | Reading and extracting information from Word documents |
| OCR processing | Extracting text from scanned documents and images, where configured |
| Image-processing tools | Image inspection and forensic analysis, where configured |
| Rule-based validation | Financial calculations, logical checks, and document-specific verification |
| Git & GitHub | Source-code management and version control |

The exact supporting libraries and versions are maintained in `requirements.txt`.

## 6. Application Architecture

The application is organized into separate modules.

| File | Responsibility |
|---|---|
| `app.py` | Streamlit application interface, uploads, and result presentation |
| `multiformated_reconciliation.py` | Document extraction, financial reconciliation, and document-level checks |
| `corporate_kyc_engines.py` | KYC and financial-document verification |
| `cross_document_verifier.py` | Cross-document consistency verification (Family D) |
| `forensic_aggregator.py` | Combines physical, structural, and image-forensics results |
| `module_aggregator.py` | Combines verification results, structured findings, and triage outputs |

### Processing Workflow

1. **Document Upload:** The user uploads one or more financial or KYC documents.
2. **Document Identification:** The system determines the document category and relevant entity information.
3. **Information Extraction:** Text, tables, financial values, transaction records, and metadata are extracted.
4. **Document Verification:** Relevant checks are performed across Families A, B, C, and D.
5. **Finding Generation:** Detected inconsistencies are converted into structured findings with available evidence.
6. **Risk Aggregation:** Findings and detector results are combined into a document review-priority assessment.
7. **Result Presentation:** The application displays extracted information, reconciliation results, findings, and triage bands.

## 7. Risk Assessment & Triage

The system combines configured detection signals to help prioritize documents for manual review.

The intended triage categories are:

| Triage Band | Meaning |
|---|---|
| LOW RISK | No anomalies detected by the checks that were successfully performed |
| NEEDS REVIEW | One or more inconsistencies or suspicious signals require investigation |
| HIGH RISK | Significant configured anomalies require priority review |

Detector coverage is important when interpreting these results:

- `PASSED` — The check was performed and passed.
- `FAILED` — The check was performed and detected an inconsistency.
- `NOT_RUN` — The check was not performed.
- `NOT_APPLICABLE` — The check does not apply to that document.
- `COULD_NOT_CHECK` — The check could not be completed reliably.

A review-priority score must not be interpreted as a probability of fraud unless it has been calibrated and validated using a representative, independently labelled dataset.

## 8. Project Setup

### Prerequisites

- Python 3.10 or newer
- Git
- pip

### Clone the Repository

```bash
git clone https://github.com/Anmol-devil/Loan-Assistant.git
cd Loan-Assistant
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

### Activate the Environment (Windows)

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 9. Testing & Validation

The verification system should be evaluated using both original documents and controlled test copies.

Examples of validation scenarios:

- Change a transaction debit or credit and check whether reconciliation detects it.
- Modify a closing balance and verify the running-balance checks.
- Introduce inconsistent applicant information across documents.
- Test documents from different banks and document layouts.
- Check incomplete, malformed, or unsupported document extractions.
- Verify that unavailable checks are not incorrectly marked as passed.

Regression tests should cover multiple document types, layouts, and file formats.

## 10. Security & Responsible Use

- Upload only documents you are authorized to process.
- Do not commit API keys, passwords, credentials, or private customer documents to GitHub.
- Keep sensitive documents out of source-control history.
- Treat automated findings as review signals, not definitive fraud determinations.
- Preserve supporting evidence for findings wherever available.
- Require human review before making consequential lending or fraud decisions.

## 11. Project Scope & Limitations

This project focuses on automated document extraction, internal verification, forensic analysis, and cross-document consistency checking.

Its accuracy depends on document quality, extraction completeness, supported layouts, and the coverage of implemented verification rules.

A successful check does not prove that a document is genuine. Similarly, a failed check may result from extraction errors, unsupported formatting, or actual document inconsistencies. Findings must therefore be interpreted with their supporting evidence.

---

**Project:** Loan Assistant  
**Module:** Module 1 — AI Document Checker & Risk Assessment  
**Interface:** Streamlit  
**Language:** Python  
**Repository:** https://github.com/Anmol-devil/Loan-Assistant
