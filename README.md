<!-- # Loan Assistant — Module 1: AI Document Checker & Risk Assessment

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

**Project:** Loan Assistant | **Module:** 1 — AI Document Checker & Risk Assessment -->











# Loan Assistant — AI Document Checker & Risk Assessment

**An intelligent document verification and financial reconciliation system for loan application processing.**

**Module:** 1 — Document Checker & Risk Calibration  
**Application:** Python • Streamlit • OCR • LLM • Computer Vision

---

## 1. Overview

Loan processing involves reviewing multiple documents, including PAN cards, Aadhaar cards, bank statements, salary slips, GST certificates, ITR returns and financial statements.

Manual verification can be time-consuming and may overlook:

- Inconsistent applicant information.
- Incorrect financial calculations.
- Missing or incorrectly extracted fields.
- Suspicious image alterations.
- Unexpected PDF structural changes.
- Differences between information across submitted documents.

**Loan Assistant** is designed to automate initial document analysis, extract structured information, perform deterministic verification checks, identify potential anomalies and generate evidence-based risk assessment reports.

The system combines Python-based validation, OCR, image processing and LLM-assisted document understanding.

It is designed as a decision-support system. Its findings are not definitive proof of fraud or document authenticity, and final decisions require appropriate verification and human review.

## 2. Key Features

### Document Processing
- Upload PDF, PNG, JPG, JPEG and DOCX files.
- Identify document types.
- Extract text and structured information using OCR and document parsing.
- Process digital and scanned PDF documents.
- Extract document-specific fields.

### Financial Verification
- Bank transaction extraction.
- Opening and closing balance identification.
- Credit and debit calculation.
- Running balance verification.
- Ledger total reconciliation.
- Financial consistency checks.

### KYC Verification
- PAN card information extraction.
- Aadhaar information extraction.
- PAN format validation.
- Name and date-of-birth extraction.
- Aadhaar information masking.
- Portrait-region analysis.
- Suspicious portrait and image anomaly detection.

### Document Forensics
- PDF structural inspection.
- Metadata and font analysis.
- Image manipulation indicators.
- Portrait and image-region analysis.
- Cross-document consistency checks.

### Risk Assessment
- Structured findings with supporting evidence.
- Verification status for individual checks.
- Risk scoring and triage classification.
- Manual-review recommendations.

## 3. Supported Document Categories

The application interface includes the following document categories. Actual verification coverage depends on the parser and validation rules implemented for each document type.

| Document | Primary verification objectives |
|---|---|
| PAN Card | PAN format, applicant details and portrait analysis |
| Aadhaar Card | KYC information, privacy masking and portrait analysis |
| Bank Statement | Transactions, balances and financial reconciliation |
| Salary Slip | Salary components and income consistency |
| GST Certificate | Registration information and business identity |
| Udyam / MSME | Business registration information |
| ITR Return | Income and tax-related information |
| Balance Sheet | Assets, liabilities and financial consistency |

Additional verification coverage should be tested before using each document category in a production lending workflow.

## 4. System Architecture

The application uses four verification families.

| Family | Name | Responsibility |
|---|---|---|
| A | Logic & Arithmetic Verification | Checks calculations, financial totals, formats and logical consistency |
| B | Structural Document Forensics | Examines PDF/DOCX metadata, fonts, text overlays and document structure |
| C | Image Forensics | Examines scanned documents and images for potential visual manipulation |
| D | Cross-Document Verification | Compares identity and financial information across submitted documents |

### Processing Pipeline

```text
               DOCUMENT UPLOAD
                      |
                      v
              SECURITY GATE
                      |
                      v
          DOCUMENT IDENTIFICATION
                      |
                      v
         OCR / TEXT / LAYOUT EXTRACTION
                      |
                      v
          LLM-ASSISTED UNDERSTANDING
                      |
                      v
           PYTHON VALIDATION ENGINE
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       FAMILY A    FAMILY B    FAMILY C
       Logic       Structure   Image
          |           |           |
          +-----------+-----------+
                      |
                      v
           FAMILY D — CROSS-DOCUMENT
                      |
                      v
             RISK AGGREGATION
                      |
                      v
           STRUCTURED FINDINGS
                      |
                      v
              FINAL REPORT
```

### LLM and Python Responsibilities

The application separates intelligent document understanding from deterministic verification.

**LLM responsibilities:**
- Document interpretation.
- Layout understanding.
- Field identification and extraction assistance.
- Classification of ambiguous content.
- Natural-language summaries.

**Python responsibilities:**
- Data extraction and normalization.
- Required-field validation.
- PAN and other format checks.
- Date and amount validation.
- Financial calculations.
- Running balance reconciliation.
- Document consistency checks.
- Risk aggregation and structured findings.

The LLM does not independently establish document authenticity or determine the correctness of financial calculations.

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application and verification logic |
| Streamlit | User interface |
| PyMuPDF | PDF extraction and analysis |
| RapidOCR | Optical character recognition |
| OpenCV | Image preprocessing and forensic analysis |
| NumPy | Image and numerical processing |
| Pillow | Image handling |
| LLM integration | Document interpretation and structured extraction |
| ONNX-compatible models | Local model inference where configured |
| Git & GitHub | Version control |

Dependencies are maintained in `requirements.txt`.

## 6. Project Structure

```text
Loan-Assistant/
│
├── app.py
│   └── Streamlit user interface and processing workflow
│
├── bank_parser/
│   └── Bank statement extraction and reconciliation
│
├── kyc_spoof_watchlist/
│   └── Local reference images for configured spoof checks
│
├── corporate_kyc_engines.py
│   └── KYC and financial-document verification
│
├── cross_document_verifier.py
│   └── Cross-document verification (Family D)
│
├── forensic_aggregator.py
│   └── Aggregation of forensic findings
│
├── module_aggregator.py
│   └── Module 1 output and risk aggregation
│
├── multiformated_reconciliation.py
│   └── Document extraction and reconciliation
│
├── recocile_spans.py
│   └── Span reconciliation utilities
│
├── requirements.txt
│   └── Python dependencies
│
├── .gitignore
│   └── Excluded files and sensitive artifacts
│
└── README.md
```

## 7. Installation and Setup

### Prerequisites

- Python 3.10 or newer.
- Git.
- Internet access for initial dependency and model downloads.
- Configured LLM access if LLM-assisted processing is enabled.
- Suitable hardware for the selected OCR and inference models.

### Step 1 — Clone the repository

```bash
git clone https://github.com/Anmol-devil/Loan-Assistant.git
cd Loan-Assistant
```

### Step 2 — Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4 — Configure LLM integration

Configure the required model provider, endpoint, credentials and model settings according to the application's configuration.

Never hardcode API keys or commit credentials to GitHub.

### Step 5 — Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed by Streamlit, normally:

```text
http://localhost:8501
```

## 8. Risk Assessment

The application generates three primary triage categories.

| Triage band | Meaning |
|---|---|
| LOW RISK | No configured significant anomalies detected |
| NEEDS REVIEW | Suspicious or uncertain findings require investigation |
| HIGH RISK | Significant configured anomalies detected |

The risk score is generated from configured verification rules and findings.

A low-risk result does not establish that a document is genuine. Missing, unexecuted or unsupported checks must not be interpreted as successful verification.

### Verification Statuses

| Status | Meaning |
|---|---|
| PASSED | Verification completed successfully |
| FAILED | A defined validation condition failed |
| FLAGGED | Suspicious evidence was identified |
| NOT_RUN | The verification was not executed |
| NOT_APPLICABLE | The check does not apply |
| NOT_VERIFIED | Insufficient evidence to establish the result |

The application should distinguish extraction confidence from verification confidence.

## 9. Security and Privacy

The application is intended for controlled processing of sensitive financial and KYC documents.

Important security considerations:

- Upload only documents authorized for processing.
- Protect customer documents and extracted personal information.
- Mask sensitive identifiers in generated reports and logs.
- Encrypt stored documents and sensitive application data.
- Restrict access through appropriate authentication and authorization.
- Never commit API keys, credentials or customer documents to GitHub.
- Define retention and deletion policies.
- Use consent-based processing and appropriate audit logging.

**LLM deployment note:** The privacy of document processing depends on the configured inference endpoint. If an external LLM provider is used, extracted document information or prompts may leave the company's infrastructure. Strict on-premise operation requires an appropriately secured self-hosted model and verification of network data flows.

## 10. Limitations and Important Notes

- OCR accuracy depends on document quality, language, resolution and layout.
- Different banks and document formats may require separate extraction rules.
- Image-forensic indicators are not conclusive proof of manipulation.
- Face similarity does not prove document authenticity.
- Official KYC verification requires appropriate authorized verification mechanisms.
- Financial and KYC anomalies require supporting evidence and, where appropriate, human review.
- Risk thresholds must be evaluated against labelled genuine and manipulated document samples.

This project is an automated document analysis and risk-assessment tool, not a standalone loan approval or fraud determination system.

## 11. Development Priorities

Planned development and validation areas include:

- Improve Aadhaar QR and digital-signature verification.
- Improve bank statement extraction across different bank formats.
- Add independent field-completeness and layout validation.
- Improve portrait manipulation and anti-spoofing checks.
- Expand cross-document verification.
- Evaluate privately hosted LLM inference.
- Improve risk calibration and false-positive testing.
- Strengthen security, performance and auditability.

## 12. Contributing

Contributions, bug reports and improvements are welcome.

Before submitting changes:

1. Test document extraction and verification changes.
2. Avoid including real customer documents or personal data.
3. Never commit credentials or private keys.
4. Include supporting test cases where possible.
5. Document new verification rules and their limitations.

## 13. Disclaimer

Loan Assistant is an automated document analysis and decision-support prototype.

Its findings are intended to assist authorized reviewers. Automated checks may produce false positives or false negatives. The system does not independently guarantee document authenticity, establish fraud, or make final lending decisions.

All important findings should be evaluated using supporting evidence and appropriate verification procedures.

---

**Project:** Loan Assistant  
**Module:** 1 — AI Document Checker & Risk Assessment  
**Repository:** https://github.com/Anmol-devil/Loan-Assistant