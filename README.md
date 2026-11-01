Loan Assistant — Module 1
Module 1 of the Loan Assistant project analyzes uploaded financial and KYC documents.
Main components
- app.py — Streamlit application interface.
- multiformated_reconciliation.py — document extraction and financial reconciliation.
- corporate_kyc_engines.py — KYC and financial-document verification.
- cross_document_verifier.py — checks consistency across uploaded documents.
- forensic_aggregator.py — combines document-forensics checks.
- module_aggregator.py — combines Module 1 analysis results.
Setup
Use Python 3.10 or newer. Create and activate a virtual environment, then install dependencies:
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
Run
streamlit run app.py
Upload only documents you are authorized to process. Do not commit API keys, credentials, or private customer documents to GitHub.
Results are automated review signals, not proof that a document is genuine or fraudulent. Findings should be reviewed with the supporting evidence.
