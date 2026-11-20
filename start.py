import os
import subprocess
import sys

os.environ.setdefault("HOME", "/tmp")
port = os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT", "9000")

cmd = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    "app.py",
    f"--server.port={port}",
    "--server.address=0.0.0.0",
    "--server.headless=true",
    "--server.enableCORS=false",
    "--server.enableXsrfProtection=false",
]

sys.exit(subprocess.call(cmd))