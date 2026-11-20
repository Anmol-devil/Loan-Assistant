import os
import subprocess
import sys

os.environ.setdefault("HOME", "/tmp")
port = os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT", "9000")
base_dir = os.path.dirname(os.path.abspath(__file__))
vendor_dir = os.path.join(base_dir, "vendor")
ocr_dir = "/tmp/ocr_libs"

if sys.platform.startswith("linux"):
    for p in (vendor_dir, ocr_dir):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    ocr_req = os.path.join(base_dir, "requirements-ocr.txt")
    done_flag = os.path.join(ocr_dir, ".installed")
    if os.path.isfile(ocr_req) and not os.path.isfile(done_flag):
        os.makedirs(ocr_dir, exist_ok=True)
        subprocess.Popen([
            sys.executable, "-m", "pip", "install",
            "--no-cache-dir", "--target", ocr_dir, "-r", ocr_req,
        ])

env = os.environ.copy()
if sys.platform.startswith("linux"):
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{vendor_dir}:{ocr_dir}:{base_dir}:{existing_pp}".rstrip(":")

cmd = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    os.path.join(base_dir, "app.py"),
    f"--server.port={port}",
    "--server.address=0.0.0.0",
    "--server.headless=true",
    "--server.enableCORS=false",
    "--server.enableXsrfProtection=false",
]

sys.exit(subprocess.call(cmd, env=env))