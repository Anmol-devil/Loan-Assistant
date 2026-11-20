import os
import subprocess
import sys

os.environ.setdefault("HOME", "/tmp")
port = os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT", "9000")
base_dir = os.path.dirname(os.path.abspath(__file__))
vendor_dir = os.path.join(base_dir, "vendor")
tmp_dir = "/tmp/pylibs"

if sys.platform.startswith("linux"):
    for p in (vendor_dir, tmp_dir):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

try:
    import streamlit  # noqa: F401
except ImportError:
    os.makedirs(tmp_dir, exist_ok=True)
    req_file = os.path.join(base_dir, "requirements.txt")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir", "--target", tmp_dir, "-r", req_file,
    ])
    sys.path.insert(0, tmp_dir)

env = os.environ.copy()
if sys.platform.startswith("linux"):
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{vendor_dir}:{tmp_dir}:{base_dir}:{existing_pp}".rstrip(":")

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