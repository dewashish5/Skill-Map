"""One command start: python run_all.py  (MySQL must be running; loads the bundle on first run)."""
import subprocess, sys, time, os
from pathlib import Path
H = Path(__file__).resolve().parent
os.chdir(H)
py = sys.executable
if not (H / "lmis_bundle.pkl").exists():
    sys.exit("lmis_bundle.pkl missing: run notebook/LMIS_Part1_Part2.ipynb first and copy lmis_bundle.pkl here.")
if "--reload-db" in sys.argv or "--init" in sys.argv:
    subprocess.check_call([py, "load_db.py"])
api = subprocess.Popen([py, "-m", "uvicorn", "api:app", "--port", "8000"])
time.sleep(3)
try:
    subprocess.call([py, "-m", "streamlit", "run", "app.py"])
finally:
    api.terminate()
