from pathlib import Path
import argparse,importlib,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); PROJECTS={"fastapi":"fastapi>=0.110,<1","uvicorn":"uvicorn[standard]>=0.27,<1","pydantic":"pydantic>=2.6,<3","cv2":"opencv-python>=4.8,<5","numpy":"numpy>=1.26,<3","yaml":"PyYAML>=6,<7","multipart":"python-multipart>=0.0.9,<1","mediapipe":"mediapipe>=0.10.14,<0.11"}
def missing():
    out=[]
    for module,dist in PROJECTS.items():
        try: importlib.import_module(module); print(f"[OK] {module}")
        except Exception as exc: print(f"[MISSING] {module}: {exc}"); out.append(dist)
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--check-only",action="store_true"); args=ap.parse_args(); print(f"Python {sys.version.split()[0]}")
    if sys.version_info<(3,11): print("Python 3.11+ required"); return 1
    absent=missing()
    if absent and not args.check_only:
        print("Installing missing dependencies..."); subprocess.check_call([sys.executable,"-m","pip","install",*absent]); importlib.invalidate_caches(); absent=missing()
    if absent: return 1
    subprocess.check_call([sys.executable,str(ROOT/"scripts/fetch_models.py")],cwd=ROOT); print("HOW Vision READY"); return 0
if __name__=="__main__": raise SystemExit(main())
