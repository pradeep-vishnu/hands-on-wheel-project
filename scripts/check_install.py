from pathlib import Path
import importlib,subprocess,sys,urllib.request,os,tempfile
REQ={'fastapi':'fastapi>=0.110,<1','uvicorn':'uvicorn[standard]>=0.27,<1','pydantic':'pydantic>=2.6,<3','cv2':'opencv-python>=4.8,<5','numpy':'numpy>=1.26,<3','yaml':'PyYAML>=6,<7','multipart':'python-multipart>=0.0.9,<1','mediapipe':'mediapipe>=0.10.14,<0.11','torch':'torch>=2.2,<3'}
def main():
 missing=[]
 for m,p in REQ.items():
  try:importlib.import_module(m);print('[OK]',m)
  except Exception:missing.append(p)
 if missing:subprocess.check_call([sys.executable,'-m','pip','install',*missing])
 dest=Path('models/hand_landmarker.task');dest.parent.mkdir(exist_ok=True)
 if not dest.exists():
  url='https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';fd,t=tempfile.mkstemp(dir=dest.parent);os.close(fd);urllib.request.urlretrieve(url,t);os.replace(t,dest);print('[OK] downloaded hand model')
 return 0
if __name__=='__main__':raise SystemExit(main())
