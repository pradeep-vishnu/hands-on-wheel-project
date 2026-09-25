from pathlib import Path
import importlib,subprocess,sys,urllib.request,os,tempfile
req={'fastapi':'fastapi','uvicorn':'uvicorn[standard]','cv2':'opencv-python','numpy':'numpy','yaml':'PyYAML','multipart':'python-multipart','mediapipe':'mediapipe','torch':'torch','psutil':'psutil'};missing=[]
for module,dist in req.items():
 try:importlib.import_module(module);print('[OK]',module)
 except Exception:missing.append(dist)
if missing:subprocess.check_call([sys.executable,'-m','pip','install',*missing])
p=Path('models/hand_landmarker.task');p.parent.mkdir(exist_ok=True)
if not p.exists():
 fd,tmp=tempfile.mkstemp(dir=p.parent);os.close(fd);urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',tmp);os.replace(tmp,p)
