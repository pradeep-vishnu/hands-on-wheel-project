from pathlib import Path
import os,tempfile,urllib.request
p=Path('models/hand_landmarker.task');p.parent.mkdir(exist_ok=True)
if not p.exists():
 fd,t=tempfile.mkstemp(dir=p.parent);os.close(fd);urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',t)
 if Path(t).stat().st_size<1000000:raise RuntimeError('invalid model download')
 os.replace(t,p)
