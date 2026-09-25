from pathlib import Path
import re,json,os,tempfile,hashlib
IMAGES={'.jpg','.jpeg','.png','.bmp','.webp'};VIDEOS={'.mp4','.avi','.mov','.mkv'}
def safe(v):
 if Path(v).name!=v or not re.fullmatch(r'[A-Za-z0-9._-]+',v):raise ValueError('invalid name')
 return v
def allocate(root,prefix):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 for i in range(1,100000):
  p=root/f'{prefix}_{i:04d}'
  try:p.mkdir();return p.name,p
  except FileExistsError:pass
 raise RuntimeError('namespace exhausted')
def atomic(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent)
 with os.fdopen(fd,'w') as f:json.dump(o,f,indent=2,default=str)
 os.replace(t,p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
