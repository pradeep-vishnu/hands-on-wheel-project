from pathlib import Path
import re,hashlib,json,os,tempfile
IMAGES={'.jpg','.jpeg','.png','.bmp','.webp'}; VIDEOS={'.mp4','.avi','.mov','.mkv'}
def safe_name(v):
 if Path(v).name!=v or not re.fullmatch(r'[A-Za-z0-9._-]+',v): raise ValueError('invalid name')
 return v
def under(root,name): return Path(root)/safe_name(name)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def atomic_json(p,obj):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=p.parent)
 with os.fdopen(fd,'w') as f: json.dump(obj,f,indent=2,default=str)
 os.replace(t,p)
def allocate(root,prefix):
 root=Path(root); root.mkdir(parents=True,exist_ok=True)
 for i in range(1,100000):
  p=root/f'{prefix}_{i:04d}'
  try: p.mkdir(); return p.name,p
  except FileExistsError: pass
 raise RuntimeError('namespace exhausted')
