from pathlib import Path
import hashlib,json,os,re,tempfile
def safe(v):
 if not v or Path(v).name!=v or not re.fullmatch(r'[A-Za-z0-9._-]+',v):raise ValueError('invalid name')
 return v
def under(root,name):
 base=Path(root).resolve();p=(base/safe(name)).resolve()
 if base not in p.parents:raise ValueError('path traversal rejected')
 return p
def allocate(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 for i in range(1,1000000):
  p=root/f'run_{i:04d}'
  try:p.mkdir();return p.name,p
  except FileExistsError:pass
 raise RuntimeError('run namespace exhausted')
def atomic(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=path.parent)
 with os.fdopen(fd,'w') as f:json.dump(obj,f,indent=2,default=str)
 os.replace(tmp,path)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
