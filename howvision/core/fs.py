from pathlib import Path
import hashlib,json,os,re,tempfile
IMAGES={'.jpg','.jpeg','.png','.bmp','.webp'}; VIDEOS={'.mp4','.avi','.mov','.mkv'}
def safe_name(v:str)->str:
    if Path(v).name!=v or not re.fullmatch(r'[A-Za-z0-9._-]+',v): raise ValueError('invalid filename')
    return v
def allocate(root,prefix):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    for i in range(1,100000):
        p=root/f'{prefix}_{i:04d}'
        try:p.mkdir();return p.name,p
        except FileExistsError:pass
    raise RuntimeError('namespace exhausted')
def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(obj,f,indent=2,default=str)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
