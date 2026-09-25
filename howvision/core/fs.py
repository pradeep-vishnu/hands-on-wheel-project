from pathlib import Path
import hashlib, json, os, re, tempfile
IMAGES={".jpg",".jpeg",".png",".bmp",".webp"}; VIDEOS={".mp4",".avi",".mov",".mkv"}
def safe_name(name:str)->str:
    raw=Path(name).name
    if raw != name or raw in {"",".",".."}: raise ValueError("invalid filename")
    clean=re.sub(r"[^A-Za-z0-9._-]+","_",raw)
    if not clean: raise ValueError("invalid filename")
    return clean
def resolve_under(root:Path,name:str)->Path:
    root=root.resolve(); p=(root/safe_name(name)).resolve()
    if root not in p.parents: raise ValueError("path traversal rejected")
    return p
def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def atomic_json(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=".tmp-")
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(obj,f,indent=2,default=str)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
def allocate_run(root:Path):
    root.mkdir(parents=True,exist_ok=True)
    for i in range(1,1_000_000):
        rid=f"run_{i:04d}"; p=root/rid
        try: p.mkdir(); return rid,p
        except FileExistsError: pass
    raise RuntimeError("run namespace exhausted")
