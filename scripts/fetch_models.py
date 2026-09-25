from pathlib import Path
import hashlib,json,os,tempfile,time,urllib.request
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEST=Path("models/hand_landmarker/hand_landmarker.task")
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def main():
    DEST.parent.mkdir(parents=True,exist_ok=True)
    if DEST.exists() and DEST.stat().st_size>1_000_000: print(f"[OK] cached model {DEST} sha256={digest(DEST)}"); return 0
    fd,tmp=tempfile.mkstemp(dir=DEST.parent,prefix=".download-"); os.close(fd); tmp=Path(tmp)
    try:
        print(f"Downloading official MediaPipe Hand Landmarker model\n{URL}")
        urllib.request.urlretrieve(URL,tmp)
        if tmp.stat().st_size<1_000_000: raise RuntimeError("downloaded model is unexpectedly small")
        os.replace(tmp,DEST); meta={"name":"MediaPipe Hand Landmarker","variant":"float16/1","source":URL,"license":"Apache-2.0 model/code ecosystem; verify deployment obligations","sha256":digest(DEST),"downloaded_at":time.time(),"size":DEST.stat().st_size}; Path("models/hand_landmarker/model.json").write_text(json.dumps(meta,indent=2)); print(f"[OK] {DEST} ({DEST.stat().st_size/1e6:.1f} MB) sha256={meta['sha256']}"); return 0
    finally:
        if tmp.exists(): tmp.unlink()
if __name__=="__main__": raise SystemExit(main())
