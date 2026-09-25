from pathlib import Path
from collections.abc import Iterator
import cv2
from howvision.types import Frame
from howvision.core.fs import IMAGES,VIDEOS
def frames(path:Path)->Iterator[Frame]:
    ext=path.suffix.lower()
    if ext in IMAGES:
        im=cv2.imread(str(path));
        if im is None: raise ValueError(f"cannot decode image: {path.name}")
        h,w=im.shape[:2]; yield Frame(im,path.name,0,0,0.0,{"kind":"image","value":0.0},w,h,"image"); return
    if ext not in VIDEOS: raise ValueError(f"unsupported extension: {ext}")
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise ValueError(f"cannot open video: {path.name}")
    i=0
    try:
        while True:
            ok,im=cap.read()
            if not ok: break
            native=float(cap.get(cv2.CAP_PROP_POS_MSEC)); h,w=im.shape[:2]
            yield Frame(im,path.name,i,i,native,{"kind":"opencv_pos_msec","value":native},w,h,"video"); i+=1
    finally: cap.release()
