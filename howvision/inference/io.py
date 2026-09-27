from pathlib import Path
import cv2
from howvision.core.fs import IMAGES,VIDEOS
from .types import FrameIdentity
def frames(path):
    path=Path(path);ext=path.suffix.lower()
    if ext in IMAGES:
        im=cv2.imread(str(path));
        if im is None:raise ValueError('image decode failed')
        h,w=im.shape[:2];yield im,FrameIdentity(path.name,0,0,0.,{'kind':'image','value':0},w,h);return
    cap=cv2.VideoCapture(str(path));i=0
    try:
        while True:
            ok,im=cap.read()
            if not ok:break
            h,w=im.shape[:2];ts=float(cap.get(cv2.CAP_PROP_POS_MSEC));yield im,FrameIdentity(path.name,i,i,ts,{'kind':'opencv_pos_msec','value':ts},w,h);i+=1
    finally:cap.release()
