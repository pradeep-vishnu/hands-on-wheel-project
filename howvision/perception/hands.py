from abc import ABC,abstractmethod
from pathlib import Path
import cv2, numpy as np
from howvision.types import HandDetection,HandSide
class HandDetector(ABC):
    @abstractmethod
    def predict(self,frame): ...
class MediaPipeTaskHandDetector(HandDetector):
    def __init__(self,cfg,model_path):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as e: raise RuntimeError("MediaPipe missing. Run: python scripts/check_install.py --install") from e
        path=Path(model_path)
        if not path.is_file(): raise RuntimeError("Hand model missing. Run: python scripts/fetch_models.py")
        self.mp=mp; self.vision=vision
        options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(path)),running_mode=vision.RunningMode.IMAGE,num_hands=cfg.get("max_hands",2),min_hand_detection_confidence=cfg.get("min_detection_confidence",.5),min_hand_presence_confidence=cfg.get("min_presence_confidence",.5),min_tracking_confidence=cfg.get("min_tracking_confidence",.5))
        self.detector=vision.HandLandmarker.create_from_options(options)
    def predict(self,frame):
        rgb=cv2.cvtColor(frame.image,cv2.COLOR_BGR2RGB); image=self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=rgb); result=self.detector.detect(image); out=[]
        for i,lms in enumerate(result.hand_landmarks):
            cat=result.handedness[i][0] if i<len(result.handedness) and result.handedness[i] else None
            label=(cat.category_name or "UNKNOWN").upper() if cat else "UNKNOWN"; side=HandSide(label) if label in {"LEFT","RIGHT"} else HandSide.UNKNOWN; conf=float(cat.score) if cat else 0.0
            pts=[(float(p.x),float(p.y),float(p.z)) for p in lms]; xy=np.array([(x,y) for x,y,_ in pts],np.float32); hull=cv2.convexHull(xy).reshape(-1,2)
            x0,y0=xy.min(0); x1,y1=xy.max(0); out.append(HandDetection(side,conf,(float(x0),float(y0),float(x1),float(y1)),pts,[(float(x),float(y)) for x,y in hull]))
        return out
