from abc import ABC,abstractmethod
import cv2, numpy as np
from howvision.types import WheelDetection
class WheelDetector(ABC):
    @abstractmethod
    def predict(self,frame): ...
class GeometricWheelDetector(WheelDetector):
    def __init__(self,cfg): self.c=cfg
    def predict(self,frame):
        gray=cv2.cvtColor(frame.image,cv2.COLOR_BGR2GRAY); edge=cv2.Canny(gray,self.c.get("canny_low",60),self.c.get("canny_high",160)); contours,_=cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE); h,w=gray.shape; best=None
        for contour in contours:
            if len(contour)<20: continue
            center,(a,b),angle=cv2.fitEllipse(contour); ratio=min(a,b)/max(a,b); area=np.pi*a*b/4/(w*h)
            if ratio<self.c.get("min_axis_ratio",.42) or not self.c.get("min_area_ratio",.04)<=area<=self.c.get("max_area_ratio",.78): continue
            score=min(1.0,ratio*(1-abs(area-.28)))
            if best is None or score>best[0]: best=(score,center,a,b,angle,contour)
        if not best: return WheelDetection(False,0.0)
        score,(cx,cy),a,b,angle,contour=best
        return WheelDetection(score>=self.c.get("min_confidence",.25),float(score),(cx/w,cy/h),(a/w,b/h),float(angle),[(float(x/w),float(y/h)) for [[x,y]] in contour])
