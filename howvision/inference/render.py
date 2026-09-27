import cv2,numpy as np
class OverlayRenderer:
    def render(self,frame,result):
        out=frame.copy();h,w=out.shape[:2]
        for x in result.hands:cv2.polylines(out,[np.array([[int(a*w),int(b*h)] for a,b in x.polygon])],True,(255,80,205),2)
        p=np.array([[int(a*w),int(b*h)] for a,b in result.wheel.polygon]);
        if len(p):cv2.polylines(out,[p],True,(50,210,255),2)
        return out
