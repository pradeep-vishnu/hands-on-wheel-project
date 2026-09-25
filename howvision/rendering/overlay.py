import cv2,numpy as np
from howvision.types import HOWStatus
class OverlayRenderer:
    def render(self,image,result):
        out=image.copy(); h,w=out.shape[:2]; color={HOWStatus.HOW_ON:(139,227,66),HOWStatus.HOW_OFF:(103,92,255),HOWStatus.UNKNOWN:(75,184,229)}[result.final_status]
        if result.wheel.center and result.wheel.axes: cv2.ellipse(out,(int(result.wheel.center[0]*w),int(result.wheel.center[1]*h)),(int(result.wheel.axes[0]*w/2),int(result.wheel.axes[1]*h/2)),result.wheel.angle,0,360,(230,184,75),2)
        for hand in (result.left_hand,result.right_hand):
            if not hand: continue
            pts=np.array([(int(x*w),int(y*h)) for x,y in hand.approximate_mask],np.int32)
            if len(pts)>2:
                layer=out.copy(); cv2.fillConvexPoly(layer,pts,(255,120,220)); out=cv2.addWeighted(layer,.18,out,.82,0); cv2.polylines(out,[pts],True,(255,120,220),1)
            for x,y,_ in hand.landmarks: cv2.circle(out,(int(x*w),int(y*h)),2,(240,240,240),-1)
        cv2.rectangle(out,(8,8),(350,82),(11,13,16),-1); cv2.putText(out,f"{result.final_status.value}  {result.confidence:.0%}",(16,38),cv2.FONT_HERSHEY_SIMPLEX,.75,color,2); cv2.putText(out,f"frame {result.frame_id}   {result.timestamp_ms/1000:.3f}s",(16,67),cv2.FONT_HERSHEY_SIMPLEX,.48,(230,232,235),1); return out
