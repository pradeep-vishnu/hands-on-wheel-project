import cv2,math,numpy as np
from .types import HandDetection,WheelDetection,ContactFeatures,Prediction
class MediaPipeHandDetector:
    def __init__(self,model_path,confidence=.5):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        opts=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model_path),running_mode=vision.RunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=confidence);self.mp=mp;self.detector=vision.HandLandmarker.create_from_options(opts)
    def predict(self,frame):
        res=self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)));out=[]
        for i,lms in enumerate(res.hand_landmarks):
            cat=res.handedness[i][0];pts=[[float(p.x),float(p.y)] for p in lms];poly=cv2.convexHull(np.array(pts,np.float32)).reshape(-1,2).tolist();out.append(HandDetection((cat.category_name or 'UNKNOWN').upper(),float(cat.score),pts,poly))
        return out
class EllipseWheelDetector:
    def predict(self,frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);cs,_=cv2.findContours(cv2.Canny(gray,60,160),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);h,w=gray.shape;best=None
        for c in cs:
            if len(c)<20:continue
            (x,y),(a,b),ang=cv2.fitEllipse(c);ratio=min(a,b)/max(a,b);area=np.pi*a*b/4/(w*h)
            if .04<area<.78 and ratio>.42 and (best is None or ratio>best[0]):best=(ratio,x/w,y/h,a/w,b/h,ang)
        if not best:return WheelDetection(False,0,[])
        s,x,y,a,b,ang=best;poly=cv2.ellipse2Poly((int(x*w),int(y*h)),(int(a*w/2),int(b*h/2)),int(ang),0,360,10);return WheelDetection(True,float(s),[[u/w,v/h] for u,v in poly],[x,y],[a,b])
def features(hands,wheel):
    f={'left_distance':2.,'right_distance':2.,'wheel_confidence':wheel.confidence,'left_confidence':0.,'right_confidence':0.}
    if wheel.center:
        cx,cy=wheel.center;rx,ry=wheel.axes[0]/2,wheel.axes[1]/2
        for h in hands:s=h.side.lower();f[s+'_distance']=min(abs(math.hypot((x-cx)/rx,(y-cy)/ry)-1) for x,y in h.landmarks);f[s+'_confidence']=h.confidence
    return ContactFeatures(**f)
class RuleClassifier:
    def __init__(self,contact=.18):self.contact=contact
    def predict(self,f,hands,wheel):
        if not wheel.visible:return Prediction('UNKNOWN',0)
        l=f.left_distance<self.contact;r=f.right_distance<self.contact;state='BOTH_ON' if l and r else 'LEFT_ON' if l else 'RIGHT_ON' if r else 'NONE_ON' if hands else 'UNKNOWN';return Prediction(state,min(1.,wheel.confidence*max(f.left_confidence,f.right_confidence,.2)))
class TorchClassifier:
    def __init__(self,path):
        import torch
        from torch import nn
        b=torch.load(path,map_location='cpu');w=b['hidden_width'];self.labels=b['labels'];self.model=nn.Sequential(nn.Linear(5,w),nn.ReLU(),nn.Linear(w,5));self.model.load_state_dict(b['state']);self.model.eval()
    def predict(self,f,hands,wheel):
        import torch
        x=torch.tensor([[f.left_distance,f.right_distance,f.wheel_confidence,f.left_confidence,f.right_confidence]],dtype=torch.float32);p=torch.softmax(self.model(x),1)[0];return Prediction(self.labels[int(p.argmax())],float(p.max()))
