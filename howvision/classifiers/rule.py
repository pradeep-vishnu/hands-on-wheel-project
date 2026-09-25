from abc import ABC,abstractmethod
import math
from howvision.types import *
class ContactFeatureExtractor:
    def extract(self,hands,wheel):
        values={"wheel_confidence":wheel.confidence}; available={HandSide.LEFT:False,HandSide.RIGHT:False}
        if wheel.center and wheel.axes:
            cx,cy=wheel.center; rx,ry=max(wheel.axes[0]/2,1e-6),max(wheel.axes[1]/2,1e-6)
            for hand in hands:
                if hand.side not in available: continue
                available[hand.side]=True; key=[hand.landmarks[i] for i in (0,4,5,8,9,12,13,16,17,20)]; distances=[abs(math.sqrt(((x-cx)/rx)**2+((y-cy)/ry)**2)-1) for x,y,_ in key]; p=hand.side.value.lower(); values[p+"_wheel_distance"]=min(distances); values[p+"_hand_confidence"]=hand.confidence
        return ContactFeatures(values,available[HandSide.LEFT],available[HandSide.RIGHT])
class HOWClassifier(ABC):
    @abstractmethod
    def predict(self,features): ...
class RuleBasedHOWClassifier(HOWClassifier):
    def __init__(self,cfg): self.c=cfg
    def predict(self,f):
        if f.values.get("wheel_confidence",0)<self.c.get("wheel_confidence",.25): return Prediction(HOWState.UNKNOWN,0.0,["wheel unavailable"])
        seen=[]; on={}
        for side in ("left","right"):
            d=f.values.get(side+"_wheel_distance"); hc=f.values.get(side+"_hand_confidence",0)
            if d is not None and hc>=self.c.get("hand_confidence",.45): seen.append(side); on[side]=d<=self.c.get("contact_distance",.18)
        if not seen: return Prediction(HOWState.UNKNOWN,.2,["no confident hand evidence"])
        state=HOWState.BOTH_ON if on.get("left") and on.get("right") else HOWState.LEFT_ON if on.get("left") else HOWState.RIGHT_ON if on.get("right") else HOWState.NONE_ON if all(f.values[s+"_wheel_distance"]>=self.c.get("no_contact_distance",.30) for s in seen) else HOWState.UNKNOWN
        confidence=min(1.0,sum(f.values[s+"_hand_confidence"] for s in seen)/len(seen)*f.values["wheel_confidence"]); return Prediction(state,confidence,[])
