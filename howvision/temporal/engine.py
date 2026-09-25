from collections import deque,Counter
from howvision.types import *
class TemporalDecisionEngine:
    def __init__(self,cfg): self.c=cfg; self.reset()
    def reset(self): self.window=deque(maxlen=self.c.get("smoothing_window",5)); self.stable=HOWState.UNKNOWN; self.candidate=None; self.count=0; self.unknown=0
    def process(self,p):
        state=p.state if p.confidence>=self.c.get("confidence_threshold",.42) else HOWState.UNKNOWN; self.window.append(state); mode=Counter(self.window).most_common(1)[0][0]
        if mode==HOWState.UNKNOWN:
            self.unknown+=1
            if self.unknown>=self.c.get("unknown_timeout",4): self.stable=HOWState.UNKNOWN
            return Prediction(self.stable,p.confidence,["temporal uncertainty"])
        self.unknown=0
        if mode==self.candidate: self.count+=1
        else: self.candidate=mode; self.count=1
        required=self.c.get("off_confirmation_frames",3) if mode==HOWState.NONE_ON else self.c.get("on_confirmation_frames",2)
        if self.count>=required: self.stable=mode
        return Prediction(self.stable,p.confidence,["temporally smoothed"])
