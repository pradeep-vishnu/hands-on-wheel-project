from collections import deque,Counter
from .types import Prediction
class TemporalDecision:
    def __init__(self,cfg):self.cfg=cfg;self.window=deque(maxlen=cfg['smoothing_window']);self.stable='UNKNOWN';self.candidate=None;self.count=0;self.unknowns=0
    def reset(self):self.window.clear();self.stable='UNKNOWN';self.candidate=None;self.count=0;self.unknowns=0
    def process(self,p):
        state=p.state if p.confidence>=self.cfg['confidence_threshold'] else 'UNKNOWN';self.window.append(state)
        if state=='UNKNOWN':
            self.unknowns+=1
            if self.unknowns>=self.cfg['unknown_timeout']:self.stable='UNKNOWN'
            return Prediction(self.stable,p.confidence)
        self.unknowns=0;major=Counter(self.window).most_common(1)[0][0]
        if major==self.stable:return Prediction(self.stable,p.confidence)
        if major!=self.candidate:self.candidate=major;self.count=1
        else:self.count+=1
        need=self.cfg['off_confirmation_frames'] if major=='NONE_ON' else self.cfg['on_confirmation_frames']
        if self.count>=need:self.stable=major;self.candidate=None;self.count=0
        return Prediction(self.stable,p.confidence)
