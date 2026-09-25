from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import StrEnum
from typing import Any
import numpy as np
class HandSide(StrEnum): LEFT="LEFT"; RIGHT="RIGHT"; UNKNOWN="UNKNOWN"
class HOWState(StrEnum): LEFT_ON="LEFT_ON"; RIGHT_ON="RIGHT_ON"; BOTH_ON="BOTH_ON"; NONE_ON="NONE_ON"; UNKNOWN="UNKNOWN"
class HOWStatus(StrEnum): HOW_ON="HOW_ON"; HOW_OFF="HOW_OFF"; UNKNOWN="UNKNOWN"
def to_status(s: HOWState)->HOWStatus:
    if s in {HOWState.LEFT_ON,HOWState.RIGHT_ON,HOWState.BOTH_ON}: return HOWStatus.HOW_ON
    if s == HOWState.NONE_ON: return HOWStatus.HOW_OFF
    return HOWStatus.UNKNOWN
@dataclass(slots=True)
class Frame:
    image:np.ndarray; source_file:str; frame_id:int; sequence_index:int; timestamp_ms:float; timestamp_native:dict[str,Any]; width:int; height:int; media_type:str
@dataclass(slots=True)
class HandDetection:
    side:HandSide; confidence:float; bbox:tuple[float,float,float,float]; landmarks:list[tuple[float,float,float]]; approximate_mask:list[tuple[float,float]]=field(default_factory=list)
@dataclass(slots=True)
class WheelDetection:
    visible:bool; confidence:float; center:tuple[float,float]|None=None; axes:tuple[float,float]|None=None; angle:float=0.0; contour:list[tuple[float,float]]=field(default_factory=list)
@dataclass(slots=True)
class ContactFeatures: values:dict[str,float]; left_available:bool; right_available:bool
@dataclass(slots=True)
class Prediction: state:HOWState; confidence:float; reasons:list[str]=field(default_factory=list)
@dataclass(slots=True)
class FrameResult:
    run_id:str; source_file:str; frame_id:int; sequence_index:int; timestamp_ms:float; timestamp_native:dict[str,Any]; left_hand:HandDetection|None; right_hand:HandDetection|None; wheel:WheelDetection; contact_features:ContactFeatures; raw_prediction:Prediction; temporal_prediction:Prediction; final_status:HOWStatus; confidence:float; model_version:str="mediapipe-hand-landmarker-f16-v1+rule-v1"; decision_version:str="temporal-v1"
    def dict(self): return asdict(self)
