from dataclasses import dataclass,asdict
from typing import Any
@dataclass(frozen=True)
class FrameIdentity:
    source_file:str; frame_id:int; sequence_index:int; timestamp_ms:float; native_timestamp:dict[str,Any]; width:int; height:int
@dataclass(frozen=True)
class HandDetection:
    side:str; confidence:float; landmarks:list[list[float]]; polygon:list[list[float]]
@dataclass(frozen=True)
class WheelDetection:
    visible:bool; confidence:float; polygon:list[list[float]]; center:list[float]|None=None; axes:list[float]|None=None
@dataclass(frozen=True)
class ContactFeatures:
    left_distance:float; right_distance:float; wheel_confidence:float; left_confidence:float; right_confidence:float
@dataclass(frozen=True)
class Prediction:
    state:str; confidence:float
@dataclass(frozen=True)
class FrameResult:
    identity:FrameIdentity; hands:list[HandDetection]; wheel:WheelDetection; features:ContactFeatures; raw:Prediction; temporal:Prediction; how_status:str
    def json(self):
        d=asdict(self);d['source_file']=d['identity']['source_file'];d['frame_id']=d['identity']['frame_id'];d['sequence_index']=d['identity']['sequence_index'];d['timestamp_ms']=d['identity']['timestamp_ms'];return d
