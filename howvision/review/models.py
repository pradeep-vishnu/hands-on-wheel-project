from pydantic import BaseModel,Field
class Stroke(BaseModel):
 tool:str
 points:list[tuple[float,float]]=Field(default_factory=list)
class AnnotationRequest(BaseModel):
 run_id:str
 frame_id:int=Field(ge=0)
 label:str
 original_label:str
 hand_strokes:list[Stroke]=Field(default_factory=list)
 wheel_strokes:list[Stroke]=Field(default_factory=list)
 include_training:bool=True
