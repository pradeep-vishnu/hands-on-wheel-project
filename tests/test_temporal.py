from howvision.inference.temporal import TemporalDecision
from howvision.inference.types import Prediction
def cfg():return {'confidence_threshold':.5,'smoothing_window':3,'on_confirmation_frames':2,'off_confirmation_frames':2,'unknown_timeout':2}
def test_confirmation_unknown_and_reset():
 t=TemporalDecision(cfg());assert t.process(Prediction('LEFT_ON',.9)).state=='UNKNOWN';assert t.process(Prediction('LEFT_ON',.9)).state=='LEFT_ON';assert t.process(Prediction('UNKNOWN',.1)).state=='LEFT_ON';assert t.process(Prediction('UNKNOWN',.1)).state=='UNKNOWN';t.reset();assert t.stable=='UNKNOWN'
def test_off_confirmation():
 t=TemporalDecision(cfg());t.process(Prediction('NONE_ON',.9));assert t.process(Prediction('NONE_ON',.9)).state=='NONE_ON'
