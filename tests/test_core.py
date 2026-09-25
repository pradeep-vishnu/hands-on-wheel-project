import csv,json,cv2,numpy as np,yaml
from pathlib import Path
from howvision.types import *
from howvision.core.fs import allocate_run,safe_name
from howvision.pipeline.source import frames
from howvision.pipeline.run import execute,CSV_COLUMNS
from howvision.temporal.engine import TemporalDecisionEngine
class Hands:
    def predict(self,f): return []
class Wheel:
    def predict(self,f): return WheelDetection(False,0)
def config(tmp):
    c=yaml.safe_load(Path('configs/default.yaml').read_text()); c['paths']={'data':str(tmp/'data'),'output':str(tmp/'out'),'hand_model':str(tmp/'missing.task')}; return c
def test_mapping_and_security():
    assert to_status(HOWState.UNKNOWN)==HOWStatus.UNKNOWN
    try: safe_name('../x.png'); assert False
    except ValueError: pass
def test_allocation(tmp_path):
    assert allocate_run(tmp_path)[0]=='run_0001'; assert allocate_run(tmp_path)[0]=='run_0002'
def test_image_outputs(tmp_path):
    p=tmp_path/'x.png'; cv2.imwrite(str(p),np.zeros((60,80,3),np.uint8)); rid,rd,_=execute(config(tmp_path),p,hand_detector=Hands(),wheel_detector=Wheel()); assert (rd/'output.png').exists(); assert json.loads((rd/'manifest.json').read_text())['status']=='COMPLETED'; rows=list(csv.DictReader((rd/'predictions.csv').open())); assert list(rows[0])==CSV_COLUMNS
def test_video_order_timestamp(tmp_path):
    p=tmp_path/'x.avi'; w=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48)); [w.write(np.full((48,64,3),i,np.uint8)) for i in range(4)]; w.release(); fs=list(frames(p)); assert [f.frame_id for f in fs]==[0,1,2,3]; assert all('value' in f.timestamp_native for f in fs)
def test_temporal():
    e=TemporalDecisionEngine({'smoothing_window':1,'confidence_threshold':.4,'on_confirmation_frames':2,'off_confirmation_frames':2,'unknown_timeout':2}); p=Prediction(HOWState.LEFT_ON,.9); assert e.process(p).state==HOWState.UNKNOWN; assert e.process(p).state==HOWState.LEFT_ON; e.reset(); assert e.stable==HOWState.UNKNOWN
