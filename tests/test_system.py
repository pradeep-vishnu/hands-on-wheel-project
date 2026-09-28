import json
from pathlib import Path
from fastapi.testclient import TestClient
from howvision.api.app import create_app
from howvision.core.device import resolve
from howvision.core.fs import allocate,safe
from howvision.launcher import free_port

def test_security_and_allocation(tmp_path):
 try:safe('../x');assert False
 except ValueError:pass
 assert allocate(tmp_path)[0]=='run_0001'
 assert allocate(tmp_path)[0]=='run_0002'

def test_devices_and_fallback():
 c=TestClient(create_app());d=c.get('/api/devices').json()['devices']
 assert any(x['id']=='cpu' and x['available'] for x in d)
 assert resolve('invalid').actual=='cpu'

def test_ui_and_endpoints():
 c=TestClient(create_app());h=c.get('/').text
 for x in ['Inference','Review Studio','Fine-tune','Models','Reviewed ▶','Predicted masks']:assert x in h
 for p in ['/api/system','/api/devices','/api/training/cues','/api/models']:assert c.get(p).status_code==200

def test_annotation_schema():
 c=TestClient(create_app());p={'run_id':'run_0001','frame_id':0,'label':'LEFT_ON','original_label':'UNKNOWN','reward':1,'hand_strokes':[{'tool':'hand','points':[[.1,.2]]}],'wheel_strokes':[],'include_training':True}
 assert c.post('/api/review/run_0001/annotations',json=p).status_code==200

def test_training_guard():
 c=TestClient(create_app());assert c.post('/api/training',json={'selected_frames':[],'hyperparameters':{}}).status_code==400

def test_random_port():
 assert free_port()>0
