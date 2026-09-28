import json
from pathlib import Path
from fastapi.testclient import TestClient
from howvision.api.app import create_app
from howvision.core.device import resolve

def test_devices_and_fallback():
 c=TestClient(create_app());d=c.get('/api/devices').json()['devices'];assert any(x['id']=='cpu' and x['available'] for x in d);assert resolve('definitely-bad').actual=='cpu'
def test_annotation_contract_prevents_422(tmp_path,monkeypatch):
 c=TestClient(create_app());payload={'run_id':'run_0001','frame_id':0,'label':'LEFT_ON','original_label':'UNKNOWN','hand_strokes':[{'tool':'hand','points':[[.1,.2],[.2,.3]]}],'wheel_strokes':[],'include_training':True};r=c.post('/api/review/run_0001/annotations',json=payload);assert r.status_code==200,r.text
def test_annotation_rejects_mismatch():
 c=TestClient(create_app());p={'run_id':'run_2','frame_id':0,'label':'LEFT_ON','original_label':'UNKNOWN','hand_strokes':[],'wheel_strokes':[],'include_training':True};assert c.post('/api/review/run_1/annotations',json=p).status_code==400
def test_ui_has_device_and_guard():
 c=TestClient(create_app());h=c.get('/').text;assert 'Compute device' in h and 'Save & continue' in h
