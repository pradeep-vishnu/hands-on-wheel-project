from pathlib import Path
from fastapi.testclient import TestClient
from howvision.api.app import create_app
def test_ui_and_health():
 c=TestClient(create_app());html=c.get('/').text
 assert c.get('/api/health').json()['version']=='0.7.0'
 assert 'Waiting for inference' not in html
 assert 'Open queue' not in html and 'Magic Select' not in html
 assert 'Training influence' in html and 'Input cues' in html
def test_training_requires_selection():
 c=TestClient(create_app());r=c.post('/api/training',json={'selected_frames':[]});assert r.status_code==422
