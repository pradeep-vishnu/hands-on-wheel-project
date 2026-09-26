from fastapi.testclient import TestClient
from howvision.api.app import create_app
def test_contract():
 c=TestClient(create_app());h=c.get('/').text;j=c.get('/api/health').json()
 assert j['version']=='0.9.0' and 'cpu' in j and 'gpu' in j and 'time' in j
 for x in ['Undo last','Reviewed ▶','◀ Reviewed','Move']:assert x in h
def test_training_guard():
 c=TestClient(create_app());assert c.post('/api/training',json={'selected_frames':[]}).status_code==422
