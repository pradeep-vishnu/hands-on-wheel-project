from fastapi.testclient import TestClient
from howvision.api.app import create_app
def test_contract():
 c=TestClient(create_app());assert c.get('/').status_code==200;h=c.get('/api/health').json();assert h['version']=='1.1.0' and h['gpu']['usage'] is None;assert c.get('/api/models').json()['baseline']['parameter_count']==269
def test_errors_and_training_guard():
 c=TestClient(create_app());assert c.get('/api/jobs/nope').status_code==404;assert c.post('/api/runs',json={'input':'missing.png','checkpoint':'baseline'}).status_code==404;assert c.post('/api/training',json={'selected_frames':[],'hyperparameters':{}}).status_code==422
