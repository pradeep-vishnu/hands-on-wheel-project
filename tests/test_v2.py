import socket
from fastapi.testclient import TestClient
from howvision.api.app import create_app
from howvision.launcher import free_port

def test_random_ports_are_available():
 p1=free_port();p2=free_port();assert p1>0 and p2>0

def test_full_ui_contract():
 c=TestClient(create_app());h=c.get('/').text
 for word in ['Inference','Review Studio','Fine-tune','Models']:assert word in h
 assert c.get('/api/training/cues').status_code==200
 assert c.get('/api/models').status_code==200

def test_training_guard():
 c=TestClient(create_app());r=c.post('/api/training',json={'selected_frames':[],'device':'cpu','hyperparameters':{}});assert r.status_code==400
