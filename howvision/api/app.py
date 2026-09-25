from pathlib import Path
from threading import Thread,Lock
from uuid import uuid4
import json
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from howvision.config import load
from howvision.core.fs import under,safe_name,IMAGES,VIDEOS
from howvision.core.db import Store
from howvision.pipeline.inference import Engine
from howvision.learning.trainer import train
jobs={};lock=Lock()
class Annotation(BaseModel):
 run_id:str; frame_id:int; label:str; original_label:str; reward:float=1.0; hand_polygons:list=[]; wheel_polygon:list=[]; flags:list=[]; annotator:str='local-user'
def create_app():
 cfg=load(); data=Path(cfg['paths']['data']);out=Path(cfg['paths']['output']);data.mkdir(exist_ok=True);out.mkdir(exist_ok=True);store=Store(cfg['paths']['database']);app=FastAPI(title='Hands On Wheel Project',version='0.3.0')
 @app.get('/api/health')
 def health(): return {'status':'READY' if Path(cfg['paths']['hand_model']).exists() else 'SETUP_REQUIRED','review_enabled':True,'learning':'reward-weighted policy optimization'}
 @app.post('/api/input/upload')
 async def upload(file:UploadFile=File(...)):
  p=under(data,safe_name(file.filename or ''))
  if p.suffix.lower() not in IMAGES|VIDEOS:raise HTTPException(400,'unsupported file')
  with p.open('wb') as f:
   while b:=await file.read(1048576):f.write(b)
  return {'name':p.name}
 @app.get('/api/input')
 def inputs():return [p.name for p in data.iterdir() if p.is_file()]
 @app.post('/api/runs')
 def run(body:dict):
  p=under(data,body.get('input',''))
  if not p.exists():raise HTTPException(404,'input not found')
  jid='job_'+uuid4().hex[:10];jobs[jid]={'job_id':jid,'kind':'inference','status':'QUEUED','progress':0}
  def update(x):
   with lock:jobs[jid].update(x)
  def work():
   try:update({'status':'RUNNING'});rid=Engine(cfg).run(p,update);update({'status':'COMPLETED','run_id':rid,'progress':1})
   except Exception as e:update({'status':'FAILED','error':str(e)})
  Thread(target=work,daemon=True).start();return jobs[jid]
 @app.get('/api/jobs/{jid}')
 def job(jid:str):
  if jid not in jobs:raise HTTPException(404,'job not found')
  return jobs[jid]
 @app.get('/api/runs')
 def runs():
  z=[]
  for p in sorted(out.glob('run_*'),reverse=True):
   if (p/'manifest.json').exists():z.append(json.loads((p/'manifest.json').read_text()))
  return z
 @app.get('/api/review/{rid}')
 def review(rid:str):
  p=out/rid/'predictions.jsonl'
  if not p.exists():raise HTTPException(404,'run not found')
  anns={x['frame_id']:x for x in store.annotations(rid)};recs=[]
  for line in p.open():
   r=json.loads(line);r['overlay']=f'/api/runs/{rid}/frames/{r["frame_id"]}';r['annotation']=anns.get(r['frame_id']);recs.append(r)
  return recs
 @app.post('/api/review/{rid}/annotations')
 def annotate(rid:str,a:Annotation):
  if rid!=a.run_id:raise HTTPException(400,'run mismatch')
  if a.label not in ['LEFT_ON','RIGHT_ON','BOTH_ON','NONE_ON','UNKNOWN']:raise HTTPException(400,'invalid label')
  store.annotate(a.model_dump());return {'saved':True}
 @app.post('/api/training')
 def training():
  jid='job_'+uuid4().hex[:10];jobs[jid]={'job_id':jid,'kind':'training','status':'QUEUED','progress':0}
  def update(x):jobs[jid].update(x)
  def work():
   try:update({'status':'RUNNING'});mid,m=train(cfg,store,update);update({'status':'COMPLETED','model_id':mid,'metrics':m,'progress':1})
   except Exception as e:update({'status':'FAILED','error':str(e)})
  Thread(target=work,daemon=True).start();return jobs[jid]
 @app.get('/api/runs/{rid}/frames/{fid}')
 def frame(rid:str,fid:int):
  p=out/rid/'overlays'/f'frame_{fid:06d}.jpg'
  if not p.exists():raise HTTPException(404,'frame not found')
  return FileResponse(p)
 app.mount('/',StaticFiles(directory='frontend',html=True),name='web');return app
