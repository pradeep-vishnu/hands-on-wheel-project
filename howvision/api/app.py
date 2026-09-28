from pathlib import Path
from threading import Thread
from uuid import uuid4
import json
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from howvision.config import load
from howvision.core.fs import safe,under
from howvision.core.device import discover,resolve
from howvision.core.db import Store
from howvision.review.models import AnnotationRequest
from howvision.pipeline.run import run
from howvision.training.service import reviewed,suggest,train,model_info
jobs={}
def create_app():
 c=load();data=Path(c['paths']['data']);out=Path(c['paths']['output']);data.mkdir(exist_ok=True);out.mkdir(exist_ok=True);store=Store(c['paths']['database']);app=FastAPI(title='HOW Vision',version='2.0.0')
 @app.get('/api/devices')
 def devices():return {'devices':discover()}
 @app.post('/api/input/upload')
 async def upload(file:UploadFile=File(...)):
  p=under(data,safe(file.filename or ''));allowed=set(c['input']['images']+c['input']['videos'])
  if p.suffix.lower() not in allowed:raise HTTPException(400,'unsupported extension')
  with p.open('wb') as f:
   while b:=await file.read(1048576):f.write(b)
  return {'name':p.name}
 @app.get('/api/input')
 def inputs():return [{'name':p.name} for p in data.iterdir() if p.is_file()]
 @app.get('/api/input/{name}/preview')
 def input_preview(name:str):
  import cv2
  p=under(data,name);target=data/f'.preview_{name}.jpg'
  if p.suffix.lower() in c['input']['videos']:
   cap=cv2.VideoCapture(str(p));ok,im=cap.read();cap.release()
  else:im=cv2.imread(str(p));ok=im is not None
  if not ok:raise HTTPException(422,'preview unavailable')
  cv2.imwrite(str(target),im);return FileResponse(target)

 @app.post('/api/runs')
 def start(body:dict):
  try:p=under(data,body.get('input',''))
  except ValueError as e:raise HTTPException(400,str(e))
  if not p.exists():raise HTTPException(404,'input not found')
  choice=resolve(body.get('device','auto'));jid='job_'+uuid4().hex[:8];jobs[jid]={'job_id':jid,'status':'QUEUED','progress':0,'device':choice.dict()}
  def work():
   try:jobs[jid]['status']='RUNNING';rid=run(c,p,choice,lambda x:jobs[jid].update(x));jobs[jid].update(status='COMPLETED',run_id=rid,progress=1)
   except Exception as e:jobs[jid].update(status='FAILED',error=str(e))
  Thread(target=work,daemon=True).start();return jobs[jid]
 @app.get('/api/jobs/{jid}')
 def job(jid:str):
  if jid not in jobs:raise HTTPException(404,'job not found')
  return jobs[jid]
 @app.get('/api/runs')
 def runs():return [json.loads((p/'manifest.json').read_text()) for p in sorted(out.glob('run_*'),reverse=True) if (p/'manifest.json').exists()]
 @app.get('/api/review/{rid}')
 def review(rid:str):
  p=out/rid/'predictions.jsonl'
  if not p.exists():raise HTTPException(404,'run not found')
  anns={x['frame_id']:x for x in store.run(rid)};z=[]
  for line in p.open():x=json.loads(line);x['image']=f'/api/runs/{rid}/raw/{x["frame_id"]}';x['annotation']=anns.get(x['frame_id']);z.append(x)
  return z
 @app.post('/api/review/{rid}/annotations')
 def annotate(rid:str,a:AnnotationRequest):
  if rid!=a.run_id:raise HTTPException(400,'run mismatch')
  if a.label not in {'LEFT_ON','RIGHT_ON','BOTH_ON','NONE_ON','UNKNOWN'}:raise HTTPException(400,'invalid label')
  store.save(a.model_dump());return {'saved':True,'run_id':rid,'frame_id':a.frame_id}

 @app.get('/api/training/cues')
 def training_cues():
  items=reviewed(c,store);return {'items':items,'suggested':suggest(items),'baseline_model':model_info()}
 @app.post('/api/training')
 def training(body:dict):
  if len(body.get('selected_frames',[]))<2:raise HTTPException(400,'select at least two frames')
  jid='job_'+uuid4().hex[:8];jobs[jid]={'job_id':jid,'status':'QUEUED','progress':0}
  def work():
   try:jobs[jid]['status']='RUNNING';meta=train(c,store,body,lambda x:jobs[jid].update(x));jobs[jid].update(status='COMPLETED',progress=1,model=meta)
   except Exception as e:jobs[jid].update(status='FAILED',error=str(e))
  Thread(target=work,daemon=True).start();return jobs[jid]
 @app.get('/api/models')
 def models():
  p=Path('models/registry.json');return json.loads(p.read_text()) if p.exists() else {'default':'baseline','models':[]}

 @app.get('/api/runs/{rid}/frames/{fid}')
 def frame(rid:str,fid:int):
  p=out/rid/'overlays'/f'frame_{fid:06d}.jpg'
  if not p.exists():raise HTTPException(404,'frame not found')
  return FileResponse(p)
 @app.get('/api/runs/{rid}/raw/{fid}')
 def raw(rid:str,fid:int):
  p=out/rid/'frames'/f'frame_{fid:06d}.jpg'
  if not p.exists():raise HTTPException(404,'frame not found')
  return FileResponse(p)
 app.mount('/',StaticFiles(directory='frontend',html=True),name='ui');return app
