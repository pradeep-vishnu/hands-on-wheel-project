from pathlib import Path
from threading import Thread
from uuid import uuid4
import json,cv2
from fastapi import FastAPI,UploadFile,File,HTTPException,Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from howvision.config import load
from howvision.core.fs import safe,IMAGES,VIDEOS
from howvision.core.db import Store
from howvision.core.models import read,write
from howvision.pipeline.engine import Engine
from howvision.learning.trainer import train,samples
jobs={}
class Ann(BaseModel):run_id:str;frame_id:int;label:str;original_label:str;reward:float=1;hand_strokes:list=[];wheel_strokes:list=[];flags:list=[]
def create_app():
 c=load();data=Path(c['paths']['data']);out=Path(c['paths']['output']);data.mkdir(exist_ok=True);out.mkdir(exist_ok=True);store=Store(c['paths']['database']);app=FastAPI(title='Hands On Wheel Project',version='0.6.0')
 @app.get('/api/health')
 def health():return {'status':'READY' if Path(c['paths']['hand_model']).exists() else 'SETUP_REQUIRED','brush_radius_px':c['review']['brush_radius_px']}
 @app.post('/api/input/upload')
 async def upload(file:UploadFile=File(...)):
  name=safe(file.filename or '');p=data/name
  if p.suffix.lower() not in IMAGES|VIDEOS:raise HTTPException(400,'unsupported')
  with p.open('wb') as f:
   while b:=await file.read(1048576):f.write(b)
  return {'name':name,'preview':f'/api/input/{name}/preview'}
 @app.get('/api/input')
 def inputs():return [{'name':p.name,'preview':f'/api/input/{p.name}/preview','type':'video' if p.suffix.lower() in VIDEOS else 'image'} for p in sorted((x for x in data.iterdir() if x.is_file() and not x.name.startswith('.preview_')),key=lambda p:p.stat().st_mtime,reverse=True)]
 @app.get('/api/input/{name}/preview')
 def preview(name:str):
  p=data/safe(name);target=data/f'.preview_{name}.jpg'
  if p.suffix.lower() in VIDEOS:cap=cv2.VideoCapture(str(p));ok,im=cap.read();cap.release()
  else:im=cv2.imread(str(p));ok=im is not None
  if not ok:raise HTTPException(422,'preview unavailable')
  cv2.imwrite(str(target),im);return FileResponse(target)
 @app.get('/api/models')
 def models():return read(c['paths']['registry'])
 @app.post('/api/models/{mid}/default')
 def default(mid:str):
  r=read(c['paths']['registry']);valid={'baseline'}|{x['id'] for x in r['checkpoints']}
  if mid not in valid:raise HTTPException(404,'not found')
  r['default']=mid;write(c['paths']['registry'],r);return r
 @app.post('/api/runs')
 def run(b:dict):
  p=data/safe(b.get('input',''));mid=b.get('checkpoint',read(c['paths']['registry'])['default']);jid='job_'+uuid4().hex[:8];jobs[jid]={'job_id':jid,'status':'QUEUED','progress':0}
  def work():
   try:jobs[jid]['status']='RUNNING';rid=Engine(c,mid).run(p,lambda x:jobs[jid].update(x));jobs[jid].update(status='COMPLETED',run_id=rid,progress=1)
   except Exception as e:jobs[jid].update(status='FAILED',error=str(e))
  Thread(target=work,daemon=True).start();return jobs[jid]
 @app.get('/api/jobs/{jid}')
 def job(jid:str):return jobs[jid]
 @app.get('/api/runs')
 def runs():return [json.loads((p/'manifest.json').read_text()) for p in sorted(out.glob('run_*'),reverse=True) if (p/'manifest.json').exists()]
 @app.get('/api/review/{rid}')
 def review(rid:str):
  anns={(x['run_id'],x['frame_id']):x for x in store.all(rid)};z=[]
  for line in (out/rid/'predictions.jsonl').open():
   x=json.loads(line);x['image']=f'/api/runs/{rid}/frames/{x["frame_id"]}?view=raw';x['annotation']=anns.get((rid,x['frame_id']));z.append(x)
  return z
 @app.post('/api/review/{rid}')
 def annotation(rid:str,a:Ann):store.save(a.model_dump());return {'saved':True}
 @app.get('/api/training/preview')
 def training_preview(limit:int=Query(12,ge=1,le=50)):
  z=[]
  for a,r in samples(c,store)[:limit]:z.append({'run_id':a['run_id'],'frame_id':a['frame_id'],'label':a['label'],'image':f'/api/runs/{a["run_id"]}/frames/{a["frame_id"]}?view=raw','hand_strokes':json.loads(a['hand_strokes']),'wheel_strokes':json.loads(a['wheel_strokes'])})
  return z
 @app.post('/api/training')
 def training():
  jid='job_'+uuid4().hex[:8];jobs[jid]={'job_id':jid,'status':'QUEUED','progress':0}
  def work():
   try:jobs[jid]['status']='RUNNING';mid,m=train(c,store,lambda x:jobs[jid].update(x));jobs[jid].update(status='COMPLETED',model_id=mid,metrics=m,progress=1)
   except Exception as e:jobs[jid].update(status='FAILED',error=str(e))
  Thread(target=work,daemon=True).start();return jobs[jid]
 @app.get('/api/runs/{rid}/frames/{fid}')
 def frame(rid:str,fid:int,view:str='overlay'):
  folder='frames' if view=='raw' else 'overlays';return FileResponse(out/rid/folder/f'frame_{fid:06d}.jpg')
 @app.get('/api/runs/{rid}/csv')
 def csv(rid:str):return FileResponse(out/rid/'predictions.csv',filename=f'{rid}.csv')
 app.mount('/',StaticFiles(directory='frontend',html=True),name='ui');return app
