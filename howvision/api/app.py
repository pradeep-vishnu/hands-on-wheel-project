from pathlib import Path
from threading import Thread
from uuid import uuid4
from datetime import datetime,timezone
import json,cv2,psutil
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from howvision.config import load
from howvision.core.fs import safe_name,IMAGES,VIDEOS
from howvision.core.db import Store
from howvision.core.models import registry,save,parameters
from howvision.inference.engine import Engine
from howvision.training.trainer import train,records,validate
jobs={}
class Annotation(BaseModel):run_id:str;frame_id:int;label:str;original_label:str;sample_weight:float=Field(ge=0,le=2);hand_strokes:list=[];wheel_strokes:list=[]
class TrainRequest(BaseModel):selected_frames:list[str]=Field(min_length=2);hyperparameters:dict
def create_app(config='configs/default.yaml'):
    cfg=load(config);data=Path(cfg['paths']['data']);out=Path(cfg['paths']['output']);data.mkdir(exist_ok=True);out.mkdir(exist_ok=True);store=Store(cfg['paths']['database']);app=FastAPI(title='HOW Vision',version='1.1.0')
    @app.get('/api/health')
    def health():
        gpu={'name':'CPU','usage':None}
        try:
            import torch
            if torch.cuda.is_available():gpu={'name':torch.cuda.get_device_name(0),'usage':None}
        except:pass
        return {'status':'ACTIVE' if Path(cfg['paths']['hand_model']).exists() else 'SETUP','version':'1.1.0','cpu':psutil.cpu_percent(),'memory':psutil.virtual_memory().percent,'process_mb':psutil.Process().memory_info().rss/1048576,'gpu':gpu,'time':datetime.now(timezone.utc).isoformat()}
    @app.post('/api/input/upload')
    async def upload(file:UploadFile=File(...)):
        try:name=safe_name(file.filename or '')
        except ValueError:raise HTTPException(400,'invalid filename')
        p=data/name
        if p.suffix.lower() not in IMAGES|VIDEOS:raise HTTPException(400,'unsupported file')
        with p.open('wb') as f:
            while b:=await file.read(1048576):f.write(b)
        return {'name':name}
    @app.get('/api/input')
    def inputs():return [{'name':p.name,'type':'video' if p.suffix.lower() in VIDEOS else 'image','preview':f'/api/input/{p.name}/preview'} for p in sorted((x for x in data.iterdir() if x.is_file() and not x.name.startswith('.preview_')),key=lambda x:x.stat().st_mtime,reverse=True)]
    @app.get('/api/input/{name}/preview')
    def preview(name:str):
        try:p=data/safe_name(name)
        except ValueError:raise HTTPException(400,'invalid filename')
        target=data/f'.preview_{name}.jpg'
        if p.suffix.lower() in VIDEOS:cap=cv2.VideoCapture(str(p));ok,im=cap.read();cap.release()
        else:im=cv2.imread(str(p));ok=im is not None
        if not ok:raise HTTPException(422,'preview unavailable')
        cv2.imwrite(str(target),im);return FileResponse(target)
    @app.get('/api/models')
    def models():
        r=registry(cfg['paths']['registry']);hp=Path(cfg['paths']['hand_model']);return {'default':r['default'],'checkpoints':r['checkpoints'],'baseline':{'id':'baseline','parameter_count':parameters(24),'model_size_bytes':0},'perception_size_bytes':hp.stat().st_size if hp.exists() else 0}
    @app.post('/api/models/{mid}/default')
    def default(mid:str):
        r=registry(cfg['paths']['registry']);valid={'baseline'}|{x['id'] for x in r['checkpoints']}
        if mid not in valid:raise HTTPException(404,'model not found')
        r['default']=mid;save(cfg['paths']['registry'],r);return r
    @app.post('/api/runs')
    def run(body:dict):
        try:p=data/safe_name(body.get('input',''))
        except ValueError:raise HTTPException(400,'invalid input')
        mid=body.get('checkpoint','baseline');reg=registry(cfg['paths']['registry']);valid={'baseline'}|{x['id'] for x in reg['checkpoints']}
        if not p.exists():raise HTTPException(404,'input not found')
        if mid not in valid:raise HTTPException(400,'unknown checkpoint')
        jid='job_'+uuid4().hex[:8];jobs[jid]={'job_id':jid,'status':'QUEUED','progress':0}
        def work():
            try:jobs[jid]['status']='RUNNING';rid=Engine(cfg,mid).run(p,lambda x:jobs[jid].update(x));jobs[jid].update(status='COMPLETED',run_id=rid,progress=1)
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
        anns={(x['run_id'],x['frame_id']):x for x in store.all(rid)};result=[]
        for line in p.open():x=json.loads(line);x['image']=f'/api/runs/{rid}/frames/{x["frame_id"]}?view=raw';x['annotation']=anns.get((rid,x['frame_id']));result.append(x)
        return result
    @app.post('/api/review/{rid}')
    def annotate(rid:str,a:Annotation):
        if rid!=a.run_id:raise HTTPException(400,'run mismatch')
        if a.label not in {'LEFT_ON','RIGHT_ON','BOTH_ON','NONE_ON','UNKNOWN'}:raise HTTPException(400,'invalid label')
        store.save(a.model_dump());return {'saved':True}
    @app.get('/api/training/preview')
    def training_preview():return [{'id':f'{a["run_id"]}:{a["frame_id"]}','run_id':a['run_id'],'frame_id':a['frame_id'],'label':a['label'],'image':f'/api/runs/{a["run_id"]}/frames/{a["frame_id"]}?view=raw','hand_strokes':json.loads(a['hand_strokes']),'wheel_strokes':json.loads(a['wheel_strokes'])} for a,r in records(cfg,store)]
    @app.get('/api/training/defaults')
    def defaults():return cfg['training']
    @app.post('/api/training')
    def training(req:TrainRequest):
        try:validate(req.hyperparameters)
        except ValueError as e:raise HTTPException(422,str(e))
        jid='job_'+uuid4().hex[:8];jobs[jid]={'job_id':jid,'status':'QUEUED','progress':0}
        def work():
            try:jobs[jid]['status']='RUNNING';m=train(cfg,store,req.selected_frames,req.hyperparameters,lambda x:jobs[jid].update(x));jobs[jid].update(status='COMPLETED',progress=1,model=m)
            except Exception as e:jobs[jid].update(status='FAILED',error=str(e))
        Thread(target=work,daemon=True).start();return jobs[jid]
    @app.get('/api/runs/{rid}/frames/{fid}')
    def frame(rid:str,fid:int,view:str='overlay'):
        if view not in {'raw','overlay'}:raise HTTPException(400,'invalid view')
        p=out/rid/('frames' if view=='raw' else 'overlays')/f'frame_{fid:06d}.jpg'
        if not p.exists():raise HTTPException(404,'frame unavailable')
        return FileResponse(p)
    @app.get('/api/runs/{rid}/csv')
    def csv(rid:str):return FileResponse(out/rid/'predictions.csv',filename=f'{rid}.csv')
    app.mount('/',StaticFiles(directory='frontend',html=True),name='ui');return app
