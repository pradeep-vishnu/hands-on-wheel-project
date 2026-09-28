from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json, random
from howvision.core.device import resolve
LABELS=['LEFT_ON','RIGHT_ON','BOTH_ON','NONE_ON','UNKNOWN']
FEATURES=['wheel_confidence','left_distance','right_distance','left_confidence','right_confidence']
def _feature(record):
 f=record.get('features',{});return [float(f.get(k,0 if 'confidence' in k else 2)) for k in FEATURES]
def reviewed(cfg,store):
 out=[]
 for a in store.all():
  p=Path(cfg['paths']['output'])/a['run_id']/'predictions.jsonl'
  if not p.exists():continue
  for line in p.open():
   r=json.loads(line)
   if r['frame_id']==a['frame_id']:
    out.append({'id':f"{a['run_id']}:{a['frame_id']}",'annotation':a,'record':r,'image':f"/api/runs/{a['run_id']}/raw/{a['frame_id']}"});break
 return out
def suggest(items):
 n=len(items);counts=Counter(x['annotation']['label'] for x in items);imb=max(counts.values(),default=1)/max(1,min(counts.values(),default=1));return {'epochs':40 if n<50 else 24,'learning_rate':0.0005 if n<30 else 0.001,'batch_size':min(16,max(2,n//4 or 2)),'hidden_width':24 if n<100 else 48,'weight_decay':0.0001,'validation_fraction':0.2,'early_stopping_patience':6,'gradient_clip':1.0,'reason':f'{n} reviewed samples; class imbalance ratio {imb:.2f}'}
def model_info(path=None,hidden=24):
 params=5*hidden+hidden+hidden*5+5;size=Path(path).stat().st_size if path and Path(path).exists() else 0;return {'parameters':params,'bytes':size}
def train(cfg,store,req,emit):
 import torch
 from torch import nn
 device=resolve(req.get('device','auto'));d=torch.device(device.actual);all_items=reviewed(cfg,store);ids=set(req['selected_frames']);items=[x for x in all_items if x['id'] in ids]
 if len(items)<2:raise RuntimeError('Select at least two reviewed frames')
 hp=req['hyperparameters'];seed=int(hp['seed']);random.seed(seed);torch.manual_seed(seed);random.shuffle(items);cut=max(1,int(len(items)*(1-float(hp['validation_fraction']))));tr=items[:cut];va=items[cut:] or items[-1:];hidden=int(hp['hidden_width']);model=nn.Sequential(nn.Linear(5,hidden),nn.ReLU(),nn.Linear(hidden,5)).to(d);opt=torch.optim.AdamW(model.parameters(),lr=float(hp['learning_rate']),weight_decay=float(hp['weight_decay']));history=[];best=-1;stale=0;best_state=None
 def evaluate(epoch):
  rows=[];correct=0
  with torch.no_grad():
   for item in va:
    x=torch.tensor([_feature(item['record'])],dtype=torch.float32,device=d);p=torch.softmax(model(x),1)[0];pred=LABELS[int(p.argmax())];manual=item['annotation']['label'];correct+=pred==manual;rows.append({'id':item['id'],'manual_label':manual,'before_prediction':item['record'].get('raw_prediction','UNKNOWN'),'after_prediction':pred,'after_confidence':float(p.max().cpu())})
  return correct/len(va),rows
 epochs=int(hp['epochs'])
 for epoch in range(epochs):
  total=0
  for item in tr:
   x=torch.tensor([_feature(item['record'])],dtype=torch.float32,device=d);y=torch.tensor([LABELS.index(item['annotation']['label'])],device=d);loss=nn.functional.cross_entropy(model(x),y);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),float(hp['gradient_clip']));opt.step();total+=float(loss.detach().cpu())
  acc,examples=evaluate(epoch);row={'epoch':epoch+1,'train_loss':total/max(1,len(tr)),'validation_accuracy':acc,'accuracy_delta':acc-(history[-1]['validation_accuracy'] if history else 0),'examples':examples};history.append(row);emit({'status':'TRAINING','progress':(epoch+1)/epochs,**row})
  if acc>best:best=acc;stale=0;best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
  else:stale+=1
  if stale>=int(hp['early_stopping_patience']):break
 tag=datetime.now(timezone.utc).strftime('how_%Y%m%d_%H%M%S_utc');root=Path('models/versions')/tag;root.mkdir(parents=True);path=root/'model.pt';torch.save({'state':best_state or model.state_dict(),'hidden_width':hidden,'labels':LABELS,'features':FEATURES},path);meta={'id':tag,'created_at':datetime.now(timezone.utc).isoformat(),'parent':'baseline','path':str(path),'metrics':{'validation_accuracy':best},'history':history,'hyperparameters':hp,'device':device.dict(),'model':model_info(path,hidden),'training_samples':len(items)};(root/'metadata.json').write_text(json.dumps(meta,indent=2));reg=Path('models/registry.json');r=json.loads(reg.read_text()) if reg.exists() else {'default':'baseline','models':[]};r['models'].append(meta);reg.write_text(json.dumps(r,indent=2));return meta
