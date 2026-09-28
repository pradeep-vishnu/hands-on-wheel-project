from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import json,random
from howvision.core.device import resolve
LABELS=['LEFT_ON','RIGHT_ON','BOTH_ON','NONE_ON','UNKNOWN'];FEATURES=['wheel_confidence','left_distance','right_distance','left_confidence','right_confidence']
def items(cfg,store):
 out=[]
 for a in store.all():
  p=Path(cfg['paths']['output'])/a['run_id']/'predictions.jsonl'
  if not p.exists():continue
  for line in p.open():
   r=json.loads(line)
   if r['frame_id']==a['frame_id']:out.append({'id':f"{a['run_id']}:{a['frame_id']}",'annotation':a,'record':r,'image':f"/api/runs/{a['run_id']}/raw/{a['frame_id']}"});break
 return out
def suggest(data):
 n=len(data);counts=Counter(x['annotation']['label'] for x in data);imb=max(counts.values(),default=1)/max(1,min(counts.values(),default=1));return {'epochs':40 if n<40 else 24,'learning_rate':.0005 if n<30 else .001,'batch_size':min(16,max(2,n//4 or 2)),'hidden_width':24 if n<100 else 48,'weight_decay':.0001,'validation_fraction':.2,'early_stopping_patience':6,'gradient_clip':1.,'seed':42,'reason':f'{n} reviewed cues, imbalance ratio {imb:.2f}'}
def model_stats(hidden,path=None):return {'parameters':5*hidden+hidden+hidden*5+5,'bytes':Path(path).stat().st_size if path and Path(path).exists() else 0}
def train(cfg,store,req,emit):
 import torch
 from torch import nn
 dev=resolve(req.get('device','auto'));d=torch.device(dev.actual);all_data=items(cfg,store);selected=[x for x in all_data if x['id'] in set(req['selected_frames'])]
 if len(selected)<2:raise RuntimeError('select at least two reviewed cues')
 hp=req['hyperparameters'];random.seed(int(hp['seed']));torch.manual_seed(int(hp['seed']));random.shuffle(selected);cut=max(1,int(len(selected)*(1-float(hp['validation_fraction']))));tr=selected[:cut];va=selected[cut:] or selected[-1:];hidden=int(hp['hidden_width']);m=nn.Sequential(nn.Linear(5,hidden),nn.ReLU(),nn.Linear(hidden,5)).to(d);opt=torch.optim.AdamW(m.parameters(),lr=float(hp['learning_rate']),weight_decay=float(hp['weight_decay']));history=[];best=-1;stale=0;best_state=None
 def feature(r):return [float(r['features'].get(k,0 if 'confidence' in k else 2)) for k in FEATURES]
 for epoch in range(int(hp['epochs'])):
  total=0
  for x in tr:
   z=m(torch.tensor([feature(x['record'])],dtype=torch.float32,device=d));y=torch.tensor([LABELS.index(x['annotation']['label'])],device=d);loss=nn.functional.cross_entropy(z,y);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),float(hp['gradient_clip']));opt.step();total+=float(loss.detach().cpu())
  examples=[];correct=0
  with torch.no_grad():
   for x in va:
    p=torch.softmax(m(torch.tensor([feature(x['record'])],dtype=torch.float32,device=d)),1)[0];pred=LABELS[int(p.argmax())];manual=x['annotation']['label'];correct+=pred==manual;examples.append({'id':x['id'],'manual_label':manual,'before_prediction':x['record']['raw_prediction'],'after_prediction':pred,'confidence':float(p.max().cpu()),'image':x['image']})
  acc=correct/len(va);row={'epoch':epoch+1,'train_loss':total/max(1,len(tr)),'validation_accuracy':acc,'accuracy_delta':acc-(history[-1]['validation_accuracy'] if history else 0),'examples':examples};history.append(row);emit({'status':'TRAINING','progress':(epoch+1)/int(hp['epochs']),**row})
  if acc>best:best=acc;stale=0;best_state={k:v.detach().cpu() for k,v in m.state_dict().items()}
  else:stale+=1
  if stale>=int(hp['early_stopping_patience']):break
 tag=datetime.now(timezone.utc).strftime('how_%Y%m%d_%H%M%S_utc');root=Path('models/versions')/tag;root.mkdir(parents=True);path=root/'model.pt';torch.save({'state':best_state or m.state_dict(),'hidden_width':hidden,'labels':LABELS,'features':FEATURES},path);meta={'id':tag,'created_at':datetime.now(timezone.utc).isoformat(),'parent':'baseline','path':str(path),'metrics':{'validation_accuracy':best},'history':history,'hyperparameters':hp,'device':dev.dict(),'model':model_stats(hidden,path),'training_samples':len(selected)};(root/'metadata.json').write_text(json.dumps(meta,indent=2));reg=Path(cfg['paths']['registry']);reg.parent.mkdir(exist_ok=True);r=json.loads(reg.read_text()) if reg.exists() else {'default':'baseline','models':[]};r['models'].append(meta);reg.write_text(json.dumps(r,indent=2));return meta
