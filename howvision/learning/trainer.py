from pathlib import Path
import json,random,time
import numpy as np
from howvision.core.fs import allocate,atomic_json,sha
LABELS=['LEFT_ON','RIGHT_ON','BOTH_ON','NONE_ON','UNKNOWN']; FEATURES=['left_distance','right_distance','wheel_confidence','left_confidence','right_confidence']
def vector(rec):
 f=rec.get('features',{}); hs={x.get('side','').lower():x for x in rec.get('hands',[])}; return [float(f.get('left_distance',2)),float(f.get('right_distance',2)),float(rec.get('wheel',{}).get('confidence',0)),float(hs.get('left',{}).get('confidence',0)),float(hs.get('right',{}).get('confidence',0))]
def train(cfg,store,progress=lambda x:None):
 import torch
 from torch import nn
 anns=store.annotations(); samples=[]
 for a in anns:
  p=Path(cfg['paths']['output'])/a['run_id']/'predictions.jsonl'
  if not p.exists():continue
  for line in p.open():
   r=json.loads(line)
   if r['frame_id']==a['frame_id']: samples.append((vector(r),LABELS.index(a['label']),float(a['reward'])));break
 if len(samples)<2: raise RuntimeError('At least two confirmed reviewed frames are required')
 seed=cfg['training']['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); random.shuffle(samples); cut=max(1,int(len(samples)*.8)); training=samples[:cut]; validation=samples[cut:] or samples[-1:]
 model=nn.Sequential(nn.Linear(5,24),nn.ReLU(),nn.Linear(24,5)); opt=torch.optim.Adam(model.parameters(),lr=cfg['training']['learning_rate'])
 for epoch in range(cfg['training']['epochs']):
  random.shuffle(training); total=0
  for x,y,reward in training:
   logits=model(torch.tensor([x],dtype=torch.float32)); ce=nn.functional.cross_entropy(logits,torch.tensor([y])); policy=-torch.log_softmax(logits,1)[0,y]*reward; entropy=-(torch.softmax(logits,1)*torch.log_softmax(logits,1)).sum(); loss=ce+policy-cfg['training']['entropy_beta']*entropy; opt.zero_grad(); loss.backward();opt.step();total+=loss.item()
  with torch.no_grad(): acc=sum(int(model(torch.tensor([x],dtype=torch.float32)).argmax(1).item()==y) for x,y,_ in validation)/len(validation)
  progress({'stage':'TRAINING','epoch':epoch+1,'epochs':cfg['training']['epochs'],'loss':total/max(1,len(training)),'val_accuracy':acc,'progress':(epoch+1)/cfg['training']['epochs']})
 mid,md=allocate('models/versions','how'); path=md/'model.pt'; torch.save({'state_dict':model.state_dict(),'labels':LABELS,'features':FEATURES},path); metrics={'validation_accuracy':acc,'samples':len(samples),'method':'reward-weighted supervised policy optimization'}; atomic_json(md/'metadata.json',{'model_id':mid,'path':str(path),'created_at':time.time(),'metrics':metrics,'hash':sha(path),'active':False}); return mid,metrics
