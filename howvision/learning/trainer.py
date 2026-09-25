from pathlib import Path
import json,random,numpy as np
from howvision.core.fs import allocate
from howvision.core.models import register
LABELS=['LEFT_ON','RIGHT_ON','BOTH_ON','NONE_ON','UNKNOWN'];KEYS=['left_distance','right_distance','wheel_confidence','left_confidence','right_confidence']
def reviewed(c,store):
 out=[]
 for a in store.all():
  p=Path(c['paths']['output'])/a['run_id']/'predictions.jsonl'
  if p.exists():
   for line in p.open():
    r=json.loads(line)
    if r['frame_id']==a['frame_id']:out.append((a,r));break
 return out
def train(c,store,selected,emit):
 import torch
 from torch import nn
 chosen=set(selected);raw=[x for x in reviewed(c,store) if f"{x[0]['run_id']}:{x[0]['frame_id']}" in chosen]
 if len(raw)<2:raise RuntimeError('Select at least two reviewed frames')
 samples=[([r['features'][k] for k in KEYS],LABELS.index(a['label']),a['reward']) for a,r in raw];random.seed(c['training']['seed']);np.random.seed(c['training']['seed']);torch.manual_seed(c['training']['seed']);random.shuffle(samples);cut=max(1,int(.8*len(samples)));tr=samples[:cut];va=samples[cut:] or samples[-1:];m=nn.Sequential(nn.Linear(5,24),nn.ReLU(),nn.Linear(24,5));opt=torch.optim.Adam(m.parameters(),lr=c['training']['learning_rate'])
 for e in range(c['training']['epochs']):
  total=0
  for x,y,reward in tr:
   z=m(torch.tensor([x],dtype=torch.float32));ce=nn.functional.cross_entropy(z,torch.tensor([y]));policy=-torch.log_softmax(z,1)[0,y]*reward;entropy=-(torch.softmax(z,1)*torch.log_softmax(z,1)).sum();loss=ce+policy-c['training']['entropy_beta']*entropy;opt.zero_grad();loss.backward();opt.step();total+=loss.item()
  with torch.no_grad():acc=sum(m(torch.tensor([x],dtype=torch.float32)).argmax(1).item()==y for x,y,_ in va)/len(va)
  emit({'stage':'TRAINING','epoch':e+1,'epochs':c['training']['epochs'],'loss':total/len(tr),'val_accuracy':acc,'progress':(e+1)/c['training']['epochs'],'selected_samples':len(samples)})
 _,d=allocate('models/versions','candidate');tmp=d/'model.pt';torch.save({'state':m.state_dict()},tmp);mid=register(c['paths']['registry'],tmp,{'validation_accuracy':acc,'samples':len(samples)});final=d.parent/mid;d.rename(final);reg=json.loads(Path(c['paths']['registry']).read_text());reg['checkpoints'][-1]['path']=str(final/'model.pt');Path(c['paths']['registry']).write_text(json.dumps(reg,indent=2));return mid,reg['checkpoints'][-1]['metrics']
