from pathlib import Path
from datetime import datetime,timezone
import json,random,numpy as np
from howvision.core.fs import atomic_json
from howvision.core.models import registry,save,parameters
LABELS=['LEFT_ON','RIGHT_ON','BOTH_ON','NONE_ON','UNKNOWN']; KEYS=['left_distance','right_distance','wheel_confidence','left_confidence','right_confidence']
def validate(h):
    limits={'epochs':(1,500),'learning_rate':(1e-6,1),'batch_size':(1,1024),'hidden_width':(4,1024),'weight_decay':(0,1),'entropy_coefficient':(0,1),'validation_fraction':(.05,.5),'random_seed':(0,2147483647),'early_stopping_patience':(1,100),'gradient_clip':(.01,100)}
    for k,(a,b) in limits.items():
        if k not in h or not a<=h[k]<=b:raise ValueError(f'{k} outside [{a}, {b}]')
def records(cfg,store):
    out=[]
    for a in store.all():
        p=Path(cfg['paths']['output'])/a['run_id']/'predictions.jsonl'
        if not p.exists():continue
        for line in p.open():
            r=json.loads(line)
            if r['frame_id']==a['frame_id']:out.append((a,r));break
    return out
def train(cfg,store,selected,h,emit):
    import torch
    from torch import nn
    validate(h);chosen=set(selected);raw=[x for x in records(cfg,store) if f"{x[0]['run_id']}:{x[0]['frame_id']}" in chosen]
    if len(raw)<2:raise RuntimeError('select at least two reviewed frames')
    data=[]
    for a,r in raw:
        f=r['features'];data.append((np.array([f[k] for k in KEYS],np.float32),LABELS.index(a['label']),a['sample_weight'],a,r))
    random.seed(h['random_seed']);np.random.seed(h['random_seed']);torch.manual_seed(h['random_seed']);random.shuffle(data);cut=max(1,min(len(data)-1,int(len(data)*(1-h['validation_fraction']))));tr,va=data[:cut],data[cut:];w=h['hidden_width'];model=nn.Sequential(nn.Linear(5,w),nn.ReLU(),nn.Linear(w,5));opt=torch.optim.Adam(model.parameters(),lr=h['learning_rate'],weight_decay=h['weight_decay']);history=[];best=None;best_loss=float('inf');stale=0;previous={}
    for epoch in range(1,h['epochs']+1):
        model.train();losses=[];random.shuffle(tr)
        for s in range(0,len(tr),h['batch_size']):
            b=tr[s:s+h['batch_size']];x=torch.tensor(np.stack([q[0] for q in b]));y=torch.tensor([q[1] for q in b]);weights=torch.tensor([q[2] for q in b]);logits=model(x);ce=nn.functional.cross_entropy(logits,y,reduction='none');entropy=-(torch.softmax(logits,1)*torch.log_softmax(logits,1)).sum(1);loss=(ce*weights).mean()-h['entropy_coefficient']*entropy.mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),h['gradient_clip']);opt.step();losses.append(float(loss))
        model.eval();vl=[];correct=0;examples=[];matrix=[[0]*5 for _ in range(5)]
        with torch.no_grad():
            for x,y,weight,a,r in va:
                logits=model(torch.tensor([x]));prob=torch.softmax(logits,1)[0];pred=int(prob.argmax());vl.append(float(nn.functional.cross_entropy(logits,torch.tensor([y]))));correct+=pred==y;matrix[y][pred]+=1;key=f"{a['run_id']}:{a['frame_id']}";examples.append({'id':key,'image':f'/api/runs/{a["run_id"]}/frames/{a["frame_id"]}?view=raw','manual_label':LABELS[y],'original_prediction':r['raw']['state'],'previous_prediction':previous.get(key,r['raw']['state']),'current_prediction':LABELS[pred],'current_confidence':float(prob.max())});previous[key]=LABELS[pred]
        val_loss=sum(vl)/len(vl);acc=correct/len(va);row={'epoch':epoch,'train_loss':sum(losses)/len(losses),'validation_loss':val_loss,'validation_accuracy':acc,'accuracy_delta':acc-(history[-1]['validation_accuracy'] if history else 0),'confusion_matrix':matrix,'examples':examples};history.append(row);emit({'status':'RUNNING','progress':epoch/h['epochs'],**row})
        if val_loss<best_loss:best_loss=val_loss;best={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        if stale>=h['early_stopping_patience']:break
    model.load_state_dict(best);tag=datetime.now(timezone.utc).strftime('how_%Y%m%d_%H%M%S_utc');d=Path('models/versions')/tag;d.mkdir(parents=True);path=d/'model.pt';torch.save({'state':model.state_dict(),'hidden_width':w,'labels':LABELS},path);params=sum(p.numel() for p in model.parameters());meta={'id':tag,'path':str(path),'created_at':datetime.now(timezone.utc).isoformat(),'hyperparameters':h,'history':history,'parameter_count':params,'baseline_parameter_count':parameters(24),'parameter_delta':params-parameters(24),'model_size_bytes':path.stat().st_size,'baseline_model_size_bytes':0};atomic_json(d/'metadata.json',meta);reg=registry(cfg['paths']['registry']);reg['checkpoints'].append(meta);save(cfg['paths']['registry'],reg);return meta
