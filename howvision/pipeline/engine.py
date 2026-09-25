from pathlib import Path
from collections import deque,Counter
import cv2,json,csv,time,math,numpy as np,psutil
from howvision.core.fs import allocate,atomic,sha,VIDEOS
LABELS=['LEFT_ON','RIGHT_ON','BOTH_ON','NONE_ON','UNKNOWN']
class Engine:
 def __init__(self,c,checkpoint='baseline'):
  self.c=c;self.checkpoint=checkpoint;import mediapipe as mp;from mediapipe.tasks import python;from mediapipe.tasks.python import vision
  opts=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(Path(c['paths']['hand_model']))),running_mode=vision.RunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=c['perception']['hand_confidence']);self.mp=mp;self.det=vision.HandLandmarker.create_from_options(opts);self.model=None
  if checkpoint!='baseline':
   import torch;from torch import nn
   reg=json.loads(Path(c['paths']['registry']).read_text());blob=torch.load(next(x for x in reg['checkpoints'] if x['id']==checkpoint)['path'],map_location='cpu');self.model=nn.Sequential(nn.Linear(5,24),nn.ReLU(),nn.Linear(24,5));self.model.load_state_dict(blob['state']);self.model.eval()
 def run(self,p,emit):
  rid,rd=allocate(self.c['paths']['output'],'run');(rd/'overlays').mkdir();(rd/'frames').mkdir();(rd/'logs').mkdir();seq=[];ext=p.suffix.lower()
  if ext in VIDEOS:
   cap=cv2.VideoCapture(str(p));fps=cap.get(cv2.CAP_PROP_FPS) or 25
   while True:
    ok,im=cap.read()
    if not ok:break
    seq.append((im,len(seq),float(cap.get(cv2.CAP_PROP_POS_MSEC))))
   cap.release()
  else:seq=[(cv2.imread(str(p)),0,0.)];fps=0
  start=time.monotonic();proc=psutil.Process();rows=[];writer=None;window=deque(maxlen=self.c['temporal']['smoothing_window']);jf=(rd/'predictions.jsonl').open('w')
  for im,i,ts in seq:
   hands=self.hands(im);wheel=self.wheel(im);f=self.features(hands,wheel);raw,conf=self.classify(f,hands,wheel);window.append(raw if conf>=self.c['temporal']['confidence_threshold'] else 'UNKNOWN');smooth=Counter(window).most_common(1)[0][0];status='HOW_ON' if smooth in LABELS[:3] else 'HOW_OFF' if smooth=='NONE_ON' else 'UNKNOWN';rec={'run_id':rid,'source_file':p.name,'input_type':'video' if ext in VIDEOS else 'image','frame_id':i,'sequence_index':i,'timestamp_ms':ts,'timestamp_native':{'kind':'opencv_pos_msec' if ext in VIDEOS else 'image','value':ts},'hands':hands,'wheel':wheel,'features':f,'raw_prediction':raw,'how_state':smooth,'how_status':status,'confidence':conf,'checkpoint':self.checkpoint};jf.write(json.dumps(rec)+'\n');rows.append(rec);cv2.imwrite(str(rd/'frames'/f'frame_{i:06d}.jpg'),im);ov=self.render(im,rec);cv2.imwrite(str(rd/'overlays'/f'frame_{i:06d}.jpg'),ov)
   if ext in VIDEOS:
    if writer is None:writer=cv2.VideoWriter(str(rd/'output.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(im.shape[1],im.shape[0]))
    writer.write(ov)
   else:cv2.imwrite(str(rd/f'output{ext}'),ov)
   elapsed=max(time.monotonic()-start,.001);emit({'run_id':rid,'frame':i+1,'total':len(seq),'progress':(i+1)/len(seq),'fps':(i+1)/elapsed,'cpu':psutil.cpu_percent(),'memory':psutil.virtual_memory().percent,'process_mb':proc.memory_info().rss/1048576,'label':status,'confidence':conf,'preview':f'/api/runs/{rid}/frames/{i}?view=overlay'})
  jf.close();
  if writer:writer.release()
  cols=['run_id','source_file','frame_id','sequence_index','timestamp_ms','raw_prediction','how_state','how_status','confidence','checkpoint']
  with (rd/'predictions.csv').open('w',newline='') as q:w=csv.DictWriter(q,fieldnames=cols);w.writeheader();w.writerows({k:r[k] for k in cols} for r in rows)
  elapsed=time.monotonic()-start;atomic(rd/'metrics.json',{'frames_processed':len(rows),'seconds':elapsed,'average_fps':len(rows)/max(elapsed,.001)});atomic(rd/'config.json',self.c);atomic(rd/'manifest.json',{'run_id':rid,'status':'COMPLETED','input':p.name,'input_type':'video' if ext in VIDEOS else 'image','input_hash':sha(p),'frames':len(rows),'checkpoint':self.checkpoint,'model_hash':sha(self.c['paths']['hand_model'])});return rid
 def hands(self,im):
  r=self.det.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)));out=[]
  for i,l in enumerate(r.hand_landmarks):
   c=r.handedness[i][0];pts=[[float(x.x),float(x.y)] for x in l];poly=cv2.convexHull(np.array(pts,np.float32)).reshape(-1,2).tolist();out.append({'side':(c.category_name or 'UNKNOWN').upper(),'confidence':float(c.score),'landmarks':pts,'polygon':poly})
  return out
 def wheel(self,im):
  g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);cs,_=cv2.findContours(cv2.Canny(g,60,160),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);h,w=g.shape;best=None
  for c in cs:
   if len(c)<20:continue
   (x,y),(a,b),ang=cv2.fitEllipse(c);ar=np.pi*a*b/4/(w*h);ratio=min(a,b)/max(a,b)
   if .04<ar<.78 and ratio>.42 and (best is None or ratio>best[0]):best=(ratio,x/w,y/h,a/w,b/h,ang)
  if not best:return {'visible':False,'confidence':0,'polygon':[]}
  s,x,y,a,b,ang=best;poly=cv2.ellipse2Poly((int(x*w),int(y*h)),(int(a*w/2),int(b*h/2)),int(ang),0,360,10);return {'visible':True,'confidence':s,'center':[x,y],'axes':[a,b],'polygon':[[u/w,v/h] for u,v in poly]}
 def features(self,hands,w):
  f={'wheel_confidence':w.get('confidence',0),'left_distance':2,'right_distance':2,'left_confidence':0,'right_confidence':0}
  if w.get('center'):
   cx,cy=w['center'];rx,ry=w['axes'][0]/2,w['axes'][1]/2
   for h in hands:
    s=h['side'].lower();f[s+'_distance']=min(abs(math.hypot((x-cx)/rx,(y-cy)/ry)-1) for x,y in h['landmarks']);f[s+'_confidence']=h['confidence']
  return f
 def classify(self,f,h,w):
  if self.model:
   import torch;v=[f[k] for k in ['left_distance','right_distance','wheel_confidence','left_confidence','right_confidence']];p=torch.softmax(self.model(torch.tensor([v],dtype=torch.float32)),1)[0];return LABELS[int(p.argmax())],float(p.max())
  if not w.get('visible'):return 'UNKNOWN',0
  l=f['left_distance']<self.c['classifier']['contact_distance'];r=f['right_distance']<self.c['classifier']['contact_distance'];return ('BOTH_ON' if l and r else 'LEFT_ON' if l else 'RIGHT_ON' if r else 'NONE_ON' if h else 'UNKNOWN'),min(1.,w['confidence']*max(f['left_confidence'],f['right_confidence'],.2))
 def render(self,im,r):
  o=im.copy();h,w=o.shape[:2]
  for q in r['hands']:cv2.polylines(o,[np.array([[int(x*w),int(y*h)] for x,y in q['polygon']])],True,(255,80,205),2)
  p=np.array([[int(x*w),int(y*h)] for x,y in r['wheel']['polygon']]);
  if len(p):cv2.polylines(o,[p],True,(65,210,255),2)
  return o
