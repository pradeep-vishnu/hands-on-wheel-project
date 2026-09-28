from pathlib import Path
from collections import Counter,deque
from datetime import datetime,timezone
import cv2,csv,json,math,time,numpy as np
from howvision.core.fs import allocate,atomic,sha
class Engine:
 def __init__(self,cfg,device):self.c=cfg;self.device=device;self.hand=None
 def init_hand(self):
  try:
   import mediapipe as mp
   from mediapipe.tasks import python
   from mediapipe.tasks.python import vision
   p=Path(self.c['paths']['hand_model'])
   if not p.exists():return
   o=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(p)),running_mode=vision.RunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=self.c['perception']['hand_confidence']);self.mp=mp;self.hand=vision.HandLandmarker.create_from_options(o)
  except Exception:self.hand=None
 def run(self,path,emit):
  self.init_hand();rid,rd=allocate(self.c['paths']['output']);[(rd/x).mkdir() for x in ['frames','overlays','logs']];seq=[];ext=path.suffix.lower()
  if ext in self.c['input']['videos']:
   cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS) or 25
   while True:
    ok,im=cap.read()
    if not ok:break
    seq.append((im,len(seq),float(cap.get(cv2.CAP_PROP_POS_MSEC))))
   cap.release()
  else:seq=[(cv2.imread(str(path)),0,0.0)];fps=0
  created=datetime.now(timezone.utc).isoformat();rows=[];writer=None;window=deque(maxlen=self.c['temporal']['smoothing_window']);start=time.time()
  with (rd/'predictions.jsonl').open('w') as rich:
   for im,i,ts in seq:
    if im is None:continue
    hands=self.hands(im);wheel=self.wheel(im);features=self.features(hands,wheel);raw,conf=self.classify(features,hands,wheel);window.append(raw if conf>=self.c['temporal']['confidence_threshold'] else 'UNKNOWN');state=Counter(window).most_common(1)[0][0];status='HOW_ON' if state in {'LEFT_ON','RIGHT_ON','BOTH_ON'} else 'HOW_OFF' if state=='NONE_ON' else 'UNKNOWN';r={'run_id':rid,'source_file':path.name,'frame_id':i,'timestamp_ms':ts,'hands':hands,'wheel':wheel,'features':features,'raw_prediction':raw,'how_state':state,'how_status':status,'confidence':conf,'device':self.device.dict()};rows.append(r);rich.write(json.dumps(r)+'\n');cv2.imwrite(str(rd/'frames'/f'frame_{i:06d}.jpg'),im);ov=self.render(im,r);cv2.imwrite(str(rd/'overlays'/f'frame_{i:06d}.jpg'),ov)
    if ext in self.c['input']['videos']:
     if writer is None:writer=cv2.VideoWriter(str(rd/'output.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(im.shape[1],im.shape[0]))
     writer.write(ov)
    else:cv2.imwrite(str(rd/f'output{ext}'),ov)
    emit({'status':'RUNNING','run_id':rid,'processed':i+1,'total':len(seq),'progress':(i+1)/max(1,len(seq)),'how_status':status,'confidence':conf,'preview':f'/api/runs/{rid}/frames/{i}','device':self.device.dict()})
  if writer:writer.release()
  cols=['run_id','source_file','frame_id','timestamp_ms','raw_prediction','how_state','how_status','confidence']
  with (rd/'predictions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:r[k] for k in cols} for r in rows)
  manifest={'run_id':rid,'status':'COMPLETED','created_at':created,'completed_at':datetime.now(timezone.utc).isoformat(),'input':path.name,'input_hash':sha(path),'frames':len(rows),'requested_device':self.device.requested,'actual_device':self.device.actual,'device_name':self.device.name,'fallback_reason':self.device.fallback_reason};atomic(rd/'manifest.json',manifest);atomic(rd/'metrics.json',{'frames':len(rows),'seconds':time.time()-start});return rid
 def hands(self,im):
  if self.hand is None:return []
  rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);res=self.hand.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=rgb));out=[]
  for i,lms in enumerate(res.hand_landmarks):
   cat=res.handedness[i][0];pts=[[float(x.x),float(x.y),float(x.z)] for x in lms];h=cv2.convexHull(np.array([[x,y] for x,y,_ in pts],np.float32)).reshape(-1,2);out.append({'side':(cat.category_name or 'UNKNOWN').upper(),'confidence':float(cat.score),'landmarks':pts,'polygon':[[float(x),float(y)] for x,y in h]})
  return out
 def wheel(self,im):
  g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);cs,_=cv2.findContours(cv2.Canny(g,60,160),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);h,w=g.shape;best=None
  for c in cs:
   if len(c)<20:continue
   (x,y),(a,b),ang=cv2.fitEllipse(c);ratio=min(a,b)/max(a,b);area=np.pi*a*b/4/(w*h)
   if .04<area<.78 and ratio>.42 and (best is None or ratio>best[0]):best=(ratio,x/w,y/h,a/w,b/h,ang)
  if not best:return {'visible':False,'confidence':0,'polygon':[]}
  s,x,y,a,b,ang=best;p=cv2.ellipse2Poly((int(x*w),int(y*h)),(int(a*w/2),int(b*h/2)),int(ang),0,360,10);return {'visible':True,'confidence':float(s),'center':[x,y],'axes':[a,b],'angle':ang,'polygon':[[u/w,v/h] for u,v in p]}
 def features(self,hands,w):
  f={'wheel_confidence':w.get('confidence',0),'left_distance':2,'right_distance':2,'left_confidence':0,'right_confidence':0}
  if w.get('center'):
   cx,cy=w['center'];rx,ry=w['axes'][0]/2,w['axes'][1]/2
   for hand in hands:
    side=hand['side'].lower();d=min(abs(math.hypot((x-cx)/rx,(y-cy)/ry)-1) for x,y,_ in hand['landmarks']);f[side+'_distance']=d;f[side+'_confidence']=hand['confidence']
  return f
 def classify(self,f,hands,w):
  if not w.get('visible') or w['confidence']<self.c['perception']['wheel_confidence']:return 'UNKNOWN',0
  l=f['left_distance']<self.c['classifier']['contact_distance'];r=f['right_distance']<self.c['classifier']['contact_distance'];state='BOTH_ON' if l and r else 'LEFT_ON' if l else 'RIGHT_ON' if r else 'NONE_ON' if hands and min(f['left_distance'],f['right_distance'])>=self.c['classifier']['no_contact_distance'] else 'UNKNOWN';return state,min(1.,w['confidence']*max(f['left_confidence'],f['right_confidence'],.25))
 def render(self,im,r):
  o=im.copy();h,w=o.shape[:2]
  for q in r['hands']:
   p=np.array([[int(x*w),int(y*h)] for x,y in q['polygon']],np.int32);cv2.polylines(o,[p],True,(255,70,210),2)
  p=np.array([[int(x*w),int(y*h)] for x,y in r['wheel']['polygon']],np.int32)
  if len(p):cv2.polylines(o,[p],True,(50,210,255),2)
  return o
