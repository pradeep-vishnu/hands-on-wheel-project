from pathlib import Path
import cv2,json,csv,time,math,numpy as np
from howvision.core.fs import allocate,sha,atomic_json,IMAGES,VIDEOS
LABELS=['LEFT_ON','RIGHT_ON','BOTH_ON','NONE_ON','UNKNOWN']
def status(x): return 'HOW_ON' if x in LABELS[:3] else 'HOW_OFF' if x=='NONE_ON' else 'UNKNOWN'
class Engine:
 def __init__(self,cfg):
  self.cfg=cfg
  import mediapipe as mp
  from mediapipe.tasks import python
  from mediapipe.tasks.python import vision
  p=Path(cfg['paths']['hand_model']);
  if not p.exists(): raise RuntimeError('hand model missing; run scripts/check_install.py')
  opts=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(p)),running_mode=vision.RunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=cfg['perception']['hand_confidence']); self.mp=mp; self.detector=vision.HandLandmarker.create_from_options(opts)
 def hands(self,img):
  x=self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); r=self.detector.detect(x); out=[]
  for i,lms in enumerate(r.hand_landmarks):
   c=r.handedness[i][0]; pts=[(float(p.x),float(p.y)) for p in lms]; hull=cv2.convexHull(np.array(pts,np.float32)).reshape(-1,2).tolist(); out.append({'side':(c.category_name or 'UNKNOWN').upper(),'confidence':float(c.score),'landmarks':pts,'polygon':hull})
  return out
 def wheel(self,img):
  g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); cs,_=cv2.findContours(cv2.Canny(g,60,160),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE); h,w=g.shape; best=None
  for c in cs:
   if len(c)<20:continue
   (cx,cy),(a,b),ang=cv2.fitEllipse(c); ar=np.pi*a*b/4/(w*h); ratio=min(a,b)/max(a,b)
   if .04<=ar<=.78 and ratio>.42:
    score=ratio*(1-abs(ar-.28))
    if best is None or score>best[0]:best=(score,cx/w,cy/h,a/w,b/h,ang)
  if not best:return {'visible':False,'confidence':0,'polygon':[]}
  sc,cx,cy,a,b,ang=best; poly=cv2.ellipse2Poly((int(cx*w),int(cy*h)),(int(a*w/2),int(b*h/2)),int(ang),0,360,10); return {'visible':sc>=.25,'confidence':float(sc),'center':[cx,cy],'axes':[a,b],'angle':ang,'polygon':[[float(x/w),float(y/h)] for x,y in poly]}
 def classify(self,hands,wheel):
  if not wheel.get('visible'): return 'UNKNOWN',0,{}
  cx,cy=wheel['center']; rx,ry=wheel['axes'][0]/2,wheel['axes'][1]/2; f={}; on=[]
  for h in hands:
   d=min(abs(math.sqrt(((x-cx)/rx)**2+((y-cy)/ry)**2)-1) for x,y in h['landmarks']); f[h['side'].lower()+'_distance']=d
   if d<=self.cfg['classifier']['contact_distance']:on.append(h['side'])
  label='BOTH_ON' if {'LEFT','RIGHT'}<=set(on) else 'LEFT_ON' if 'LEFT' in on else 'RIGHT_ON' if 'RIGHT' in on else 'NONE_ON' if hands else 'UNKNOWN'; conf=min(1.,wheel['confidence']*(sum(h['confidence'] for h in hands)/len(hands))) if hands else .2; return label,conf,f
 def run(self,source,progress=lambda x:None):
  rid,rd=allocate(self.cfg['paths']['output'],'run'); (rd/'overlays').mkdir(); started=time.time(); ext=source.suffix.lower(); cap=None
  if ext in IMAGES:
   im=cv2.imread(str(source)); seq=[(im,0,0.0)]
  elif ext in VIDEOS:
   cap=cv2.VideoCapture(str(source)); seq=[]; i=0
   while True:
    ok,im=cap.read()
    if not ok:break
    seq.append((im,i,float(cap.get(cv2.CAP_PROP_POS_MSEC)))); i+=1
   cap.release()
  else: raise ValueError('unsupported input')
  rows=[]; jf=(rd/'predictions.jsonl').open('w'); writer=None
  for im,i,ts in seq:
   hands=self.hands(im); wheel=self.wheel(im); label,conf,features=self.classify(hands,wheel); rec={'run_id':rid,'source_file':source.name,'frame_id':i,'sequence_index':i,'timestamp_ms':ts,'timestamp_native':{'kind':'opencv_pos_msec' if ext in VIDEOS else 'image','value':ts},'hands':hands,'wheel':wheel,'features':features,'how_state':label,'how_status':status(label),'confidence':conf,'model_version':'mediapipe-f16-rule-v1'}; jf.write(json.dumps(rec)+'\n'); rows.append(rec); ov=render(im,rec); cv2.imwrite(str(rd/'overlays'/f'frame_{i:06d}.jpg'),ov)
   if ext in VIDEOS:
    if writer is None: writer=cv2.VideoWriter(str(rd/'output.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(im.shape[1],im.shape[0]))
    writer.write(ov)
   else: cv2.imwrite(str(rd/f'output{ext}'),ov)
   progress({'run_id':rid,'frame':i,'total':len(seq),'progress':(i+1)/max(1,len(seq)),'status':'RUNNING','preview':f'/api/runs/{rid}/frames/{i}','label':rec['how_status'],'confidence':conf})
  if writer:writer.release()
  jf.close(); cols=['run_id','source_file','frame_id','sequence_index','timestamp_ms','how_state','how_status','confidence','model_version']
  with (rd/'predictions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:r[k] for k in cols} for r in rows)
  atomic_json(rd/'manifest.json',{'run_id':rid,'status':'COMPLETED','input':source.name,'input_hash':sha(source),'model_version':'mediapipe-f16-rule-v1','frames':len(rows),'created_at':started,'completed_at':time.time()}); atomic_json(rd/'metrics.json',{'frames':len(rows),'seconds':time.time()-started}); return rid
def render(im,r):
 o=im.copy(); h,w=o.shape[:2]; col=(66,227,139) if r['how_status']=='HOW_ON' else (103,92,255) if r['how_status']=='HOW_OFF' else (75,184,229)
 for x in r['hands']:
  p=np.array([[int(a*w),int(b*h)] for a,b in x['polygon']],np.int32); cv2.polylines(o,[p],True,(255,120,220),2)
 p=np.array([[int(a*w),int(b*h)] for a,b in r['wheel']['polygon']],np.int32)
 if len(p):cv2.polylines(o,[p],True,(230,184,75),2)
 cv2.rectangle(o,(8,8),(330,72),(10,12,15),-1);cv2.putText(o,f"{r['how_status']} {r['confidence']:.0%}",(16,37),0,.7,col,2);cv2.putText(o,f"frame {r['frame_id']} {r['timestamp_ms']/1000:.3f}s",(16,61),0,.45,(230,230,230),1);return o
