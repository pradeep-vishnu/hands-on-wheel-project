from pathlib import Path
from collections import deque,Counter
import cv2,json,time,numpy as np
from howvision.core.fs import allocate,atomic,sha

def run(cfg,path,device,emit):
 rid,rd=allocate(Path(cfg['paths']['output']));(rd/'frames').mkdir();(rd/'overlays').mkdir();ext=path.suffix.lower();seq=[]
 if ext in cfg['input']['videos']:
  cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS) or 25
  while True:
   ok,im=cap.read()
   if not ok:break
   seq.append((im,len(seq),float(cap.get(cv2.CAP_PROP_POS_MSEC))))
  cap.release()
 else:seq=[(cv2.imread(str(path)),0,0.0)];fps=0
 started=time.time();records=[];writer=None;window=deque(maxlen=cfg['temporal']['smoothing_window'])
 with (rd/'predictions.jsonl').open('w') as f:
  for im,i,ts in seq:
   if im is None:continue
   wheel=detect_wheel(im);hands=detect_hands(im);state,conf=classify(hands,wheel,cfg);window.append(state if conf>=cfg['temporal']['confidence_threshold'] else 'UNKNOWN');stable=Counter(window).most_common(1)[0][0];status='HOW_ON' if stable in {'LEFT_ON','RIGHT_ON','BOTH_ON'} else 'HOW_OFF' if stable=='NONE_ON' else 'UNKNOWN';r={'run_id':rid,'source_file':path.name,'frame_id':i,'timestamp_ms':ts,'hands':hands,'wheel':wheel,'features':{'wheel_confidence':wheel.get('confidence',0),'left_distance':2,'right_distance':2,'left_confidence':0,'right_confidence':0},'raw_prediction':state,'how_state':stable,'how_status':status,'confidence':conf,'device':device.dict()};records.append(r);f.write(json.dumps(r)+'\n');cv2.imwrite(str(rd/'frames'/f'frame_{i:06d}.jpg'),im);ov=render(im,r);cv2.imwrite(str(rd/'overlays'/f'frame_{i:06d}.jpg'),ov)
   if ext in cfg['input']['videos']:
    if writer is None:writer=cv2.VideoWriter(str(rd/'output.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(im.shape[1],im.shape[0]))
    writer.write(ov)
   else:cv2.imwrite(str(rd/f'output{ext}'),ov)
   emit({'status':'RUNNING','run_id':rid,'frame_id':i,'processed':i+1,'total':len(seq),'progress':(i+1)/max(1,len(seq)),'how_status':status,'confidence':conf,'preview':f'/api/runs/{rid}/frames/{i}','device':device.dict()})
 if writer:writer.release()
 manifest={'run_id':rid,'status':'COMPLETED','created_at':started,'completed_at':time.time(),'input':path.name,'input_hash':sha(path),'frames':len(records),'requested_device':device.requested,'actual_device':device.actual,'device_name':device.name,'device_fallback_reason':device.fallback_reason};atomic(rd/'manifest.json',manifest);atomic(rd/'metrics.json',{'frames':len(records),'seconds':time.time()-started});return rid
def detect_hands(im):
 return []
def detect_wheel(im):
 g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);cs,_=cv2.findContours(cv2.Canny(g,60,160),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);h,w=g.shape;best=None
 for c in cs:
  if len(c)<20:continue
  (x,y),(a,b),ang=cv2.fitEllipse(c);ratio=min(a,b)/max(a,b);area=np.pi*a*b/4/(w*h)
  if .04<area<.78 and ratio>.42 and (best is None or ratio>best[0]):best=(ratio,x/w,y/h,a/w,b/h,ang)
 if not best:return {'visible':False,'confidence':0,'polygon':[]}
 s,x,y,a,b,ang=best;p=cv2.ellipse2Poly((int(x*w),int(y*h)),(int(a*w/2),int(b*h/2)),int(ang),0,360,10);return {'visible':True,'confidence':float(s),'center':[x,y],'axes':[a,b],'polygon':[[u/w,v/h] for u,v in p]}
def classify(hands,wheel,cfg):
 if not wheel['visible'] or wheel['confidence']<cfg['classifier']['wheel_confidence']:return 'UNKNOWN',0.0
 return ('NONE_ON',float(wheel['confidence'])) if hands else ('UNKNOWN',.2)
def render(im,r):
 o=im.copy();h,w=o.shape[:2];p=np.array([[int(x*w),int(y*h)] for x,y in r['wheel']['polygon']],np.int32)
 if len(p):cv2.polylines(o,[p],True,(60,210,255),2)
 return o
