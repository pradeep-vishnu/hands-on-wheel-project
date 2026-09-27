from pathlib import Path
from datetime import datetime,timezone
import csv,json,platform,sys,time,cv2,psutil
from howvision.core.fs import allocate,atomic_json,sha256,VIDEOS
from howvision.core.models import registry,parameters
from .io import frames
from .adapters import MediaPipeHandDetector,EllipseWheelDetector,RuleClassifier,TorchClassifier,features
from .temporal import TemporalDecision
from .render import OverlayRenderer
from .types import FrameResult
CSV_COLUMNS=['run_id','source_file','frame_id','sequence_index','timestamp_ms','raw_prediction','raw_confidence','temporal_prediction','temporal_confidence','how_status','checkpoint']
class Engine:
    def __init__(self,cfg,checkpoint='baseline',hand_detector=None,wheel_detector=None,classifier=None,renderer=None):
        self.cfg=cfg;self.checkpoint=checkpoint;self.hand=hand_detector or MediaPipeHandDetector(cfg['paths']['hand_model'],cfg['perception']['hand_confidence']);self.wheel=wheel_detector or EllipseWheelDetector();self.renderer=renderer or OverlayRenderer();self.temporal=TemporalDecision(cfg['temporal']);self.width=24
        if classifier:self.classifier=classifier
        elif checkpoint=='baseline':self.classifier=RuleClassifier(cfg['classifier']['contact_distance'])
        else:
            meta=next(x for x in registry(cfg['paths']['registry'])['checkpoints'] if x['id']==checkpoint);self.classifier=TorchClassifier(meta['path']);self.width=meta['hyperparameters']['hidden_width']
    def run(self,path,emit=lambda x:None):
        rid,rd=allocate(self.cfg['paths']['output'],'run');[(rd/x).mkdir() for x in ('frames','overlays','logs')];started=time.time();processed=failed=skipped=0;rows=[];results=[];writer=None;error=None
        atomic_json(rd/'resolved_config.json',self.cfg)
        try:
            for frame,ident in frames(path):
                try:
                    hs=self.hand.predict(frame);wh=self.wheel.predict(frame);feat=features(hs,wh);raw=self.classifier.predict(feat,hs,wh);temp=self.temporal.process(raw);status='HOW_ON' if temp.state in {'LEFT_ON','RIGHT_ON','BOTH_ON'} else 'HOW_OFF' if temp.state=='NONE_ON' else 'UNKNOWN';res=FrameResult(ident,hs,wh,feat,raw,temp,status);results.append(res);processed+=1;cv2.imwrite(str(rd/'frames'/f'frame_{ident.frame_id:06d}.jpg'),frame);ov=self.renderer.render(frame,res);cv2.imwrite(str(rd/'overlays'/f'frame_{ident.frame_id:06d}.jpg'),ov)
                    if path.suffix.lower() in VIDEOS:
                        if writer is None:writer=cv2.VideoWriter(str(rd/'output.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(ident.width,ident.height))
                        writer.write(ov)
                    else:cv2.imwrite(str(rd/f'output{path.suffix.lower()}'),ov)
                    row={'run_id':rid,'source_file':ident.source_file,'frame_id':ident.frame_id,'sequence_index':ident.sequence_index,'timestamp_ms':ident.timestamp_ms,'raw_prediction':raw.state,'raw_confidence':raw.confidence,'temporal_prediction':temp.state,'temporal_confidence':temp.confidence,'how_status':status,'checkpoint':self.checkpoint};rows.append(row);emit({'run_id':rid,'frame':processed,'total':0,'progress':0,'label':status,'confidence':temp.confidence,'preview':f'/api/runs/{rid}/frames/{ident.frame_id}?view=overlay'})
                except Exception as e:failed+=1;(rd/'logs'/'errors.log').open('a').write(f'{ident.frame_id}: {e}\n')
        except Exception as e:error=str(e)
        finally:
            if writer:writer.release()
        with (rd/'predictions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=CSV_COLUMNS);w.writeheader();w.writerows(rows)
        with (rd/'predictions.jsonl').open('w') as f:
            for r in results:f.write(json.dumps(r.json())+'\n')
        elapsed=time.time()-started;atomic_json(rd/'metrics.json',{'processed':processed,'failed':failed,'skipped':skipped,'seconds':elapsed,'average_fps':processed/max(elapsed,.001)});status='FAILED' if error else 'COMPLETED';manifest={'run_id':rid,'status':status,'created_at':datetime.now(timezone.utc).isoformat(),'input':path.name,'input_hash':sha256(path),'hand_model_hash':sha256(self.cfg['paths']['hand_model']),'checkpoint':self.checkpoint,'policy_parameters':parameters(self.width),'application_version':self.cfg['app']['version'],'python_version':sys.version,'platform':platform.platform(),'cpu_count':psutil.cpu_count(),'cuda':None,'processed':processed,'failed':failed,'skipped':skipped,'codec':'mp4v' if path.suffix.lower() in VIDEOS else None,'error':error};atomic_json(rd/'manifest.json',manifest)
        if error:raise RuntimeError(error)
        return rid
