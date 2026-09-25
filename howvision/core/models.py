from pathlib import Path
from datetime import datetime,timezone
import json
def read(path):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():p.write_text(json.dumps({'default':'baseline','checkpoints':[]},indent=2))
 return json.loads(p.read_text())
def save(path,r):Path(path).write_text(json.dumps(r,indent=2))
def register(path,file,metrics):
 r=read(path);tag=datetime.now(timezone.utc).strftime('how_%Y%m%d_%H%M%S_utc');r['checkpoints'].append({'id':tag,'path':str(file),'created_at':datetime.now(timezone.utc).isoformat(),'metrics':metrics});save(path,r);return tag
