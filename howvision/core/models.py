from pathlib import Path
from datetime import datetime,timezone
import json
def read(p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():p.write_text(json.dumps({'default':'baseline','checkpoints':[]},indent=2))
 return json.loads(p.read_text())
def write(p,o):Path(p).write_text(json.dumps(o,indent=2))
def register(p,file,metrics):
 r=read(p);tag=datetime.now(timezone.utc).strftime('how_%Y%m%d_%H%M%S_utc');r['checkpoints'].append({'id':tag,'path':str(file),'created_at':datetime.now(timezone.utc).isoformat(),'metrics':metrics});write(p,r);return tag
