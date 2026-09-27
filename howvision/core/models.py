from pathlib import Path
import json
def registry(path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():p.write_text(json.dumps({'default':'baseline','checkpoints':[]},indent=2))
    return json.loads(p.read_text())
def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2))
def parameters(width):return 5*width+width+width*5+5
