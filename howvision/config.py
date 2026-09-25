from pathlib import Path
import yaml
def load_config(path: str|Path="configs/default.yaml"):
    p=Path(path); cfg=yaml.safe_load(p.read_text(encoding="utf-8")); cfg["_config_path"]=str(p); return cfg
