from pathlib import Path
import yaml
def load(path='configs/default.yaml'):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))
