from pathlib import Path
import yaml
def load(path="configs/default.yaml"):
 c=yaml.safe_load(Path(path).read_text()); return c
