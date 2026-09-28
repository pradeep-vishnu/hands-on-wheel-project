from pathlib import Path
import yaml
def load():
 return yaml.safe_load(Path('configs/default.yaml').read_text())
