from pathlib import Path
import uvicorn
from howvision.config import load_config
def main():
    cfg=load_config(); [Path(p).mkdir(parents=True,exist_ok=True) for p in [cfg["paths"]["data"],cfg["paths"]["output"],"datasets/reviewed","datasets/versions","datasets/exports","models/versions"]]; print(f"HOW Vision READY at http://{cfg['app']['host']}:{cfg['app']['port']}"); uvicorn.run("howvision.api.app:create_app",factory=True,host=cfg["app"]["host"],port=cfg["app"]["port"])
if __name__=="__main__": main()
