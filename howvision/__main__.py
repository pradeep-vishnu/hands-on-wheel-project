from pathlib import Path
import uvicorn
from howvision.config import load
def main():
 c=load();[Path(x).mkdir(parents=True,exist_ok=True) for x in [c['paths']['data'],c['paths']['output'],'models/versions','datasets/versions']];print(f"Hands On Wheel Project READY http://{c['app']['host']}:{c['app']['port']}");uvicorn.run('howvision.api.app:create_app',factory=True,host=c['app']['host'],port=c['app']['port'])
if __name__=='__main__':main()
