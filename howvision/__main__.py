import threading,webbrowser
from pathlib import Path
import uvicorn
from howvision.config import load
def main():
 c=load();[Path(x).mkdir(parents=True,exist_ok=True) for x in [c['paths']['data'],c['paths']['output'],'models/versions','datasets/versions']];url=f"http://{c['app']['host']}:{c['app']['port']}"
 if c['app'].get('open_browser'):threading.Timer(1.2,lambda:webbrowser.open(url,new=2)).start()
 print('Hands On Wheel Project v0.7.0',url);uvicorn.run('howvision.api.app:create_app',factory=True,host=c['app']['host'],port=c['app']['port'])
if __name__=='__main__':main()
