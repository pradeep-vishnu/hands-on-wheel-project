import threading,webbrowser
from pathlib import Path
import uvicorn
from howvision.config import load
def main():
 c=load();Path(c['paths']['data']).mkdir(exist_ok=True);Path(c['paths']['output']).mkdir(exist_ok=True);url=f"http://{c['app']['host']}:{c['app']['port']}"
 if c['app'].get('open_browser'):threading.Timer(1.2,lambda:webbrowser.open(url,new=2)).start()
 uvicorn.run('howvision.api.app:create_app',factory=True,host=c['app']['host'],port=c['app']['port'])
if __name__=='__main__':main()
