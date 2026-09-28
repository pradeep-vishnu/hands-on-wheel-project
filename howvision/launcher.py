import os,socket,threading,webbrowser
import uvicorn
def free_port():
 s=socket.socket();s.bind(('127.0.0.1',0));p=s.getsockname()[1];s.close();return p
def main():
 p=free_port();url=f'http://127.0.0.1:{p}/?session={os.urandom(5).hex()}';print(url,flush=True);threading.Timer(1,lambda:webbrowser.open(url,new=2)).start();uvicorn.run('howvision.api.app:create_app',factory=True,host='127.0.0.1',port=p)
if __name__=='__main__':main()
