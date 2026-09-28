from dataclasses import dataclass,asdict
@dataclass
class DeviceChoice:
 requested:str;actual:str;fallback_reason:str|None;name:str
 def dict(self):return asdict(self)
def _torch():
 try:
  import torch
  return torch
 except Exception:return None
def discover():
 torch=_torch()
 if torch is None:return [{'id':'cpu','name':'CPU','available':True,'details':'PyTorch unavailable; OpenCV/MediaPipe CPU path'},{'id':'cuda','name':'CUDA','available':False,'details':'PyTorch not installed'},{'id':'mps','name':'Apple MPS','available':False,'details':'PyTorch not installed'}]
 capability=getattr(torch.backends.cpu,'get_cpu_capability',lambda:'DEFAULT')();values=[{'id':'cpu','name':'CPU','available':True,'details':capability}]
 if torch.cuda.is_available():values.append({'id':'cuda','name':torch.cuda.get_device_name(0),'available':True,'details':f'CUDA {torch.version.cuda}'})
 else:values.append({'id':'cuda','name':'CUDA','available':False,'details':'No compatible CUDA runtime/device'})
 mps=getattr(torch.backends,'mps',None);avail=bool(mps and mps.is_built() and mps.is_available());values.append({'id':'mps','name':'Apple MPS','available':avail,'details':'Metal Performance Shaders' if avail else 'MPS unavailable'})
 return values
def resolve(requested:str):
 torch=_torch();available={x['id']:x for x in discover() if x['available']};selected=(requested or 'auto').lower()
 if selected=='auto':selected='cuda' if 'cuda' in available else 'mps' if 'mps' in available else 'cpu'
 if selected not in {'cpu','cuda','mps'}:return DeviceChoice(requested,'cpu','Unknown device request','CPU')
 if selected not in available:return DeviceChoice(requested,'cpu',f'{selected.upper()} unavailable','CPU')
 if selected=='cpu':return DeviceChoice(requested,'cpu',None,available['cpu']['name'])
 if torch is None:return DeviceChoice(requested,'cpu','PyTorch unavailable','CPU')
 try:
  d=torch.device(selected);x=torch.ones(1,device=d);_=(x+1).cpu();return DeviceChoice(requested,selected,None,available[selected]['name'])
 except Exception as exc:return DeviceChoice(requested,'cpu',f'{selected.upper()} initialization failed: {exc}','CPU')
