from dataclasses import dataclass,asdict
@dataclass
class Choice:
 requested:str
 actual:str
 name:str
 fallback_reason:str|None
 def dict(self):return asdict(self)
def get_torch():
 try:
  import torch
  return torch
 except Exception:return None
def discover():
 t=get_torch();items=[{'id':'cpu','name':'CPU','available':True,'details':'Universal fallback'}]
 if t is None:return items+[{'id':'cuda','name':'CUDA','available':False,'details':'PyTorch unavailable'},{'id':'mps','name':'Apple MPS','available':False,'details':'PyTorch unavailable'}]
 items.append({'id':'cuda','name':t.cuda.get_device_name(0) if t.cuda.is_available() else 'CUDA','available':bool(t.cuda.is_available()),'details':f'CUDA {t.version.cuda}' if t.cuda.is_available() else 'Unavailable'})
 m=getattr(t.backends,'mps',None);ok=bool(m and m.is_built() and m.is_available());items.append({'id':'mps','name':'Apple MPS','available':ok,'details':'Metal Performance Shaders' if ok else 'Unavailable'})
 return items
def resolve(requested):
 request=(requested or 'auto').lower();available={x['id']:x for x in discover() if x['available']};actual='cuda' if request=='auto' and 'cuda' in available else 'mps' if request=='auto' and 'mps' in available else 'cpu' if request=='auto' else request
 if actual not in available:return Choice(requested,'cpu','CPU',f'{actual.upper()} unavailable')
 if actual=='cpu':return Choice(requested,'cpu','CPU',None)
 t=get_torch()
 try:
  d=t.device(actual);x=t.ones(1,device=d);_=(x+1).cpu();return Choice(requested,actual,available[actual]['name'],None)
 except Exception as e:return Choice(requested,'cpu','CPU',f'{actual.upper()} failed: {e}')
