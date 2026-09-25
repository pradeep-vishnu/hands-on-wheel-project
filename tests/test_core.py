from howvision.core.fs import safe,allocate
from howvision.core.db import Store
from howvision.core.models import read
def test_security():
 try:safe('../x');assert False
 except ValueError:pass
def test_allocate(tmp_path):assert allocate(tmp_path,'run')[0]=='run_0001' and allocate(tmp_path,'run')[0]=='run_0002'
def test_annotations(tmp_path):
 s=Store(tmp_path/'x.db');a={'run_id':'run_0001','frame_id':0,'label':'NONE_ON','original_label':'LEFT_ON','reward':1,'hands':[],'wheel':[],'flags':[]};s.save(a);assert s.all()[0]['label']=='NONE_ON'
def test_models(tmp_path):assert read(tmp_path/'r.json')['default']=='baseline'
