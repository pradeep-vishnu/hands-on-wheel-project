import json
from howvision.core.fs import safe,allocate
from howvision.core.db import Store
def test_security():
 try:safe('../x');assert False
 except ValueError:pass
def test_allocate(tmp_path):assert allocate(tmp_path,'run')[0]=='run_0001' and allocate(tmp_path,'run')[0]=='run_0002'
def test_annotation(tmp_path):
 s=Store(tmp_path/'x.db');a={'run_id':'run_0001','frame_id':1,'label':'LEFT_ON','original_label':'UNKNOWN','reward':1,'hand_strokes':[],'wheel_strokes':[],'flags':[]};s.save(a);assert s.all()[0]['frame_id']==1
