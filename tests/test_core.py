import json
from howvision.core.fs import safe,allocate
from howvision.core.db import Store
from howvision.core.models import read
def test_security():
 try:safe('../x');assert False
 except ValueError:pass
def test_allocate(tmp_path):assert allocate(tmp_path,'run')[0]=='run_0001' and allocate(tmp_path,'run')[0]=='run_0002'
def test_annotations(tmp_path):
 s=Store(tmp_path/'x.db');a={'run_id':'run_0001','frame_id':0,'label':'NONE_ON','original_label':'LEFT_ON','reward':1,'hand_strokes':[{'tool':'hand','points':[[.1,.1]]}],'wheel_strokes':[],'flags':[]};s.save(a);assert json.loads(s.all()[0]['hand_strokes'])[0]['tool']=='hand'
def test_models(tmp_path):assert read(tmp_path/'r.json')['default']=='baseline'
