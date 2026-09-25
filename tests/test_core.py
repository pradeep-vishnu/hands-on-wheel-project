from howvision.core.db import Store
from howvision.core.fs import allocate,safe_name
from howvision.pipeline.inference import status
def test_status():assert status('LEFT_ON')=='HOW_ON' and status('NONE_ON')=='HOW_OFF' and status('UNKNOWN')=='UNKNOWN'
def test_alloc(tmp_path):assert allocate(tmp_path,'run')[0]=='run_0001' and allocate(tmp_path,'run')[0]=='run_0002'
def test_security():
 try:safe_name('../x');assert False
 except ValueError:pass
def test_annotations(tmp_path):
 s=Store(tmp_path/'x.db');a={'run_id':'run_0001','frame_id':0,'label':'NONE_ON','original_label':'LEFT_ON','reward':1,'hand_polygons':[],'wheel_polygon':[],'flags':[]};s.annotate(a);assert s.annotations()[0]['label']=='NONE_ON';s.annotate({**a,'label':'LEFT_ON'});assert len(s.annotations())==1
