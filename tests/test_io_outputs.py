import csv,json,cv2,numpy as np
from howvision.inference.io import frames
from howvision.inference.engine import CSV_COLUMNS
from howvision.core.fs import allocate
def test_image_identity(tmp_path):
 p=tmp_path/'x.png';cv2.imwrite(str(p),np.zeros((20,30,3),np.uint8));items=list(frames(p));assert len(items)==1;_,i=items[0];assert (i.frame_id,i.sequence_index,i.timestamp_ms,i.width,i.height)==(0,0,0,30,20)
def test_video_order_dimensions_and_native_timestamps(tmp_path):
 p=tmp_path/'x.avi';w=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'MJPG'),10,(32,24));[w.write(np.full((24,32,3),i*30,np.uint8)) for i in range(3)];w.release();ids=[i for _,i in frames(p)];assert [x.frame_id for x in ids]==[0,1,2];assert all((x.width,x.height)==(32,24) for x in ids);assert ids[1].native_timestamp['kind']=='opencv_pos_msec'
def test_atomic_run_allocation(tmp_path):assert allocate(tmp_path,'run')[0]=='run_0001' and allocate(tmp_path,'run')[0]=='run_0002'
def test_csv_schema():assert CSV_COLUMNS==['run_id','source_file','frame_id','sequence_index','timestamp_ms','raw_prediction','raw_confidence','temporal_prediction','temporal_confidence','how_status','checkpoint']
