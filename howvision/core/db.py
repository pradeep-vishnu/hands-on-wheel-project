import sqlite3,json,time
from pathlib import Path
SCHEMA='''CREATE TABLE IF NOT EXISTS annotations(id INTEGER PRIMARY KEY,run_id TEXT NOT NULL,frame_id INTEGER NOT NULL,label TEXT NOT NULL,original_label TEXT NOT NULL,reward REAL NOT NULL,hand_polygons TEXT NOT NULL,wheel_polygon TEXT NOT NULL,flags TEXT NOT NULL,annotator TEXT NOT NULL,created_at REAL NOT NULL,UNIQUE(run_id,frame_id)); CREATE TABLE IF NOT EXISTS jobs(job_id TEXT PRIMARY KEY,kind TEXT,status TEXT,payload TEXT,updated_at REAL);'''
class Store:
 def __init__(self,path): self.path=Path(path); self.init()
 def conn(self): c=sqlite3.connect(self.path); c.row_factory=sqlite3.Row; return c
 def init(self):
  with self.conn() as c:c.executescript(SCHEMA)
 def annotate(self,a):
  with self.conn() as c:c.execute('INSERT INTO annotations(run_id,frame_id,label,original_label,reward,hand_polygons,wheel_polygon,flags,annotator,created_at) VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(run_id,frame_id) DO UPDATE SET label=excluded.label,reward=excluded.reward,hand_polygons=excluded.hand_polygons,wheel_polygon=excluded.wheel_polygon,flags=excluded.flags,annotator=excluded.annotator,created_at=excluded.created_at',(a['run_id'],a['frame_id'],a['label'],a['original_label'],a['reward'],json.dumps(a.get('hand_polygons',[])),json.dumps(a.get('wheel_polygon',[])),json.dumps(a.get('flags',[])),a.get('annotator','local-user'),time.time()))
 def annotations(self,run_id=None):
  with self.conn() as c:
   rows=c.execute('SELECT * FROM annotations'+(' WHERE run_id=?' if run_id else ''),(run_id,) if run_id else ()).fetchall(); return [dict(r) for r in rows]
