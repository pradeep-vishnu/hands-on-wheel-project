import sqlite3,json,time
class Store:
    def __init__(self,path):self.path=path;self.init()
    def db(self):c=sqlite3.connect(self.path);c.row_factory=sqlite3.Row;return c
    def init(self):
        with self.db() as c:c.execute('CREATE TABLE IF NOT EXISTS annotations(run_id TEXT,frame_id INTEGER,label TEXT,original_label TEXT,sample_weight REAL,hand_strokes TEXT,wheel_strokes TEXT,created REAL,PRIMARY KEY(run_id,frame_id))')
    def save(self,a):
        with self.db() as c:c.execute('INSERT OR REPLACE INTO annotations VALUES(?,?,?,?,?,?,?,?)',(a['run_id'],a['frame_id'],a['label'],a['original_label'],a['sample_weight'],json.dumps(a.get('hand_strokes',[])),json.dumps(a.get('wheel_strokes',[])),time.time()))
    def all(self,run=None):
        with self.db() as c:return [dict(x) for x in c.execute('SELECT * FROM annotations'+(' WHERE run_id=?' if run else ''),(run,) if run else ())]
