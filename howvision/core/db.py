import sqlite3,json,time
class Store:
 def __init__(self,p):self.p=p;self.init()
 def db(self):c=sqlite3.connect(self.p);c.row_factory=sqlite3.Row;return c
 def init(self):
  with self.db() as c:
   c.execute('CREATE TABLE IF NOT EXISTS annotations(run_id TEXT,frame_id INTEGER,label TEXT,original_label TEXT,reward REAL,hand_strokes TEXT DEFAULT "[]",wheel_strokes TEXT DEFAULT "[]",flags TEXT DEFAULT "[]",created REAL,PRIMARY KEY(run_id,frame_id))')
   cols={x[1] for x in c.execute('PRAGMA table_info(annotations)')}
   for name in ('hand_strokes','wheel_strokes','flags'):
    if name not in cols:c.execute(f'ALTER TABLE annotations ADD COLUMN {name} TEXT DEFAULT "[]"')
 def save(self,a):
  with self.db() as c:c.execute('INSERT OR REPLACE INTO annotations(run_id,frame_id,label,original_label,reward,hand_strokes,wheel_strokes,flags,created) VALUES(?,?,?,?,?,?,?,?,?)',(a['run_id'],a['frame_id'],a['label'],a['original_label'],a['reward'],json.dumps(a.get('hand_strokes',[])),json.dumps(a.get('wheel_strokes',[])),json.dumps(a.get('flags',[])),time.time()))
 def all(self,run=None):
  with self.db() as c:
   rows=c.execute('SELECT * FROM annotations'+(' WHERE run_id=?' if run else ''),(run,) if run else ()).fetchall();return [dict(x) for x in rows]
