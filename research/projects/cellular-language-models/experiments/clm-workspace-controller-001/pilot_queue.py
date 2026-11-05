"""Transactional central queue: one claim per model, LPT with dynamic workers."""
import sqlite3,json,time


def connect(path):
    db=sqlite3.connect(str(path),timeout=30);db.execute('PRAGMA journal_mode=WAL');return db

def initialize(path,tasks):
    with connect(path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS tasks (arm TEXT PRIMARY KEY, priority REAL, status TEXT, worker TEXT, started REAL, ended REAL, result TEXT)')
        for arm,priority in tasks:db.execute('INSERT OR IGNORE INTO tasks VALUES (?,?,?,NULL,NULL,NULL,NULL)',(arm,priority,'PENDING'))

def claim(path,worker):
    with connect(path) as db:
        db.execute('BEGIN IMMEDIATE');row=db.execute("SELECT arm FROM tasks WHERE status='PENDING' ORDER BY priority DESC,arm LIMIT 1").fetchone()
        if row is None:return None
        db.execute("UPDATE tasks SET status='RUNNING',worker=?,started=?,ended=NULL,result=NULL WHERE arm=?",(worker,time.monotonic(),row[0]));return row[0]

def finish(path,arm,status,result):
    with connect(path) as db:db.execute('UPDATE tasks SET status=?,ended=?,result=? WHERE arm=?',(status,time.monotonic(),json.dumps(result),arm))

def rows(path):
    with connect(path) as db:
        return [{'arm':r[0],'priority':r[1],'status':r[2],'worker':r[3],'started':r[4],'ended':r[5],'result':json.loads(r[6]) if r[6] else None} for r in db.execute('SELECT * FROM tasks ORDER BY arm')]

def predict(rates,targets=2000000,overhead=None):
    loads=[0.,0.];assignments=[];times={a:(targets.get(a,0) if isinstance(targets,dict) else targets)/r+((overhead or {}).get(a,0) if (targets.get(a,0) if isinstance(targets,dict) else targets)>0 else 0) for a,r in rates.items()}
    for arm in sorted(times,key=times.get,reverse=True):
        worker=min(range(2),key=lambda i:loads[i]);assignments.append({'arm':arm,'worker':worker,'start':loads[worker],'seconds':times[arm]});loads[worker]+=times[arm]
    return {'per_model_seconds':times,'ideal_lower_bound_seconds':max(max(times.values()),sum(times.values())/2),'lpt_seconds':max(loads),'schedule':assignments}
