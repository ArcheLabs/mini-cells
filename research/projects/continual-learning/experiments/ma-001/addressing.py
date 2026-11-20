"""Transactional mock Cell registry and address index."""
import hashlib
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from math import log


class Registry:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS cells(id TEXT PRIMARY KEY, category TEXT, description TEXT,
          session TEXT, version INTEGER, digest TEXT, active INTEGER, created TEXT, updated TEXT);
        CREATE TABLE IF NOT EXISTS addresses(event TEXT, cell TEXT REFERENCES cells(id),
          version INTEGER, descriptor TEXT, raw TEXT, PRIMARY KEY(event,cell));
        CREATE TABLE IF NOT EXISTS inbox(event TEXT PRIMARY KEY, raw TEXT);
        """)

    def create(self, cell, category, description, session):
        now = datetime.now(timezone.utc).isoformat()
        with self.db:
            self.db.execute("INSERT INTO cells VALUES(?,?,?,?,?,?,?,?,?)",
                            (cell, category, description, session, 0, "", 1, now, now))

    def write(self, event, raw, descriptor, explicit=(), binding=(), inject=False):
        targets = explicit or binding
        with self.db:
            if not targets:
                self.db.execute("INSERT INTO inbox VALUES(?,?)", (event, raw))
                return []
            cells = set()
            for target in targets:
                hits = self.db.execute(
                    "SELECT id FROM cells WHERE (id=? OR category=?) AND active=1",
                    (target, target)).fetchall()
                if not hits:
                    raise KeyError(target)
                cells.update(row[0] for row in hits)
            for cell in sorted(cells):
                version, old = self.db.execute("SELECT version,digest FROM cells WHERE id=?", (cell,)).fetchone()
                version += 1
                digest = hashlib.sha256(json.dumps([old,event,raw]).encode()).hexdigest()
                self.db.execute("UPDATE cells SET version=?,digest=?,updated=? WHERE id=?",
                    (version,digest,datetime.now(timezone.utc).isoformat(),cell))
                self.db.execute("INSERT INTO addresses VALUES(?,?,?,?,?)",
                    (event,cell,version,json.dumps(descriptor,ensure_ascii=False),raw))
                if inject:
                    raise RuntimeError("injected failure")
            return sorted(cells)

    def tombstone(self, cell):
        with self.db:
            self.db.execute("UPDATE cells SET active=0, updated=? WHERE id=?",
                            (datetime.now(timezone.utc).isoformat(),cell))

    def records(self):
        rows = self.db.execute("""SELECT event,cell,addresses.version,descriptor,raw,category
          FROM addresses JOIN cells ON cells.id=addresses.cell WHERE cells.active=1
          ORDER BY event,cell""")
        return [dict(address_id=json.dumps([e,c]),canonical_event_id=e,cell_id=c,
                     cell_version_at_write=v,descriptor_schema_version=1,category_id=cat,
                     active=True,raw=raw,**json.loads(d)) for e,c,v,d,raw,cat in rows]


def tokens(text):
    return re.findall(r"[a-z0-9]+|[\u3400-\u9fff]",text.casefold())


class Ranker:
    def __init__(self, records, raw=False):
        self.rows = records
        self.raw = raw
        self.text = [r["raw"] if raw else " ".join([r["summary"],*r["entities"],*r["topics"],*r["aliases"]])
                     for r in records]
        self.docs = [Counter(tokens(x)) for x in self.text]
        self.df = Counter(t for doc in self.docs for t in doc)
        self.avg = sum(sum(doc.values()) for doc in self.docs)/max(1,len(records))

    def scores(self, query, arm, dense=None):
        terms = set(tokens(query))
        lexical = [
            sum(log(1+(len(self.docs)-self.df[t]+.5)/(self.df[t]+.5))*doc[t]*2.2/
                (doc[t]+1.2*(.25+.75*sum(doc.values())/max(self.avg,1))) for t in terms if doc[t])
            for doc in self.docs]
        phrase = ' '+' '.join(tokens(query))+' '
        entity = [sum((' '+' '.join(tokens(alias))+' ') in phrase
                      for alias in set(r['entities']+r['aliases'])
                      if tokens(alias) and (not self.raw or
                          ' '+' '.join(tokens(alias))+' ' in ' '+' '.join(tokens(r['raw']))+' '))
                  for r in self.rows]
        if arm in ('E0','E1','E2'):
            scores = {'E0':lexical,'E1':dense,'E2':entity}[arm]
            if scores is None:
                raise RuntimeError('Dense embeddings absent')
        else:
            if dense is None:
                raise RuntimeError('Hybrid requires dense embeddings')
            scores = [0.]*len(self.rows)
            for channel in (lexical,dense,entity):
                order = sorted((i for i,s in enumerate(channel) if s>0),key=lambda i:(-channel[i],i))
                for rank,i in enumerate(order,1):
                    scores[i] += 1/(60+rank)
        grouped = {}
        for row,score in zip(self.rows,scores):
            cell = row['cell_id']
            grouped[cell] = max(grouped.get(cell,float('-inf')),float(score))
        return sorted(grouped.items(),key=lambda x:(-x[1],x[0]))

    @staticmethod
    def route(scores, threshold):
        return [cell for cell,score in scores[:3] if score>threshold]
