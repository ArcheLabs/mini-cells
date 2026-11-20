import tempfile
import unittest
from pathlib import Path
from addressing import Registry,Ranker

D = dict(summary="parallel VM",entities=["GPU-PVM"],topics=["parallel computation"],aliases=["GPU PVM"])

class AddressingTest(unittest.TestCase):
    def test_write_priority_reopen_and_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"registry.db"
            reg=Registry(path)
            reg.create("a","project/a","VM","s1")
            reg.create("b","domain/b","GPU","s2")
            self.assertEqual(reg.write("inbox","none",D),[])
            self.assertEqual(reg.write("explicit","body",D,["project/a"],["domain/b"]),["a"])
            self.assertEqual(reg.write("bound","body",D,binding=["domain/b"]),["b"])
            self.assertEqual(reg.write("multi","body",D,["project/a","domain/b"]),["a","b"])
            rows=reg.records()
            cells=reg.db.execute("SELECT * FROM cells").fetchall()
            with self.assertRaisesRegex(RuntimeError,"injected"):
                reg.write("failure","body",D,["a","b"],inject=True)
            self.assertEqual(reg.db.execute("SELECT * FROM cells").fetchall(),cells)
            reg.db.close()
            reg=Registry(path)
            self.assertEqual(reg.records(),rows)
            rank=Ranker(rows)
            self.assertEqual(rank.route(rank.scores("GPU PVM","E2"),0),["a","b"])
            self.assertEqual(rank.route(rank.scores("banana","E0"),0),[])
            reg.tombstone("a")
            self.assertEqual({r["cell_id"] for r in reg.records()},{"b"})
            reg.db.close()
