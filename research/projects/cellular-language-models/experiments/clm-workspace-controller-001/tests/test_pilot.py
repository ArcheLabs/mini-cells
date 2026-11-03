"""Two small CPU checks for scheduling/accounting; no pretend dual-GPU run."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concurrent.futures import ThreadPoolExecutor
import pytest
from pilot_queue import initialize,claim,finish,rows,predict
from pilot import inventory
from main_runner import plan
import torch


def test_transactional_lpt_queue_claims_each_arm_once(tmp_path):
    path=tmp_path/'queue.sqlite';arms=['C0','C1','C2','C3','T0-P','T0-F'];initialize(path,[(a,i) for i,a in enumerate(arms)])
    def consume(worker):
        result=[]
        while True:
            arm=claim(path,worker)
            if arm is None:return result
            result.append(arm);finish(path,arm,'COMPLETE',{'targets':2000000})
    with ThreadPoolExecutor(2) as pool:assigned=list(pool.map(consume,['worker-0','worker-1']))
    assert sorted(assigned[0]+assigned[1])==sorted(arms) and len(set(assigned[0]+assigned[1]))==6
    assert all(row['status']=='COMPLETE' for row in rows(path))
    prediction=predict({a:(1 if a in ('C2','C3') else 2) for a in arms},targets=10)
    assert prediction['lpt_seconds']==20 and prediction['ideal_lower_bound_seconds']==20


def test_exact_2m_budget_and_no_single_gpu_fallback(monkeypatch):
    documents=[{'doc_id':str(i),'ids':[2]+[4]*size+[3]} for i,size in enumerate((0,1,3,7,15,31,63,127,255))]
    p={'groups':{'documents':8}};groups,n=plan(documents,p,2000000)
    assert n==2000000 and sum(len(d['ids'])-1 for group in groups for d in group)==2000000
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    with pytest.raises(RuntimeError,match='DUAL_GPU_BLOCKED'):inventory()
