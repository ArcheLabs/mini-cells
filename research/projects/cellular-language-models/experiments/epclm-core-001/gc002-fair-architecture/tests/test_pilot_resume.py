"""One tiny synthetic CPU contract for the new checkpoint boundary, no real Pilot."""
import torch
from cached_transformer import Config
from pilot_training import PilotEngine,verify_resume


def test_pilot_committed_update_and_exact_resume(tmp_path):
    torch.set_num_threads(1)
    docs=[{'id':'a','tokens':[1,4,5,3]},{'id':'b','tokens':[1,6,7,8,3]}]
    config=Config(vocab=32,dim=16,layers=1,heads=2,kv_heads=1,ffn=32,window=4)
    engine=PilotEngine(docs,config,{'source':'synthetic','protocol':'test'},'cpu',713,2,2,steps=4)
    before=engine.model.weight_identity()
    row=engine.update();assert row['update']==1 and row['tokens']==4
    assert engine.model.weight_identity()!=before
    restored=verify_resume(engine,tmp_path/'pilot.pt')
    assert restored.resume_result['status']=='PASS'
    row=restored.update();assert row['update']==2 and row['tokens']==8
    restored.save(tmp_path/'pilot.pt')
    loaded=PilotEngine.load(tmp_path/'pilot.pt',docs,config,engine.identity,'cpu',713,2,2,steps=4)
    assert loaded.ledger.report()==restored.ledger.report()
    assert loaded.model.weight_identity()==restored.model.weight_identity()
