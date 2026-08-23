import torch
from minicells.config import resolved_config
from minicells.data import fixed_dataset
from minicells.evaluate import evaluate
from minicells.model import EchoModel
from minicells.train import load_checkpoint,save_checkpoint
from minicells.vocab import CharVocab

def test_checkpoint_round_trip(tmp_path):
    vocab=CharVocab(); config={"model":{"vocab_size":"auto","num_cells":8,"max_seq_len":8},"data":{"min_length":1,"max_length":8,"random_fraction":.7},"train":{"seed":1},"validation":{"seed":2,"examples":4}}
    config=resolved_config(config,len(vocab)); model=EchoModel(**config["model"]); batch=fixed_dataset(vocab,2,4,min_length=1,max_length=8,num_cells=8)
    before=evaluate(model,batch); path=tmp_path/"model.pt"; save_checkpoint(path,model,None,config,0,before); loaded,payload=load_checkpoint(path)
    assert payload["step"] == 0; assert before == evaluate(loaded,batch); assert torch.equal(model(batch.input_ids),loaded(batch.input_ids))
