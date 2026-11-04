"""One A2 backbone; only read locations and writer-source selection differ."""
import copy
import torch
from bootstrap import old_architecture,shared
from .state import StateBank

ARMS=('B1','B2','B3')
STATE_SEED=9100101


def initial_backbone(seed=91001,**tiny):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=old_architecture.build('A2',**tiny)
    return model


class InteractionModel(old_architecture.X4C):
    def __init__(self,arm='B1',seed=91001,state_seed=STATE_SEED,**tiny):
        if arm not in ARMS:raise ValueError(arm)
        # Reconstruct the historical constructor exactly, without perturbing caller RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);super().__init__((1,1,1),**tiny)
        self.base_initial_sha256=shared.model_hash(self)
        self.arm=arm;self.read_attention=arm in ('B1','B3');self.read_ffn=arm in ('B2','B3')
        self.reset_every_chunk=False;self.fixed_training_chunks=True;self.record_diagnostics=False;self.observations=[]
        self.collect_gradients=False;self.path_tensors={'attention':[],'attention_scores':[],'ffn':[]};self.max_context=128
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(state_seed);self.persistent=StateBank(self.dim)
        self.state_initial_sha256=shared.model_hash(self.persistent)
        self.configuration.update(name=arm,iterations=[1,1,1],state_kind='learned_slot_bank_gru',
            state_shape=[16,128],state_seed=state_seed,read_attention=self.read_attention,read_ffn=self.read_ffn,
            attention_memory_heads='existing Token Attention heads; no additional multihead module',ffn_read_heads=1,
            state_key_position='unrotated content keys; memory scores use unrotated Q',
            ffn_injection='after unchanged n2 RMSNorm, before unchanged SwiGLU',writer_sources={'B1':'attention_outputs','B2':'ffn_outputs','B3':'equal_mean_attention_and_ffn'}[arm],
            chunk_size=32,partial_chunk='explicit finish/independent forward flush; never at arbitrary prefill API boundary')
        self.configuration['attention_masks']=self.attention_masks;self.configuration['ffn_masks']=self.ffn_masks

    def forward(self,ids):
        if ids.shape[1]>128:raise ValueError('historical sample limit 128')
        return self.forward_extended(ids)

    def forward_extended(self,ids,session=None,return_session=False,finish=True):
        from .runtime import parallel_forward,flush
        logits,session=parallel_forward(self,ids,session)
        if finish:flush(self,session,close=True)
        return (logits,session) if return_session else logits

    def backbone_state_dict(self):return {k:v for k,v in self.state_dict().items() if not k.startswith('persistent.')}
