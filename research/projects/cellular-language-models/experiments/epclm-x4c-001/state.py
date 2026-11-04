"""Equal 2048-element states, query-conditioned read and gated residual writes."""
import torch
from torch import nn
from torch.nn import functional as F
from common import native


class PersistentState(nn.Module):
    def __init__(self,dim,kind='cellular',cells=16,width=128,read_dim=32):
        super().__init__()
        if kind not in ('flat','cellular'):raise ValueError(kind)
        self.kind=kind;self.cells=cells;self.width=width;self.summary=nn.Linear(dim,32,bias=False)
        self.norm=native.RMSNorm(width);self.query=nn.Linear(dim,read_dim,bias=False)
        self.key=nn.Linear(width,read_dim,bias=False);self.value=nn.Linear(width,read_dim,bias=False)
        self.output=nn.Linear(read_dim,dim,bias=False);self.read_gate=nn.Linear(dim,1)
        # Flat is a nonlinear dense low-rank update, not a weak linear memory control.
        hidden=8 if kind=='flat' else 120
        input_dim=cells*width+32 if kind=='flat' else width+32
        output_dim=cells*width if kind=='flat' else width
        self.write_in=nn.Linear(input_dim,hidden,bias=False)
        self.write_value=nn.Linear(hidden,output_dim,bias=False);self.write_gate=nn.Linear(hidden,output_dim)
        nn.init.constant_(self.write_gate.bias,-2.)
        shifts=(torch.arange(32)[None]-2*torch.arange(cells)[:,None])%32
        self.register_buffer('coordinates',shifts,persistent=False)

    def init(self,batch):
        shape=(batch,self.cells*self.width) if self.kind=='flat' else (batch,self.cells,self.width)
        return self.query.weight.new_zeros(shape)

    def slots(self,state):return state.reshape(state.shape[0],self.cells,self.width)

    def read(self,h,state):
        slots=self.norm(self.slots(state));weights=(self.query(h)@self.key(slots).transpose(-1,-2))/(self.query.out_features**.5)
        context=weights.softmax(-1)@self.value(slots)
        return h+torch.sigmoid(self.read_gate(h))*self.output(context)

    def write(self,state,hidden):
        summary=self.summary(hidden.mean(1))
        if self.kind=='flat':inputs=torch.cat((state,summary),-1)
        else:
            z=self.norm(state);local=z+.1*(z.roll(1,1)+z.roll(-1,1))/2
            inputs=torch.cat((local,summary[:,self.coordinates]),-1)
        middle=F.silu(self.write_in(inputs));delta=self.write_value(middle)
        return state+.1*torch.sigmoid(self.write_gate(middle))*delta
