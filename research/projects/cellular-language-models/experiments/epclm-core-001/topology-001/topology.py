"""Fixed channel graphs; BroadcastCore parameters, event injection and readout unchanged."""
from collections import deque
import torch
from repair_models import BroadcastCore,RepairConfig
from model import check_numerics

ARMS=('R1','IC','R2','SR','T4')


def graph(name):
    if name not in ARMS:raise ValueError('unknown topology')
    negative=[];positive=[]
    for i in range(16):
        r,c=divmod(i,4)
        if name=='IC':minus=plus=[i]
        elif name=='R1':minus=[(i-1)%16];plus=[(i+1)%16]
        elif name in ('R2','SR'):
            offset=2 if name=='R2' else 4
            minus=[(i-1)%16,(i-offset)%16];plus=[(i+1)%16,(i+offset)%16]
        else:minus=[4*r+(c-1)%4,4*((r-1)%4)+c];plus=[4*r+(c+1)%4,4*((r+1)%4)+c]
        negative.append(minus);positive.append(plus)
    neighbors=[[] if name=='IC' else negative[i]+positive[i] for i in range(16)]
    distances=[]
    for source in range(16):
        distance=[None]*16;distance[source]=0;queue=deque([source])
        while queue:
            i=queue.popleft()
            for j in neighbors[i]:
                if distance[j] is None:distance[j]=distance[i]+1;queue.append(j)
        distances.append(distance)
    lengths=[distances[i][j] for i in range(16) for j in range(16) if i!=j and distances[i][j] is not None]
    edges=sorted({tuple(sorted((i,j))) for i,row in enumerate(neighbors) for j in row})
    clustering=[]
    for row in neighbors:
        connected=sum(b in neighbors[a] for a in row for b in row if a!=b)
        clustering.append(connected/(len(row)*(len(row)-1)) if len(row)>1 else 0.)
    return {'name':name,'negative':negative,'positive':positive,'neighbors':neighbors,'degree':len(neighbors[0]),
            'undirected_edges':edges,'edge_count':len(edges),'diameter':max(lengths) if lengths else None,
            'average_shortest_path':sum(lengths)/len(lengths) if lengths else None,
            'clustering_coefficient':sum(clustering)/16,'distances':distances}


def build_ring_radius1():return graph('R1')
def build_ring_radius2():return graph('R2')
def build_shortcut_ring():return graph('SR')
def build_torus_4x4():return graph('T4')
def build_isolated():return graph('IC')


class TopologyCore(BroadcastCore):
    def __init__(self,name,config=None):
        super().__init__(config or RepairConfig(variant='C1-BROADCAST',arm='C1'))
        if self.config.cells!=16:raise ValueError('Topology-001 requires 16 Cells')
        self.topology=name;self.graph=graph(name);self.observe=False;self.observation=[]
        self.register_buffer('_negative',torch.tensor(self.graph['negative']),persistent=False)
        self.register_buffer('_positive',torch.tensor(self.graph['positive']),persistent=False)

    def channel_inputs(self,z):
        if self.topology=='IC':return z,z
        if self.topology=='R1':return z[:,self._negative[:,0]],z[:,self._positive[:,0]]
        return z[:,self._negative].mean(2),z[:,self._positive].mean(2)

    def evolve(self,state,steps=None,communication=True,trace=None):
        steps=self.config.steps if steps is None else steps
        if steps<0:raise ValueError('negative evolution count')
        if self.observe:self.observation=[]
        for _ in range(steps):
            z=self.norm(state);negative,positive=self.channel_inputs(z) if communication else (z,z)
            center=self.center(z);left=self.left(negative);right=self.right(positive)
            message=center+left+right;delta=self.update(torch.cat((z,message),-1))
            updated=state+self.config.alpha*delta
            check_numerics(updated.detach().abs().amax()<=self.config.max_state_abs,'topology state outside numerical range')
            if self.observe:
                with torch.no_grad():
                    h=updated.detach();unit=torch.nn.functional.normalize(h,dim=-1)
                    cosine=unit@unit.transpose(-1,-2);mask=~torch.eye(16,dtype=torch.bool,device=h.device)
                    self.observation.append({'cell_state_norm':h.norm(dim=-1).mean(0),'delta_norm':delta.detach().norm(dim=-1).mean(),'state_delta_norm':self.config.alpha*delta.detach().norm(dim=-1).mean(),
                        'left_message_norm':left.detach().norm(dim=-1).mean(),'center_message_norm':center.detach().norm(dim=-1).mean(),
                        'right_message_norm':right.detach().norm(dim=-1).mean(),'pairwise_cosine':cosine[:,mask].mean(),
                        'cell_state_variance':h.var(dim=1,unbiased=False).mean()})
            if trace is not None:trace.append(updated.detach().clone())
            state=updated
        return state
