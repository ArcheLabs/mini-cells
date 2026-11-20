"""Deterministic, paired vector tasks for HAM-003 M1."""
import numpy as np


def unit(x):
    return x / np.linalg.norm(x,axis=-1,keepdims=True)


def generate(seed, name, heads=2, dimension=32):
    rng = np.random.default_rng(seed)
    def vectors(n):
        return unit(rng.normal(size=(n,heads,dimension)))
    k,v = vectors(24),vectors(24)
    if name == "basic":
        return k,v,k.copy(),v.copy(),{"segments":[24]}
    if name == "correlated":
        a,b=k[::2],k[1::2]
        b=unit(b-(a*b).sum(-1,keepdims=True)*a)
        k[1::2]=.95*a+np.sqrt(1-.95**2)*b
        return k,v,unit(k+.01*rng.normal(size=k.shape)),v.copy(),{"rho":.95}
    if name == "composition":
        entity,relation=vectors(8),vectors(3)
        keys=unit((entity[:,None]+relation[None]).reshape(k.shape))
        return keys,v,keys.copy(),v.copy(),{"entities":8,"relations":3,"scope":"seen pairs"}
    if name in ("parity","conditional"):
        bits=np.array([[(-1)**(i>>j&1) for j in range(4)] for i in range(16)])
        order=rng.permutation(len(bits))
        axes=rng.permutation(4)
        bits=bits[order]
        x=np.zeros((16,heads,dimension))
        x[:,:,:4]=bits[:,None]/2
        label=(bits[:,axes[:3]].prod(-1) if name=="parity" else
               np.where(bits[:,axes[0]]>0,bits[:,axes[1]],bits[:,axes[2]]))*rng.choice([-1,1])
        targets=np.zeros_like(x)
        targets[:,:,0]=label[:,None]
        return np.concatenate((x[:12],k[:4])),np.concatenate((targets[:12],v[:4])),x[12:],targets[12:],{"seen":12,"distractors":4,"held_out":4}
    if name == "multi_hop":
        nodes=vectors(9)
        return nodes[:-1],nodes[1:],nodes[:6],nodes[3:],{"hops":3}
    if name == "open_set":
        return k,v,np.concatenate((k,vectors(24))),np.concatenate((v,np.zeros_like(v))),{"known":24}
    if name == "continual":
        return k,v,k.copy(),v.copy(),{"segments":[8,8,8]}
    raise ValueError(name)
