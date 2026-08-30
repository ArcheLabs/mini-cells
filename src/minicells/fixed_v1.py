"""Debug-only Python mirror of canonical Rust Q8.8 Echo V0."""
from __future__ import annotations
import hashlib, struct

PARAMETER_COUNT=4476; MODEL_BYTES=8952; NUM_CELLS=64; MAX_LEN=32; VOCAB="abcdefghijklmnopqrstuvwxyz0123456789 .,?!'-"
OFFSETS=(0,352,3168,3200,3712,3728,4432)

def unpack_model(data:bytes)->list[int]:
    if len(data)!=MODEL_BYTES: raise ValueError("Model Format V1 requires 8952 bytes")
    values=list(struct.unpack('<4476h',data))
    if any(v < -2048 or v > 2048 for v in values): raise ValueError("parameter outside V1 bounds")
    return values
def pack_model(values)->bytes:
    values=list(values)
    if len(values)!=PARAMETER_COUNT: raise ValueError("Model Format V1 requires 4476 parameters")
    return struct.pack('<4476h',*(max(-2048,min(2048,int(v))) for v in values))
def model_hash(data:bytes)->bytes:return hashlib.blake2b(b'mini-cells:model:v1'+data,digest_size=32).digest()
def rounded(v:int)->int:return (v+128)//256 if v>=0 else -((-v+128)//256)
def linear(p,w,b,x,row):return rounded(p[b+row]*256+sum(p[w+row*len(x)+i]*v for i,v in enumerate(x)))
def encode(text:str):
    if len(text)>MAX_LEN:raise ValueError("maximum is 32 characters")
    try: ids=[VOCAB.index(c)+1 for c in text]
    except ValueError as e:raise ValueError("unsupported character") from e
    return ids+[0]*(NUM_CELLS-len(ids))
def predict(data:bytes,text:str)->str:
    p=unpack_model(data);ids=encode(text);state=[[0]*16 for _ in range(64)]
    for _ in range(4):
        nxt=[[0]*16 for _ in range(64)]
        for cell in range(64):
            x=[]
            for n in range(cell-2,cell+3):x.extend(state[n] if 0<=n<64 else [0]*16)
            x.extend(p[ids[cell]*8:ids[cell]*8+8]);hidden=[max(0,linear(p,352,3168,x,row)) for row in range(32)]
            for row in range(16):nxt[cell][row]=max(-256,min(256,state[cell][row]+linear(p,3200,3712,hidden,row)))
        state=nxt
    out=[]
    for cell in range(len(text)):
        logits=[linear(p,3728,4432,state[cell],row) for row in range(44)];token=max(range(44),key=lambda i:logits[i]);out.append('' if token==0 else VOCAB[token-1])
    return ''.join(out)
