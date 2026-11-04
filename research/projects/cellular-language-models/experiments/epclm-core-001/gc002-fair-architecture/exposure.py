"""Document-position accounting; deterministic historical reconstruction is not observation."""
import hashlib
import json
import random
import zlib


class ExposureLedger:
    def __init__(self,dataset_prediction_tokens,identity=None):
        self.dataset_prediction_tokens=dataset_prediction_tokens
        self.consumed=0;self.updates=0;self.seen=set();self.order_hash='0'*64;self.tokens_per_update=[];self.identity=identity or {}

    def commit(self,events,accepted=True):
        if not accepted:return
        if not events:raise ValueError('empty committed update')
        for event in events:
            if len(event)!=4:raise ValueError('invalid event shape')
            doc,position,input_id,target_id=event
            if not isinstance(doc,str) or not all(isinstance(v,int) for v in (position,input_id,target_id)):raise ValueError('invalid event types')
            if position<1 or min(input_id,target_id)<0:raise ValueError('invalid token event')
        digest=hashlib.sha256(bytes.fromhex(self.order_hash)+json.dumps(events,separators=(',',':')).encode()).hexdigest()
        for doc,position,_,_ in events:self.seen.add((doc,position))
        self.order_hash=digest
        self.consumed+=len(events);self.updates+=1;self.tokens_per_update.append(len(events))

    def snapshot(self):
        return {'format':'gc002.exposure.v1','identity':self.identity,'dataset_prediction_tokens':self.dataset_prediction_tokens,
                'consumed':self.consumed,'updates':self.updates,'order_hash':self.order_hash,
                'tokens_per_update':self.tokens_per_update.copy(),'seen_zlib':zlib.compress(json.dumps(sorted(self.seen),separators=(',',':')).encode())}

    @classmethod
    def restore(cls,saved,identity):
        if saved['format']!='gc002.exposure.v1' or saved['identity']!=identity:raise ValueError('exposure identity mismatch')
        result=cls(saved['dataset_prediction_tokens'],identity)
        result.seen={tuple(row) for row in json.loads(zlib.decompress(saved['seen_zlib']))}
        result.consumed=saved['consumed'];result.updates=saved['updates'];result.order_hash=saved['order_hash'];result.tokens_per_update=saved['tokens_per_update'].copy()
        if len(result.order_hash)!=64 or len(bytes.fromhex(result.order_hash))!=32:raise ValueError('invalid exposure hash')
        if sum(result.tokens_per_update)!=result.consumed or len(result.tokens_per_update)!=result.updates or len(result.seen)>result.consumed:raise ValueError('invalid exposure counters')
        return result

    def report(self):
        return {'consumed_training_tokens':self.consumed,'unique_training_tokens':len(self.seen),
                'repeated_training_tokens':self.consumed-len(self.seen),'optimizer_updates':self.updates,
                'tokens_per_update':self.tokens_per_update,'data_order_hash':self.order_hash,
                'token_equivalent_epochs':self.consumed/self.dataset_prediction_tokens,
                'unique_unit':'document ID plus next-token target position; not lexical Token ID',
                'hash_order':'SHA256 chain of committed update event arrays, time-step then lane; identical update boundaries required'}


def reconstruct(documents,seed,target=250000,batch=4,unroll=64):
    ids=[d['id'] for d in documents]
    if len(set(ids))!=len(ids) or any(len(d['tokens'])<2 for d in documents):raise ValueError('invalid/duplicate documents')
    available=sum(len(d['tokens'])-1 for d in documents);ledger=ExposureLedger(available)
    rng=random.Random(seed);order=list(range(len(documents)));rng.shuffle(order)
    cursor=0;lanes=[None]*batch;positions=[0]*batch
    while ledger.consumed<target:
        events=[]
        for _ in range(unroll):
            if ledger.consumed+len(events)>=target:break
            for lane in range(batch):
                if lanes[lane] is None or positions[lane]>=len(documents[lanes[lane]]['tokens'])-1:
                    if cursor==len(order):rng.shuffle(order);cursor=0
                    lanes[lane]=order[cursor];cursor+=1;positions[lane]=0
                if ledger.consumed+len(events)>=target:continue
                doc=documents[lanes[lane]];p=positions[lane]
                events.append((doc['id'],p+1,doc['tokens'][p],doc['tokens'][p+1]));positions[lane]+=1
        ledger.commit(events)
    return {'status':'EXPECTED_STREAM_RECONSTRUCTED_NOT_HISTORICALLY_OBSERVED','seed':seed,
            'dataset_prediction_tokens':available,'expected':ledger.report(),
            'actual_historical_consumption_order':'NOT_RECORDED_CANNOT_RETROACTIVELY_PROVE',
            'basis':'frozen trainer Python shuffle, per-lane document reset and accepted token scheduling; no model execution',
            'historical_use':'reference only; new experiments must persist the live committed ledger and checkpoint it atomically'}
