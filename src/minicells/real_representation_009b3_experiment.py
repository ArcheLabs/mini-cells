"""Core 009B-3: deployable effect addressability."""
from __future__ import annotations
import hashlib, math, statistics
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
from .real_representation_009b1_experiment import _nlls, baseline_nlls, eta_for_target_ratio, fit_train_carrier, select_peers
from .real_representation_009b2_experiment import EffectSequence, fit_uncentered_basis

EPS=1e-12
med=lambda x: float(statistics.median(x)) if x else 0.0
mean=lambda x: float(statistics.fmean(x)) if x else 0.0
def p90(x):
    if not x:return 0.0
    y=sorted(map(float,x)); return y[max(0,min(len(y)-1,math.ceil(.9*len(y))-1))]

@dataclass(frozen=True)
class AddressRow:
    partition:str; source:str; token_sha256:str; prefix_z:torch.Tensor; effect:torch.Tensor
    @property
    def prefix_mean(self): return self.prefix_z.mean(0)

def build_rows(seqs,prefix_fraction=.5,minimum_prefix_tokens=1):
    r=fit_train_carrier(seqs).double(); out=[]
    for s in seqs:
        if s.partition not in {"train","eval"}: continue
        z=s.z.detach().cpu().double()
        n=max(minimum_prefix_tokens,int(math.floor(len(z)*prefix_fraction))); n=max(1,min(len(z),n))
        out.append(AddressRow(str(s.partition),str(s.source),s.token_sha256,z[:n].contiguous(),s.ghat.detach().cpu().double()@r))
    if not out: raise ValueError("empty address rows")
    return out,r

def fit_basis(rows,d):
    tr=[x for x in rows if x.partition=="train"]
    ers=[EffectSequence(x.partition,x.source,x.token_sha256,x.effect,float(torch.linalg.norm(x.effect))) for x in tr]
    v,_=fit_uncentered_basis(ers); return v[:,:int(d)].contiguous()

def betas(rows,V): return torch.stack([V.T@x.effect for x in rows]).double()
def effects(rows,V): return betas(rows,V)@V.T
def beta_effect(b,V): return b.double()@V.T

def effect_metrics(pred,oracle):
    rs=[]; cs=[]; pos=0
    for p,o in zip(pred.double(),oracle.double()):
        on=max(float(torch.linalg.norm(o)),EPS); pn=float(torch.linalg.norm(p)); dot=float(torch.dot(p,o))
        rs.append(float(torch.linalg.norm(p-o))/on); cs.append(0.0 if pn<=EPS else dot/(pn*on)); pos+=dot>0
    return {"count":len(rs),"median_normalized_effect_residual":med(rs),"p90_normalized_effect_residual":p90(rs),
            "median_effect_cosine":med(cs),"positive_dot_fraction":pos/max(len(rs),1)}
def oracle_basis_metrics(rows,V): return effect_metrics(effects(rows,V),torch.stack([x.effect for x in rows]))

@dataclass
class Stats: mu:torch.Tensor; sd:torch.Tensor
def stats(rows):
    x=torch.cat([r.prefix_z for r in rows]); return Stats(x.mean(0),x.std(0,unbiased=False).clamp_min(1e-6))
def xmean(rows,s): return torch.stack([((r.prefix_z-s.mu)/s.sd).mean(0) for r in rows]).double()
def xtokens(rows,s):
    T=max(len(r.prefix_z) for r in rows); D=rows[0].prefix_z.shape[1]
    x=torch.zeros(len(rows),T,D); m=torch.zeros(len(rows),T,dtype=torch.bool)
    for i,r in enumerate(rows):
        z=((r.prefix_z-s.mu)/s.sd).float(); x[i,:len(z)]=z; m[i,:len(z)]=True
    return x,m

class MLP(nn.Module):
    def __init__(self,D,H,O): super().__init__(); self.net=nn.Sequential(nn.Linear(D,H),nn.Tanh(),nn.Linear(H,O))
    def forward(self,x): return self.net(x)
class Attn(nn.Module):
    def __init__(self,D,H,O):
        super().__init__(); self.q=nn.Parameter(torch.randn(H,D)*.02); self.out=nn.Linear(H*D,O)
    def forward(self,x,m):
        a=torch.einsum("btd,hd->bht",x,self.q)/math.sqrt(x.shape[-1]); a=a.masked_fill(~m[:,None,:],-1e9)
        return self.out(torch.einsum("bht,btd->bhd",torch.softmax(a,-1),x).flatten(1))

def param_count(spec,D,O):
    f=spec["family"]
    if f=="ridge": return (D+1)*O
    if f=="mlp":
        H=int(spec["hidden_dim"]); return D*H+H+H*O+O
    H=int(spec["heads"]); return H*D+H*D*O+O

def _seed(seed,name,extra=""): return int.from_bytes(hashlib.sha256(f"{seed}:{name}:{extra}".encode()).digest()[:8],"big")%(2**31-1)
def fit_predict(spec,tr,ev,V,seed):
    s=stats(tr); Y=betas(tr,V); f=spec["family"]; D=tr[0].prefix_z.shape[1]; O=V.shape[1]
    if f=="ridge":
        X=xmean(tr,s); Z=xmean(ev,s); xm=X.mean(0); ym=Y.mean(0); A=X-xm
        W=torch.linalg.solve(A.T@A+float(spec["ridge_lambda"])*torch.eye(D,dtype=torch.float64),A.T@(Y-ym)); P=(Z-xm)@W+ym
    else:
        torch.manual_seed(_seed(seed,spec["name"]))
        if f=="mlp": model=MLP(D,int(spec["hidden_dim"]),O); X=xmean(tr,s).float(); Z=xmean(ev,s).float(); mask=None
        else: model=Attn(D,int(spec["heads"]),O); X,mask=xtokens(tr,s); Z,zmask=xtokens(ev,s)
        opt=torch.optim.AdamW(model.parameters(),lr=float(spec["learning_rate"]),weight_decay=float(spec["weight_decay"]))
        for _ in range(int(spec["steps"])):
            opt.zero_grad(); pred=model(X) if f=="mlp" else model(X,mask); loss=((pred-Y.float())**2).mean(); loss.backward(); opt.step()
        with torch.no_grad(): P=model(Z) if f=="mlp" else model(Z,zmask)
        P=P.double()
    return P,{"parameter_count":int(param_count(spec,D,O)),"family":f,"name":spec["name"]}

def mean_beta(tr,ev,V): return betas(tr,V).mean(0)[None,:].repeat(len(ev),1)
def nn_beta(tr,ev,V):
    s=stats(tr); A=xmean(tr,s); B=xmean(ev,s)
    A=A/torch.linalg.norm(A,dim=1,keepdim=True).clamp_min(EPS); B=B/torch.linalg.norm(B,dim=1,keepdim=True).clamp_min(EPS)
    return betas(tr,V)[torch.argmax(B@A.T,1)]

def ood_predict(spec,tr,ev,V,seed,baseline=None):
    ps=[]; idx=[]; srcs=[]
    for src in sorted({x.source for x in tr}&{x.source for x in ev}):
        a=[x for x in tr if x.source!=src]; pairs=[(i,x) for i,x in enumerate(ev) if x.source==src]
        if not a or not pairs: continue
        b=[x for _,x in pairs]
        if baseline=="mean": P=mean_beta(a,b,V)
        elif baseline=="nn": P=nn_beta(a,b,V)
        else: P,_=fit_predict(spec,a,b,V,_seed(seed,spec["name"],src))
        ps.extend(P); idx.extend(i for i,_ in pairs); srcs.append(src)
    return (torch.stack(ps) if ps else torch.empty(0,V.shape[1],dtype=torch.float64)),idx,srcs

def _cand(spec,tr,ev,V,seed):
    P,meta=fit_predict(spec,tr,ev,V,seed); O=effects(ev,V); iid=effect_metrics(beta_effect(P,V),O)
    Q,idx,srcs=ood_predict(spec,tr,ev,V,seed); ood=effect_metrics(beta_effect(Q,V),O[idx]) if idx else effect_metrics(O[:0],O[:0]); ood["source_count"]=len(set(srcs))
    M,midx,_=ood_predict(None,tr,ev,V,seed,"mean"); mm=effect_metrics(beta_effect(M,V),O[midx]) if midx else {"median_normalized_effect_residual":0}
    return {"name":spec["name"],"family":spec["family"],"parameter_count":meta["parameter_count"],"iid":iid,"ood":ood,
            "ood_residual_improvement_over_mean":float(mm["median_normalized_effect_residual"])-float(ood["median_normalized_effect_residual"])}

def run_discovery(seqs,protocol,seed,locked_dimension):
    rows,r=build_rows(seqs,float(protocol["context"]["prefix_fraction"]),int(protocol["context"]["minimum_prefix_tokens"]))
    tr=[x for x in rows if x.partition=="train"]; ev=[x for x in rows if x.partition=="eval"]; V=fit_basis(rows,locked_dimension); O=effects(ev,V)
    M=mean_beta(tr,ev,V); N=nn_beta(tr,ev,V)
    return {"format":"minicells.core-validation.deployable-effect-addressability-discovery-seed.v1","experiment_id":protocol["experiment_id"],
            "seed":int(seed),"scientific_decision":False,"locked_dimension":int(locked_dimension),"carrier_norm":float(torch.linalg.norm(r)),
            "train_count":len(tr),"eval_count":len(ev),"oracle_basis":oracle_basis_metrics(ev,V),
            "baselines":{"mean_effect":effect_metrics(beta_effect(M,V),O),"nearest_neighbor":effect_metrics(beta_effect(N,V),O)},
            "candidates":[_cand(s,tr,ev,V,seed) for s in protocol["router_candidates"]]}

def candidate_gate(c,ob,p,seed):
    g=p["discovery"]["gates"]; I=c["iid"]; O=c["ood"]
    checks=[ob["median_normalized_effect_residual"]<=g["maximum_oracle_basis_median_residual"],ob["p90_normalized_effect_residual"]<=g["maximum_oracle_basis_p90_residual"],
            I["median_normalized_effect_residual"]<=g["maximum_iid_median_effect_residual"],I["median_effect_cosine"]>=g["minimum_iid_median_effect_cosine"],I["positive_dot_fraction"]>=g["minimum_iid_positive_dot_fraction"],
            O["median_normalized_effect_residual"]<=g["maximum_ood_median_effect_residual"],O["median_effect_cosine"]>=g["minimum_ood_median_effect_cosine"],O["positive_dot_fraction"]>=g["minimum_ood_positive_dot_fraction"],
            O["source_count"]>=g["minimum_ood_source_count"],c["ood_residual_improvement_over_mean"]>=g["minimum_ood_residual_improvement_over_mean"],c["parameter_count"]<=g["maximum_parameter_count"]]
    return {"seed":seed,"name":c["name"],"family":c["family"],"parameter_count":c["parameter_count"],
            "iid_median_effect_residual":I["median_normalized_effect_residual"],"ood_median_effect_residual":O["median_normalized_effect_residual"],
            "ood_source_count":O["source_count"],"ood_residual_improvement_over_mean":c["ood_residual_improvement_over_mean"],"pass":all(checks)}

def select_router(payloads,p):
    rows=[]
    for spec in p["router_candidates"]:
        per=[candidate_gate(next(c for c in q["candidates"] if c["name"]==spec["name"]),q["oracle_basis"],p,q["seed"]) for q in payloads]
        rows.append({"name":spec["name"],"family":spec["family"],"priority":spec["priority"],"parameter_count":max([x["parameter_count"] for x in per],default=0),
                     "all_completed_seed_rows_viable":bool(per) and all(x["pass"] for x in per),"per_seed":per})
    rows.sort(key=lambda x:(x["priority"],x["parameter_count"],x["name"]))
    if len(payloads)!=len(p["discovery"]["seeds"]): return None,rows
    for fam in p["router_selection"]["hierarchy"]:
        v=[x for x in rows if x["family"]==fam and x["all_completed_seed_rows_viable"]]
        if v:return v[0]["name"],rows
    return None,rows

def _dir(a,r): return torch.outer(a.double(),r.double())
def _ratio(a,b): return None if b<=EPS else float(a/b)
def run_confirmation(seqs,p,seed,locked_dimension,router_name,u,lm_head_weight,device):
    rows,r=build_rows(seqs,float(p["context"]["prefix_fraction"]),int(p["context"]["minimum_prefix_tokens"]))
    tr=[x for x in rows if x.partition=="train"]; ev=[x for x in rows if x.partition=="eval"]; ce=[x for x in seqs if x.partition=="eval"]; by={x.token_sha256:x for x in ce}
    V=fit_basis(rows,locked_dimension); O=effects(ev,V); spec=next(x for x in p["router_candidates"] if x["name"]==router_name)
    P,meta=fit_predict(spec,tr,ev,V,seed); M=mean_beta(tr,ev,V); N=nn_beta(tr,ev,V); Q,qidx,qsrc=ood_predict(spec,tr,ev,V,seed); qmap={i:b for i,b in zip(qidx,Q)}
    E={"deploy":beta_effect(P,V),"mean":beta_effect(M,V),"nn":beta_effect(N,V)}
    base=baseline_nlls(ce,u,lm_head_weight,device=device); rho=float(p["confirmation"]["causal_scale"]["rho"]); npeer=int(p["confirmation"]["unrelated_different_source_peers_per_target"]); out=[]
    for i,row in enumerate(ev):
        t=by[row.token_sha256]; eta=eta_for_target_ratio(t,t.ghat,rho); _,peers=select_peers(t,ce,seed=seed,matched_count=0,unrelated_count=npeer); S=[t,*peers]
        vs={"oracle":O[i],"deploy":E["deploy"][i],"mean":E["mean"][i],"nn":E["nn"][i]}
        if i in qmap:vs["ood_deploy"]=V@qmap[i]
        nl={k:_nlls(S,_dir(a,r),eta,u,lm_head_weight,device=device) for k,a in vs.items()}; b=base[t.token_sha256]; gains={k:b-v[0] for k,v in nl.items()}
        oh=[max(x-base[z.token_sha256],0.) for z,x in zip(peers,nl["oracle"][1:])]; dh=[max(x-base[z.token_sha256],0.) for z,x in zip(peers,nl["deploy"][1:])]
        excess=None if gains["oracle"]<=EPS else (mean(dh)-mean(oh))/gains["oracle"]
        out.append({"token_sha256":row.token_sha256,"source":row.source,"oracle_gain":gains["oracle"],"deploy_gain":gains["deploy"],"ood_deploy_gain":gains.get("ood_deploy"),
                    "mean_gain":gains["mean"],"nn_gain":gains["nn"],"deploy_excess_unrelated_harm_over_oracle_gain":excess})
    orag=[x["oracle_gain"] for x in out]; dg=[x["deploy_gain"] for x in out]; og=[x["ood_deploy_gain"] for x in out if x["ood_deploy_gain"] is not None]
    def ratios(k): return [z for x in out if x.get(k) is not None and (z:=_ratio(x[k],x["oracle_gain"])) is not None]
    causal={"oracle_descent_fraction":sum(x>0 for x in orag)/len(orag),"iid_deploy_descent_fraction":sum(x>0 for x in dg)/len(dg),"ood_deploy_descent_fraction":sum(x>0 for x in og)/max(len(og),1),
            "median_iid_deploy_over_oracle_target_gain":med(ratios("deploy_gain")),"median_ood_deploy_over_oracle_target_gain":med(ratios("ood_deploy_gain")),
            "median_mean_over_oracle_target_gain":med(ratios("mean_gain")),"median_nn_over_oracle_target_gain":med(ratios("nn_gain")),
            "median_iid_deploy_excess_unrelated_harm_over_oracle_gain":med([x["deploy_excess_unrelated_harm_over_oracle_gain"] for x in out if x["deploy_excess_unrelated_harm_over_oracle_gain"] is not None])}
    return {"format":"minicells.core-validation.deployable-effect-addressability-confirmation-seed.v1","experiment_id":p["experiment_id"],"seed":seed,"scientific_decision":False,
            "locked_dimension":locked_dimension,"router_name":router_name,"router_family":spec["family"],"router_parameter_count":meta["parameter_count"],"ood_source_count":len(set(qsrc)),
            "oracle_basis":oracle_basis_metrics(ev,V),"causal":causal,"target_rows":out}

def confirmation_gate(q,p):
    g=p["confirmation"]["gates"]; b=q["oracle_basis"]; c=q["causal"]
    iid=(b["median_normalized_effect_residual"]<=g["maximum_oracle_basis_median_residual"] and b["p90_normalized_effect_residual"]<=g["maximum_oracle_basis_p90_residual"]
         and c["oracle_descent_fraction"]>=g["minimum_oracle_descent_fraction"] and c["median_iid_deploy_over_oracle_target_gain"]>=g["minimum_iid_deploy_over_oracle_target_gain"]
         and c["iid_deploy_descent_fraction"]>=g["minimum_iid_deploy_descent_fraction"] and c["median_iid_deploy_excess_unrelated_harm_over_oracle_gain"]<=g["maximum_iid_deploy_excess_unrelated_harm_over_oracle_gain"]
         and c["median_iid_deploy_over_oracle_target_gain"]-c["median_mean_over_oracle_target_gain"]>=g["minimum_iid_gain_recovery_margin_over_mean"]
         and c["median_iid_deploy_over_oracle_target_gain"]-c["median_nn_over_oracle_target_gain"]>=g["minimum_iid_gain_recovery_minus_nn"] and q["router_parameter_count"]<=g["maximum_parameter_count"])
    ood=(c["median_ood_deploy_over_oracle_target_gain"]>=g["minimum_ood_deploy_over_oracle_target_gain"] and c["ood_deploy_descent_fraction"]>=g["minimum_ood_deploy_descent_fraction"] and q["ood_source_count"]>=g["minimum_ood_source_count"])
    return {"seed":q["seed"],"router_name":q["router_name"],"iid_pass":bool(iid),"ood_pass":bool(ood),"pass":bool(iid and ood),**c}
