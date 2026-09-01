from __future__ import annotations
import json
from pathlib import Path
import torch
from minicells.real_representation_009b3_experiment import AddressRow, build_rows, confirmation_gate, effects, effect_metrics, fit_basis, fit_predict, ood_predict, param_count, select_router

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL=ROOT/"research"/"validations"/"core-009b3-deployable-effect-addressability"/"protocol.json"
def P(): return json.loads(PROTOCOL.read_text())
def _rows():
    g=torch.Generator().manual_seed(812); out=[]
    for i in range(72):
        z=torch.randn(6,64,generator=g,dtype=torch.float64)*.01; x=((i%9)-4)/4; y=(((i*5)%11)-5)/5; z[:,0]+=x; z[:,1]+=y
        a=torch.zeros(64,dtype=torch.float64); a[0]=.8*x-.2*y; a[1]=.3*x+.9*y
        out.append(AddressRow("train" if i<54 else "eval",f"s{i%6}",f"{i:064x}",z,a))
    return out
def test_prefix_boundary_excludes_suffix():
    class S: pass
    ss=[]
    for shift in [0.,999.]:
        s=S(); s.partition="train"; s.source="s"; s.token_sha256=str(shift); s.z=torch.zeros(8,64,dtype=torch.float64); s.z[:,63]=3; s.z[4:,0]=shift; s.ghat=torch.zeros(64,64,dtype=torch.float64); ss.append(s)
    a,_=build_rows([ss[0]],.5,1); b,_=build_rows([ss[1]],.5,1); assert torch.equal(a[0].prefix_z,b[0].prefix_z) and len(a[0].prefix_z)==4
def test_ridge_recovers_simple_mapping():
    rows=_rows(); V=fit_basis(rows,2); tr=[x for x in rows if x.partition=="train"]; ev=[x for x in rows if x.partition=="eval"]; spec=P()["router_candidates"][0]
    beta,meta=fit_predict(spec,tr,ev,V,81201); m=effect_metrics(beta@V.T,effects(ev,V)); assert m["median_normalized_effect_residual"]<.08 and m["median_effect_cosine"]>.99 and meta["parameter_count"]<100000
def test_source_holdout_covers_sources():
    rows=_rows(); V=fit_basis(rows,2); tr=[x for x in rows if x.partition=="train"]; ev=[x for x in rows if x.partition=="eval"]; q,idx,src=ood_predict(P()["router_candidates"][0],tr,ev,V,81201)
    assert len(idx)==len(ev) and len(set(src))>=5
def _payload(seed,ridge=True,mlp=True,attn=True):
    def c(name,fam,good):
        r=.3 if good else .9; co=.85 if good else .2; pos=.95 if good else .4
        return {"name":name,"family":fam,"parameter_count":1000,"iid":{"median_normalized_effect_residual":r,"median_effect_cosine":co,"positive_dot_fraction":pos},"ood":{"median_normalized_effect_residual":r+.1,"median_effect_cosine":co-.05,"positive_dot_fraction":pos-.05,"source_count":6},"ood_residual_improvement_over_mean":.2 if good else -.1}
    return {"seed":seed,"oracle_basis":{"median_normalized_effect_residual":.1,"p90_normalized_effect_residual":.2},"candidates":[c("ridge_prefix_mean","ridge",ridge),c("mlp_prefix_mean","mlp",mlp),c("tiny_attention_prefix","tiny_attention",attn)]}
def test_hierarchy_prefers_ridge():
    s,_=select_router([_payload(81201),_payload(81202)],P()); assert s=="ridge_prefix_mean"
def test_discovery_requires_both_seeds():
    s,_=select_router([_payload(81201)],P()); assert s is None
def test_confirmation_distinguishes_ood_failure():
    q={"seed":81211,"router_name":"ridge_prefix_mean","router_parameter_count":1000,"ood_source_count":6,"oracle_basis":{"median_normalized_effect_residual":.1,"p90_normalized_effect_residual":.2},"causal":{"oracle_descent_fraction":1.,"iid_deploy_descent_fraction":.95,"ood_deploy_descent_fraction":.9,"median_iid_deploy_over_oracle_target_gain":.8,"median_ood_deploy_over_oracle_target_gain":.7,"median_mean_over_oracle_target_gain":.5,"median_nn_over_oracle_target_gain":.78,"median_iid_deploy_excess_unrelated_harm_over_oracle_gain":.02}}
    g=confirmation_gate(q,P()); assert g["pass"]; q["causal"]["median_ood_deploy_over_oracle_target_gain"]=.4; g=confirmation_gate(q,P()); assert g["iid_pass"] and not g["ood_pass"] and not g["pass"]
def test_protocol_leakage_and_budget():
    p=P(); assert p["discovery"]["seeds"]==[81201,81202] and p["confirmation"]["seeds"]==[81211,81212,81213] and p["context"]["prefix_fraction"]==.5
    assert not p["scope"]["gradient_as_router_input"] and not p["scope"]["target_label_as_router_input"] and not p["scope"]["future_token_as_router_input"] and p["router_selection"]["causal_metrics_forbidden_for_selection"]
    for s in p["router_candidates"]: assert param_count(s,64,32)<=100000
