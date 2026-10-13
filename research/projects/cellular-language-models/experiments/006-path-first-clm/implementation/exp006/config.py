"""Contract gates run before assets, secrets or accelerator access."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
ROOT=next(p for p in BASE.parents if (p/'pyproject.toml').exists())
ARMS=('R0','R1','R2','M0')
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path=None):
    path=Path(path or BASE/'config/proposed.json');c=json.loads(path.read_text());p=json.loads((BASE/'protocol.json').read_text())
    validate(c);return c,p,path

def validate(c):
    m=c['model'];t=c['training']
    if m['memory_mode']!='static_input_readonly':raise ValueError('per-token reward requires path-independent static memory')
    if m['width']%m['heads'] or min(m['steps'],m['cell_count'],m['cell_hidden'])<1:raise ValueError('invalid model dimensions')
    if m['alpha']<=0 or m['max_sequence_length']<t['sequence_length']:raise ValueError('invalid state budget')
    if t['tokens']!=t['steps']*t['batch_size']*t['sequence_length']:raise ValueError('token budget mismatch')
    if t['policy']!='on_policy_hard' or t['cost_lambda']!=0:raise ValueError('R0-R2 fixed-budget policy required; R3 unregistered')
    if set(c['seeds']['development']) & set(c['seeds']['confirmation']):raise ValueError('seed leakage')
    ranges=[c['data'][k] for k in ('train_range','development_range','confirmation_range')]
    if any(a<0 or b<=a for a,b in ranges) or any(max(a,c)<min(b,d) for i,(a,b) in enumerate(ranges) for c,d in ranges[i+1:]):raise ValueError('overlapping story ranges')
    # Never use Experiment 005 validation prefix or reserved suffix.
    if c['data']['split']!='train':raise ValueError('006 v0 requires its independent train-split ranges')

def blockers(c,p):
    out=[]
    if p.get('status')!='REGISTERED':out.append('Gate 1: protocol is not registered')
    if p.get('registration_required'):out.extend(p['registration_required'])
    if any(v is None for v in p['thresholds'].values()):out.append('scientific thresholds not frozen')
    if not re.fullmatch('[0-9a-f]{64}',c['data'].get('manifest_sha256') or ''):out.append('assets hash not frozen')
    if not re.fullmatch('[0-9a-f]{64}',c['data'].get('source_sha256') or ''):out.append('source data hash not frozen')
    if not c['seeds']['development'] or len(c['seeds']['confirmation'])<3:out.append('independent seed matrix not frozen')
    evidence=BASE/'registration/evidence.json'
    if not evidence.is_file() or digest(evidence)!=p.get('registration_evidence_sha256'):
        out.append('registration evidence missing or unverified')
    else:
        from .reuse import canonical_sha256
        e=json.loads(evidence.read_text())
        if (e.get('experiment_id')!='006-path-first-clm' or
            e.get('config_sha256')!=canonical_sha256(c) or
            e.get('thresholds')!=p['thresholds'] or
            e.get('user_approved') is not True or
            e.get('seed_collision_audit_passed') is not True or
            not re.fullmatch('[0-9a-f]{64}',e.get('approval_record_sha256') or '')):
            out.append('registration evidence does not bind approval, seeds and frozen configuration')
    if len(set(c['seeds']['confirmation']))!=len(c['seeds']['confirmation']):out.append('duplicate confirmation seeds')
    return out

def authorize(c,p,seed,role='development',*,require_confirmation_manifest=True):
    missing=blockers(c,p)
    if missing:raise RuntimeError('NOT REGISTERED: '+ '; '.join(missing))
    if role not in ('development','confirmation') or seed not in c['seeds'][role]:raise RuntimeError('seed outside registered namespace')
    if role=='confirmation':
        if not (c['authorization']['confirmation'] and p['confirmation_review_approved']):raise RuntimeError('confirmation data and seeds are sealed')
        if p.get('system_candidate_arm') not in ('R1','R2'):raise RuntimeError('system candidate must be frozen from development before confirmation')
        if require_confirmation_manifest and not re.fullmatch('[0-9a-f]{64}',c['data'].get('confirmation_manifest_sha256') or ''):raise RuntimeError('confirmation manifest is not frozen')
    if role=='development' and not c['authorization']['gpu_development']:raise RuntimeError('GPU development not authorized')
