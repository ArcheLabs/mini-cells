"""004 uses the shared optimizer/evidence loop with an explicit two-session plan."""
from minicells.models.graph_cell import LanguageModel


def configure(config, session):
    if session not in config['sessions']:
        raise ValueError('Unknown 004 session; choose reference or graph')
    config['session'] = session
    plan = config['sessions'][session]
    config['run_models'] = [kind for queue in plan['queues'] for kind in queue]
    config['gpu_queues'] = plan['queues']
    return config


def validate(config, protocol):
    if (config['models'] != protocol['models'] or config['seeds'] != protocol['seeds']
            or config['train_tokens'] != protocol['train_tokens_per_arm']
            or config['seed'] not in config['seeds']):
        raise ValueError('004 configuration differs from registered matrix')
    if (config['sessions'] != protocol['sessions'] or config['graph_backup_k'] != protocol['graph_backup_k']
            or config['arm_budget_seconds'] != protocol['arm_budget_seconds']
            or config['graph_backup_floor'] != protocol['graph_backup_floor']):
        raise ValueError('004 session/gradient plan differs from protocol')
    for plan in config['sessions'].values():
        if len(plan['queues']) != 2:
            raise ValueError('Expected two independent GPU queues')
        budget = max(sum(config['arm_budget_seconds'][kind] for kind in queue) for queue in plan['queues'])
        if budget + 60 + config['publication_reserve_seconds'] > config['session_budget_seconds']:
            raise ValueError('004 session exceeds application limit')
    if config['session_budget_seconds'] > 3300 or config['train_tokens'] % (4 * config['batch_size'] * config['sequence_length']):
        raise ValueError('004 time cap or exact stage boundary violated')
    import torch
    with torch.device('meta'):
        capacities = {kind: LanguageModel(config, kind).parameter_metrics()['params'] for kind in config['models']}
    values = list(capacities.values())
    if min(values) < 50_000_000 or max(values) > 150_000_000 or max(values) / min(values) - 1 > protocol['capacity_tolerance_fraction']:
        raise ValueError('004 parameter capacity mismatch')
    return capacities
