"""Validation-only, full-population path and non-entry token/Cell statistics."""
import math
from collections import Counter


class GraphEvidence:
    def __init__(self, experts):
        self.experts = experts
        self.paths = Counter()
        self.joint = Counter()
        self.edges = Counter()

    def add(self, paths, ended, tokens):
        token_values = tokens.detach().reshape(-1).cpu().tolist() if tokens is not None else None
        for row_index, (row, stop) in enumerate(zip(paths, ended)):
            visited = [int(cell) for cell in row if cell >= 0]
            terminal = 'STOP' if stop >= 0 else 'FORCED_STOP'
            self.paths[tuple(visited) + (terminal,)] += 1
            for source, destination in zip(visited, visited[1:]):
                self.edges[source, destination] += 1
            if terminal == 'STOP':
                self.edges[visited[-1], self.experts] += 1
            if token_values is not None:
                # Fixed entry is excluded; count executed non-entry visits, including revisits.
                for cell in visited[1:]:
                    self.joint[cell, token_values[row_index]] += 1

    def stats(self):
        total = sum(self.paths.values())
        entropy = -sum((n / total) * math.log(n / total) for n in self.paths.values()) if total else None
        cells, tokens = Counter(), Counter()
        for (cell, token), n in self.joint.items():
            cells[cell] += n
            tokens[token] += n
        count = sum(cells.values())
        mi = sum(n / count * math.log(n * count / (cells[cell] * tokens[token]))
                 for (cell, token), n in self.joint.items()) if count else None
        matrix = [[self.edges[i, j] for j in range(self.experts + 1)] for i in range(self.experts)]
        row_entropy = []
        for row in matrix:
            mass = sum(row)
            row_entropy.append(-sum(n / mass * math.log(n / mass) for n in row if n) if mass else None)
        # Empirical edge support, including STOP; forced truncation is not a learned edge.
        return {'path_entropy_nats': entropy, 'path_diversity': len(self.paths),
                'path_population': total, 'path_counts': [{'path': list(path), 'count': n}
                                                       for path, n in sorted(self.paths.items(), key=lambda x: str(x[0]))],
                'empirical_edge_counts': matrix, 'empirical_transition_entropy_nats': row_entropy,
                'graph_edge_support_fraction': len(self.edges) / (self.experts * (self.experts + 1)),
                'cell_token_mutual_information_nats': mi,
                'mi_population': count, 'non_entry_cell_counts': dict(cells),
                'mi_scope': 'plug-in I(Cell;input token id), executed non-entry visits; finite-sample biased; descriptive, not semantic specialization',
                'path_scope': 'full paired validation population, per FFN, terminal STOP/FORCED_STOP retained'}
