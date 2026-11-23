# Query

Automatic mode scores every written registered Cell independently with the
same prompt and a fresh working cache. It averages the L2 norm of the native
Shared Memory read vector over heads, tokens and layers. It ranks high scores
first with UUID tie breaking, then selects up to `selection_top_k` Cells. It
opens new temporary streams for the selected Cells and uses the existing
full Routed + Shared hierarchical equal fusion to generate.

Pass a list of IDs for manual selection or `[]` for no memory. The result
includes `text`, `mode`, candidate and selected IDs, scores, fallback reason,
latency in seconds, token count and diagnostics. No registry gives
`no_registered_cells`. An empty selector result gives `no_match`. An explicit
empty list gives `explicit_no_cells`. Empty Cells appear among candidates and
are listed as unscored. By default there is no calibrated threshold. A supplied
threshold is a minimum norm and can reject all scored Cells. The registry cap
raises `CellLimitError` before scoring and never truncates candidates.
