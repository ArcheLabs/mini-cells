# Cells and lineage

Cell IDs are UUIDs. Display names can repeat. A committed write appends to the
current recurrent state and increments revision once, independent of chunk
count. Each write processes tokens in input order. `append` is an alias.
Cancellation returns `None` and retains the latest committed revision. Repeat
the same write, with the same chunk size and `resume=True`, to continue from
the last saved chunk.

Forking creates a copy with its own ID and revision zero. The lineage records
the parent ID and its source revision, even when the parent is later deleted.
`revisions` lists recorded metadata. Native legacy bundles can supply only
the revision metadata already present when the SDK first sees them. Historical
recurrent tensors are not included in this metadata API.
