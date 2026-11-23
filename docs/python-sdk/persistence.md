# Persistence

Each committed Cell ZIP stores a schema-compatible set of FP32 recurrent tensors
and its manifest. The compatible MoM snapshot supports 24 layers with four
routed banks and one shared bank per layer. Convolution caches and offsets
belong to the working stream. They start fresh on a new write or query. An
interrupted write can restore them from its private stream checkpoint when
the text, revision and chunk size match.

An SDK runtime holds an OS lock for one storage directory. Close the first
process before sharing that directory with another. For parallel processes use
distinct directories and export/import. An unfinished collection commit
leaves a journal that the next SDK opener rolls back. Keep legacy raw-store
tools out of a directory while an SDK process owns it. Queries never persist
working caches or change revisions.
