# Import and export

Single Cell export preserves the native two-entry ZIP and Safetensors data.
That manifest records format and schema versions, Cell metadata, model identity
and the SHA-256 of tensor bytes. Layer/head/bank dimensions and FP32 are
validated against the pinned runtime. SDK-written ZIPs add optional revision
metadata to the manifest.

Collection version 1 contains a catalog and native bundles with individual
SHA-256 hashes. Entry names, hashes, model compatibility and tensors are
verified before commit. Import defaults to conflict error and preserves IDs
and lineage. A collision or invalid bundled member leaves the registry
unchanged. Export atomically replaces the destination after writing the full
archive. Store exported files outside UUID-named registry paths.
