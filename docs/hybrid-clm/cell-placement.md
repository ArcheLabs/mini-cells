# Cell placement

Placement is explicit and serializable:

```python
CellPlacement(layer=7, experts=[1, 4, 9, 12])
```

The serialized identity records `layer_index`, `module_path`, `expert_ids`,
`placement_type`, `architecture_signature`, and `module_signature`. A placement
is structural metadata, not a claim that the location is scientifically
optimal. Auto-placement/search is reserved for a separately registered
research protocol.
