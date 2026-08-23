def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def architecture_stats(model, batch: int = 1) -> dict[str, int | str]:
    count = parameter_count(model)
    update_input = (2 * model.radius + 1) * model.hidden_dim + model.embedding_dim
    update_macs = update_input * model.mlp_width + model.mlp_width * model.hidden_dim
    cellular = batch * model.num_cells * model.iterations * update_macs
    output = batch * model.num_cells * model.hidden_dim * model.vocab_size
    return {
        "parameter_count": count,
        "estimated_fp32_weight_bytes": count * 4,
        "estimated_fp16_weight_bytes": count * 2,
        "estimated_int8_weight_bytes": count,
        "estimated_macs": cellular + output,
        "compute_note": "architecture-level compute proxy; not a PVM gas estimate",
    }
