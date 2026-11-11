# Literature and Theory / 文献与理论

This directory holds versioned theory documents and literature syntheses for
Cellular Language Models. These documents propose research questions; they do
not authorize training or replace frozen experiment protocols and results.

本目录收录 Cellular Language Models 的版本化理论文档与文献综述。研究提案不构成训练授权，也不覆盖冻结的实验协议或历史结果。

## Documents / 文档

| Document / 文档 | 中文 | English | Scope / 范围 |
|---|---|---|---|
| EPCLM-002 — Persistent State Theory v0.2 | [原文](EPCLM-002-Persistent-State-Theory-v0.2.zh-CN.md) | [Translation](EPCLM-002-Persistent-State-Theory-v0.2.en.md) | Predictive sufficiency, state updates/retrieval, SSM–Titans–local attention evidence, portability, and proposed ablations / 预测充分性、状态更新与检索、组合架构证据、可移植性及消融提案 |

The Chinese v0.2 file is preserved verbatim from the supplied source dated
2026-11-08. The English version follows its section and reference numbering.
Publication statuses, numerical claims, and literature-search boundaries are
reported from that source, without a new independent literature audit.

中文 v0.2 按用户提供的 2026-11-08 原文完整收录。英文版对应原文章节与参考文献编号。论文状态、数值结论与检索边界沿用原文，本次归档与翻译未进行新的独立文献核验。

## Existing reference collection / 已有参考资料

- [M3C references and research boundaries / M3C 参考资料与研究边界](../experiments/mamba3-cellular-series/REFERENCES.md): existing paper collection covering Mamba/SSD, hybrid attention, DART, and routing, with original-paper and implementation links.
- [M3C design / M3C 研究设计](../experiments/mamba3-cellular-series/DESIGN.zh-CN.md): the experiment-series design and its architectural boundaries.

Keep experiment-specific bibliographies with their experiments and link them
here rather than duplicating their paper entries.

各实验已有参考文献保留在原位置，通过本索引关联，不重复维护相同论文条目。

## Papers / 论文

- **Titans: Learning to Memorize at Test Time** (NeurIPS 2025). Adaptive neural memory, surprise momentum and forgetting. [Proceedings entry](https://papers.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html) · [P4 discussion and ablation plan](../experiments/p4-peer-titans-collaboration/docs/TITANS-MEMORY-POLICY.md)
