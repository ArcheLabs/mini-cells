[English](README.md) | 中文

# MiniCells

MiniCells 研究能否通过可独立管理的神经 Cell，把预训练语言模型逐步转化
为 Cellular Language Model（CLM，细胞语言模型）。

**HybridCLM 是当前桥梁：** 预训练 MoE 继续作为成熟的计算基底，在模型内部
挂接可训练、可逆的 Cell mutation。长期目标是逐步减少对原始模型的依赖，
走向独立 CLM；如果受控比较显示出真正优势，HybridCLM 也可能成为模块化的
后训练/模型演化机制。

本仓库发布 HybridCLM 工具包，同时保留历史 MiniCells 科研记录。这里没有
声称“MoE → 独立 CLM”、持续学习、灾难性遗忘控制或 HybridCLM 优于 LoRA
已经解决。

## 工程证据 · 正式验证尚未开始

PCU Hybrid Reattachment 001 v3 当前记录：

- 同 cellular zero-state 等价性通过；
- Cell OFF ranking 6.25%，Cell ON ranking 82.03%；
- 支持因果 hybrid consumption 与精确恢复；
- alpha=1 locality threshold 失败；
- 粗粒度 amplitude sweep 未找到预注册的联合通过点；
- 正式验证尚未运行。

这些是工程边界，不是正式 HybridCLM 结论。

## 公共 API

```python
from minicells import CellMutation, CellPlacement, HybridCLM

hybrid = HybridCLM.from_pretrained(
    "ibm-granite/granite-3.1-1b-a400m-base",
    revision="<不可变的 Hub commit>",
)
hybrid.cellularize(CellPlacement(layer=7, experts="all"))
mutation = CellMutation.from_pretrained("<mutation-artifact>")
hybrid.attach(mutation)
hybrid.set_alpha(mutation, 0.75)
```

使用 `pip install "mini-cells[hybrid]"` 安装 Granite/Hugging Face 可选依赖。
v0.1 支持显式 placement、结构检查、安全 safetensors mutation、attach/detach、
alpha 缩放、zero-state 诊断和回滚报告。当前只支持经过测试的 Granite MoE；
未知架构会 fail closed。

## 仓库导航

- [`src/minicells/hybrid/`](src/minicells/hybrid/)：公共 HybridCLM API。
- [`docs/hybrid-clm/`](docs/hybrid-clm/)：API、artifact、placement、安全和状态文档。
- [`research/stages/08-hybrid-clm/`](research/stages/08-hybrid-clm/)：当前科研边界与路线图。
- [`research/`](research/README.zh-CN.md)：历史协议、报告和证据目录。
- [`artifacts/`](artifacts/)：持久证据与发布资产。
- [`tests/`](tests/)：单元、集成、科研测试和公共 fixtures。
- [`docs/integrations/minijam.md`](docs/integrations/minijam.md)：MiniCells/MiniJAM 边界。

## 科研状态与非目标

自动最优 placement、通用 MoE 支持、独立转换、新 router 训练、正式
HybridCLM 执行、LoRA 优越性、生产 JAM 执行和通用持续学习方案，均不属于
此次 prerelease 范围。

## 许可证

见 [LICENSE](LICENSE)。
