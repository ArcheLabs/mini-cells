"""Render read-only observations separately from untested causal hypotheses."""

import hashlib
import json
from pathlib import Path
import subprocess


def summarize(dest):
    dest = Path(dest)
    p = json.loads((dest / "TOKEN_DISTRIBUTION_PROOF.json").read_text())
    unit = "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/"
    source = p["source_sha"]
    refs = {
        "batch_packing": (unit + "r3m-e/implementation/minicells_r3me/data.py", 161, 177),
        "tiny_dispatch": (unit + "r3m-f/implementation/minicells_r3mf/data.py", 59, 72),
        "loss_and_backward": (unit + "r3e/implementation/minicells_r3e/training.py", 58, 107),
        "freeze_rules": (unit + "r3e/implementation/minicells_r3e/stable_connection.py", 10, 22),
        "context_and_transition": (
            unit + "r3e/implementation/minicells_r3e/stable_connection.py",
            69,
            104,
        ),
        "model_and_tied_head": (
            unit + "r3c/moe-bridge/implementation/minicells_r3c_moe_bridge/model.py",
            45,
            60,
        ),
        "generation": (unit + "r3m-f/implementation/minicells_r3mf/learnability.py", 48, 63),
        "answer_scoring": (unit + "r3m-f/implementation/minicells_r3mf/learnability.py", 153, 198),
        "answer_encoding": (
            unit + "r3m-f/implementation/minicells_r3mf/arithmetic_tasks.py",
            7,
            46,
        ),
        "input_memory": (
            "research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/exp006/memory.py",
            9,
            16,
        ),
        "context_attention": (
            "research/projects/cellular-language-models/experiments/006-path-first-clm/implementation/exp006/cells.py",
            6,
            18,
        ),
        "cell_ffn": ("shared/python/experiments/common/models.py", 9, 16),
    }
    code = {}
    for key, (path, start, end) in refs.items():
        raw = subprocess.check_output(["git", "show", source + ":" + path])
        code[key] = dict(
            source_sha=source,
            path=path,
            start_line=start,
            end_line=end,
            sha256=hashlib.sha256(raw).hexdigest(),
            url="https://github.com/ArcheLabs/mini-cells/blob/"
            + source
            + "/"
            + path
            + "#L"
            + str(start),
        )
    diagnoses = {
        "SUPERVISION_DILUTION": dict(
            status="CONFIRMED",
            scope="OBJECTIVE_COMPOSITION_NOT_PROVEN_SUFFICIENT_FAILURE_CAUSE",
            evidence=dict(
                answer_targets=6661,
                total_targets=102400,
                answer_fraction=6661 / 102400,
                prompt_fraction=90934 / 102400,
                equal_target_weight=True,
            ),
            code_refs=["batch_packing", "tiny_dispatch", "loss_and_backward"],
        ),
        "CROSS_SAMPLE_CONTEXT_MISMATCH": dict(
            status="CONFIRMED",
            scope="TRAIN_EVAL_CONTEXT_DISTRIBUTION_DIFFERENCE; CAUSAL_FAILURE_ATTRIBUTION_SUPPORTED_NOT_PROVEN",
            evidence=dict(
                eos_to_prompt_edges=4800,
                answer_targets_with_prior_qa=4490,
                answer_context_fraction=4490 / 6661,
                segment_attention_mask=False,
                positions_reset_at_qa_boundary=False,
            ),
            code_refs=[
                "batch_packing",
                "context_and_transition",
                "context_attention",
                "generation",
            ],
        ),
        "ANSWER_BOUNDARY_ERROR": dict(
            status="UNKNOWN",
            error_observed=False,
            scope="NO_ERROR_FOUND_IN_CHECKED_ENCODING_SHIFT_OR_SCORING; DO_NOT_ATTRIBUTE_FAILURE_TO_BOUNDARY",
            evidence=dict(
                eight_prompts_checked=True,
                prompt_final_token_id=200,
                first_answer_positions=[
                    r["first_answer_token_position"] for r in p["boundary_validation"]
                ],
                all_answer_roundtrips_match=True,
                text_only_exact_match=0.0,
                strict_eos_exact_match=0.0,
            ),
            code_refs=["answer_encoding", "answer_scoring", "generation", "loss_and_backward"],
        ),
        "FROZEN_OUTPUT_REPRESENTATION_LIMIT": dict(
            status="UNKNOWN",
            frozen_constraints="CONFIRMED",
            scope="NO_RANK_LOGIT_OR_UNFREEZING_COUNTERFACTUAL_WAS_PERFORMED; FIXED_HEAD_DOES_NOT_IMPLY_FIXED_LOGITS",
            evidence=dict(
                tied_embedding_and_head_frozen=True,
                digit_output_weight_gradients=False,
                cell_to_logit_gradient_path_exists=True,
                frozen_context_read_once_before_trainable_cells=True,
                all_parameters_except_cells_frozen=True,
            ),
            code_refs=[
                "freeze_rules",
                "model_and_tied_head",
                "context_and_transition",
                "input_memory",
                "cell_ffn",
            ],
        ),
        "INSUFFICIENT_EVIDENCE": dict(
            status="CONFIRMED",
            evidence=[
                "Actual F terminal/intermediate raw data_rng was not available in the examined public reports.",
                "Reconstruction used NumPy "
                + p["replay_numpy_version"]
                + "; production was "
                + p["production_numpy_version"]
                + ". No new cross-version or endpoint certification is claimed.",
                "Packed per-token training losses/logits were not retained in the public curves; actual prompt/answer gradient allocation cannot be recovered from scalar CE.",
                "No model or new prediction was loaded/computed; missing frozen-representation and conditional-context counterfactuals remain untested.",
            ],
            code_refs=["loss_and_backward", "generation", "model_and_tied_head"],
        ),
    }
    rows = []
    for group, value in sorted(p["per_fact"].items()):
        answer = next(r["answer"] for r in p["boundary_validation"] if r["group"] == group)
        rows.append(
            f"| {group} | {answer} | {value['answer_target_tokens']} | {value['complete_answer_supervisions']} | {value['complete_answer_and_eos_supervisions']} |"
        )
    preds = [
        "| "
        + r["group"]
        + " | "
        + r["answer"]
        + " | "
        + repr(r["final_prediction"]).replace("|", "\\|")
        + " | "
        + str(r["final_eos"])
        + " |"
        for r in p["boundary_validation"]
    ]
    links = "\n".join(
        "- ["
        + key
        + "]("
        + v["url"]
        + ")：`"
        + v["path"]
        + "`，L"
        + str(v["start_line"])
        + "–"
        + str(v["end_line"])
        for key, v in code.items()
    )
    report = f"""# R3m-F Tiny Overfit 只读失败分析

## 范围与结论

对象：结果提交 `{p["result_commit"]}`，实际训练源码 `{source}`。
Run ID：`{p["identity"]["run_id"]}`。Seed：151001301，历史 Step 300 → 500，200 次已有更新。
本审计 **GPU 使用 0、模型加载 0、前向/反向 0、新增 Optimizer 更新 0**。只读取公开报告、冻结源码、原始 Tokenizer 和已有 RNG JSON；不读取私有 `.pt`、保留评价文本或 Confirmation。

原始 `TINY_OVERFIT_FAILED` 保持不变。没有新科学配置、重训、门槛放宽或完整 Pilot 授权。

**明确结论：训练目标稀释和跨问答上下文差异确实存在，但尚不能认定其中任何一项是失败的充分原因。未发现已检查的答案边界错位；冻结输出表示是否形成实际能力上限仍未知。**

## 1. 证据身份与重放可信范围

四份原始报告均按结果提交中 RUN_MANIFEST 的字节数和 SHA-256 核对，并具有同一运行身份。原始 Tokenizer SHA-256：`{p["tokenizer_sha256"]}`。
重新执行冻结生成器的纯数据函数，全部 A1 Train/Dev 编码清单哈希与实际 DATA_AUDIT 一致；从中取冻结的 8 个 Tiny 事实，而非重新选题。
原始 R3h Step 300 收据 SHA-256 `9b2d77efe3d6346fbf46ead8d8e4038af07538813c14125c70c2f36ba48d3ddf` 与已有采样边界证据匹配，完整状态指纹也匹配；使用其 PCG64 `data_rng` 状态，不通过 seed 重新初始化替代。

审计不导入 Torch。原始 `arithmetic_batches()` 函数通过 AST 从固定源码取出，仅以 CPU ndarray 替换最终 Tensor 分配/设备容器。独立标注器与原函数在 **1,600 行全部输入、标签及每 Step RNG 状态上一致**；连续 200 Step 中不额外消耗数据 RNG。

**证据限制：下列是从已匹配起点和冻结调用链推导的重建数量，不是新认证的逐步生产遥测。** 本地 NumPy 为 {p["replay_numpy_version"]}，本次生产为 {p["production_numpy_version"]}；没有读取本次 F 的终点/中间原始 RNG 状态，因此不声称端点认证通过。同一 NumPy 上的原函数对照也不等于跨版本独立认证。最终重建 RNG、采样轨迹哈希及完整 200 行 Step 计数均已交付，可供将来对照，不要求重新训练。

## 2. 实际目标构成与答案监督预算（CPU 重建）

每个 Step 8 行，每行构造至少 65 Token，保留前 65 个；`x=ids[:64]`，`y=ids[1:65]`。一共 102,400 个输入 Token，对应 102,400 个目标位置。

| 互斥目标类别 | 个数 | 占目标位置 |
|---|---:|---:|
| Prompt（包含操作数数字、固定题干和换行） | 90,934 | 88.802734% |
| Answer（仅答案数字） | 6,661 | 6.504883% |
| EOS | 4,805 | 4.692383% |
| 合计 | 102,400 | 100% |

跨问答有两种不同口径，**均为上述类别的叠加属性，不能再次加进总数**：

- 紧邻边界 `EOS → 下一题首 Prompt Token`：4,800 个目标，4.6875%；它们已经计入 Prompt。
- 所在行中可见更早问答的目标：76,450，74.658203%。答案目标中有 4,490/6,661（67.407296%）属于这种情况。
- 完整答案监督共 4,849 次，其中首个问答、没有前题上下文的 1,600 次；有前题上下文的 3,249 次。完整答案且包含其 EOS 的 4,805 次。

| 事实 | 答案 | 实际重建的答案目标数 | 完整答案监督次数 | 完整答案及 EOS 次数 |
|---|---|---:|---:|---:|
{chr(10).join(rows)}

**102,400 不能作为纯算术答案监督量。** 数字答案监督量是 6,661 个目标；在等权目标中约占 6.50%。折算只是 13.0098 个“512 目标位置块”，并不代表执行了 13 次答案专用更新，更不能据此声称只有 6.50% 的梯度范数来自答案。

### 截断语义

1,600 行共抽取 6,400 个问答实例；1,595 个实例被 65-Token 边界截断，涉及 99.6875% 的行。丢弃尾部 6,358 个 Token，**不进入输入/标签预算**，且不会传给下一行。
其中 1,531 个实例未保留任何答案目标，20 个只保留部分多 Token 答案，44 个保留完整答案却未保留 EOS；另 5 行恰在完整问答/EOS 处结束。
这是长度为约 16–18 Token 的问答硬拼接的结果，**不等于 1,595 个完整答案都被破坏**；大部分截掉的是最后一题的题干。其余完整答案次数如表。

## 3. 损失与对齐调用链

`R3m-F batches → R3m-E arithmetic_batches → R3m stable run_arm → training.advance → R3h Trainer.update → R3e Trainer.update`。

- Trainer 使用默认 `F.cross_entropy(..., y.reshape(-1))`：每个微批 64 个目标等权，8 个等长微批再除以 8。没有答案加权或答案掩码训练。
- `answer_positions` 在数据生成时登记，训练采样函数不读取它；它只在评价时选择 `losses[i-1]`。
- 拼接内第 `i` 个答案 Token 是 `y[i-1]`，由 `x` 中的前缀预测；EOS 追加于答案后，也作为普通下一 Token 标签参与损失。
- 无 Padding、无 `-100` 标签、无忽略 EOS 的 loss mask。行首 Prompt 的首 Token 只当输入，不被这一行监督；内部问答首 Token 被前一题 EOS 位置监督。
- 最后一位目标可见，但不必属于输入的 64 Token：这是正常的移位 LM 训练。硬截断会丢弃尾部标签，但未发现移位一格或答案完全不进梯度的错误。
- EOS 是训练目标/输入 Token，**不是**注意力重置指令。`ContextRead` 仅按序号执行因果遮罩，不按问答/EOS 分段；绝对位置也不在每道题重置。

已有最终 packed-batch CE 约 4.428；单题 Teacher-forced Full NLL 为 4.4955，Answer NLL 为 6.6845。它们来自不同输入/聚合口径，不能用减法恢复真实 packed Prompt/Answer CE，也不能从记录的标量 CE 回算逐目标梯度。没有模型推理，因此本轮不补造这些损失。

## 4. 生成边界和已保存预测

8 个 Prompt 最后一个 Token 均为 **ID 200，`Ċ`（换行）**。`3+4` 的 Prompt 长 14 Token，其余为 15；首答案的序列位置分别是 14/15，对应 logit 位置 13/14（均零基）。数字答案 10、14、16 均为两个 Token，已覆盖多 Token 检查。

Prompt 单独编码与完整问答编码的前缀严格一致；答案从这个前缀之后开始。完整 Train/Dev 编码哈希还与实际资产审计一致。推理传入的只有同一 `prompt_ids`，不含目标答案，且每题从位置 0 开始。

解码按 argmax 逐 Token 生成，遇 EOS=1 即停止；EOS 不进入解码文本，最多生成 8 Token。严格 EM 要求完整数字文本相等且已遇 EOS；另有 `text_only_match` 单独记录。此次两种 EM 均 0/8，因此失败不是 EOS 严格规则单独造成的。

| 指标 | 初始 | 最终 |
|---|---:|---:|
| 单题 Teacher-forced Answer NLL | 9.789227 | 6.684513 |
| 单题 Teacher-forced Full NLL | 7.778338 | 4.495508 |
| 实际自由生成首 Token 正确率 | 0/8 | 2/8 |
| 完整自回归 EM | 0/8 | 0/8 |

| 事实 | 正确答案 | 保存的最终自由生成文本 | 遇 EOS |
|---|---|---|---|
{chr(10).join(preds)}

7/8 最终预测以 `9` 开头；5/8 未在上限内生成 EOS，3/8 输出 `9` 后遇 EOS。初始常输出 `"I have`，最终更接近数字和题干，说明响应发生变化，但未学会正确映射。

**对相似后续文字的解释须分层：** 已确认输入共同具有相同题干和换行后缀，训练又大量监督题干及 EOS 后下一题；绝大部分答案训练还带有前面问答，推理则没有。因此“学到通用续写/弱问题条件化”是受支持的解释，与相似预测相容。不能证明某个 `He` 或 `What` Token 就由跨题边界直接导致，也不能在没有激活/Logit 对照时断言模型完全忽略了操作数。

Teacher-forced Answer NLL 在多 Token 答案的后续位置给定了正确的前答案 Token；完整生成则使用自己先前生成的 Token。NLL 下降、首 Token 有部分正确、完整 EM 为零可以同时成立。

## 5. 冻结参数及可用梯度路径

| 模块 | 更新状态 | 证据 |
|---|---|---|
| Token Embedding | 冻结 | 全参数先 `requires_grad_(False)`，只重开 `cells.*` |
| Position Embedding、`init` 投影 | 冻结 | 同上 |
| 输入记忆 `memory.norm/key/value` | 冻结 | 同上；输入生成只读记忆 |
| `context.query/output` | 冻结 | 同上；只在 Cell 序列前读取一次上下文 |
| `norm`、`final` LayerNorm | 冻结 | 同上 |
| Gate stage embedding、MLP | 冻结 | 同上；动作选择无梯度 |
| Connection | 冻结 | 同上及 Trainer 冻结指纹检查 |
| 六个 Cell 的 `up/down` FFN 权重 | 可训练 | 明确只重开 Cell，AdamW 仅接收 Cell 参数 |
| LM Head | 与冻结 Embedding **权重绑定** | `F.linear(self.final(h), self.embedding.weight)`，没有另一个可训练输出头 |

数字 Token 对应输出权重不会获得参数梯度或更新。**但其 Logit 并没有被冻结。**
固定路径内，`h[t+1] = h[t] + alpha * Cell_i(norm(h[t]))`，`logits = E * final_norm(h[8])`。损失可经固定的输出矩阵和可微 LayerNorm 反传到 `h[8]`，再到被调用的 Cell；权重冻结不等于对隐藏状态 detach。
局部梯度形式为 `dL/dh = J(final_norm)^T E^T (softmax(logits)-onehot(target))`。Cell 参数改变隐藏表示，仍能提高正确数字 Logit。原报告的有效梯度、参数变化和本地恢复检查通过也与此相符。

结构上还存在一项约束：唯一跨 Token ContextRead 在可训练 Cell 之前且被冻结，后面的八步是逐 Token Cell 变换，没有新增可训练的跨 Token 信息读取。它可能限制操作数信息的可利用性，但本轮不计算表示可分性、输出矩阵有效秩、目标 Logit 可达性或解冻反事实，**不能把这个约束认定为已证实能力上限**。

## 6. 分项判定

| 候选阻断 | 判定 | 准确含义 |
|---|---|---|
| `SUPERVISION_DILUTION` | **CONFIRMED** | 已确认等权目标构成稀释：答案只有 6.50%；未证明其单独足以导致 0/8 |
| `CROSS_SAMPLE_CONTEXT_MISMATCH` | **CONFIRMED** | 已确认训练与单题推理的上下文/位置分布不同；其失败因果解释为 SUPPORTED，尚非充分证明 |
| `ANSWER_BOUNDARY_ERROR` | **UNKNOWN** | 未发现已检查的编码、移位、掩码或生成边界错误；没有证据支持把失败归因于边界错误 |
| `FROZEN_OUTPUT_REPRESENTATION_LIMIT` | **UNKNOWN** | 冻结约束本身 CONFIRMED，但实际表示能力上限未测量；不能从冻结推导不可学习 |
| `INSUFFICIENT_EVIDENCE` | **CONFIRMED** | 缺本次端点 RNG 独立核对、packed 逐目标 loss/梯度和表示/上下文反事实 |

## 7. 结论与下一步边界

本轮已把“总 Token 预算”与“有效答案目标预算”区分，确认拼接/截断和不隔离上下文的语义，并排除了所检查范围内的明显答案边界错位。
没有依据宣布架构失败、解冻组件、调整 loss、追加 Tiny 训练或开放 Pilot。若后续要比较答案专用目标、逐题上下文隔离或冻结表示，应另立经授权的版本化诊断，不能覆盖当前失败记录。

当前仅有的可追加只读核验是：取得已保存的 F 中间/最终 **小型原始 RNG 摘要**，与本审计的 `final_replayed_rng` 对照；无需重新训练，也不需要本轮加载模型。

## 冻结代码位置

以下均指向实际运行源码 `{source}`，而非当前可变分支：

{links}

## 机器可读交付

- `TOKEN_DISTRIBUTION_PROOF.json`：原始证据哈希、起点绑定、逐事实计数、截断和边界结果、重放终点及认证限制。
- `STEP_TOKEN_COUNTS.json`：200 个 Step 的三类互斥目标计数。
- `RECONSTRUCTED_TINY_RECORDS.json`：八道原始编码，包含答案 Token/位置。
- `FAILURE_ANALYSIS_PROOF.json`：五项结论、冻结参数清单、源码位置/哈希及交付清单。
- `original-*.json`：已核验原报告字节的独立副本；原报告未改写。远端 RUN_MANIFEST 为发布前快照，`LOCAL_RETAINED` 不否定结果提交本身的存在；本审计从固定结果提交读取并验哈希。

复现：在拥有固定源码/结果 Git 对象和 NumPy/Tokenizers 的 CPU 环境直接执行 `audit.py`，无需密钥或 Notebook。不会自动联网取证、导入模型或启动训练。
"""
    (dest / "TINY_OVERFIT_FAILURE_ANALYSIS.md").write_text(report)
    proof = dict(
        schema="minicells.r3mf.failure-analysis-proof.v1",
        identity=p["identity"],
        result_commit=p["result_commit"],
        audit_scope="READ_ONLY_DATA_AND_SOURCE_NO_TRAINING",
        diagnoses=diagnoses,
        code_references=code,
        source_config_changed=False,
        original_reports_changed=False,
        models_loaded=0,
        optimizer_updates=0,
        gpu_used=False,
        original_status="TINY_OVERFIT_FAILED",
        full_pilot_authorized=False,
        replay_certification=p["replay_status"],
        actual_final_rng_independently_compared=False,
        files={
            q.name: dict(bytes=q.stat().st_size, sha256=hashlib.sha256(q.read_bytes()).hexdigest())
            for q in sorted(dest.iterdir())
            if q.is_file() and q.name != "FAILURE_ANALYSIS_PROOF.json"
        },
    )
    (dest / "FAILURE_ANALYSIS_PROOF.json").write_text(
        json.dumps(proof, sort_keys=True, indent=2) + "\n"
    )
