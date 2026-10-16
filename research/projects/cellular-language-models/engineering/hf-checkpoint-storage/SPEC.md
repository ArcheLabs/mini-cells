# MiniCells 实验产物双通道发布与跨会话恢复实施方案

**状态：实施提案（尚未修改代码、尚未验证 Kaggle→HF 的真实上传/恢复）**  
**适用范围：R3 工程能力延续及下一轮 R3b / 后续实验**  
**原则：Hugging Face 保管大文件及可恢复状态；GitHub 接收轻量研究证据；Kaggle 只承担计算。**

## 1. 调查结论：旧机制已经存在，但当前 R3 未接入

在 `ArcheLabs/mini-cells` 的源码中可以确认，以前的 Native CLM M2/M3 实验已经建立了这一分工：

- `research/projects/cellular-language-models/context/06-native-clm/scripts/research/publish_native_clm_v0_m2.py`：将各 Seed / Arm 的 `final.pt` 上传到 Hugging Face Model Repo，计算 SHA256、记录字节大小与 HF revision；`.json`、`.csv`、`.md` 等轻量结果通过 Git 发布，不提交 `.pt`。支持 `--require-hf-upload`。
- `.../publish_native_clm_v0_m3.py`：延续同一方案。
- `.../fetch_native_clm_v0_m1_checkpoint.py`：通过 `hf_hub_download` 下载指定仓库与版本，核对预期 SHA256，记录 provenance。
- `.../fetch_native_clm_v0_m3w0_checkpoints.py`：从研究协议读取 `hf_repo`、`hf_revision`、文件路径与 SHA256，验证后复用历史状态。
- `research/projects/cellular-language-models/experiments/native-clm-v0-m2-continual-language/notebook.ipynb`：在 GPU 训练前执行 HF 写权限预检，正式实验结束后执行 HF 上传与 Git 发布。
- `research/projects/cellular-language-models/experiments/shadow-cell-validation-001-copy-on-write-functional-isolation/artifacts/model-artifacts.json`：有包含 `hf_upload_status=PUBLISHED`、HF revision、具体 `.pt` SHA256/大小的历史发布记录。

历史仓库标识曾有 `archelabsxyz/native-clm-v0`、`archelabsxyz/shadow-cell-validation-001`；另有发布记录使用 `archelabs-org/native-clm-v0`。**这证明源码及历史元数据中有已实现的模式，不等于我们现在已通过带权限账户验证这些仓库仍可写。**

当前 R3 则采用 `r3/implementation/clm001_r3/publication.py` 的 GitHub 元数据白名单发布，明确把 Checkpoint 留在 Kaggle Output；`r3/implementation/clm001_r3/checkpoint.py` 已具备完整恢复、身份验证和 portable 导出；`r3/implementation/clm001_r3/automatic.py` / `runner.py` 还未在每个提交点调用 HF 远程持久化。

**结论：复用既有 HF 出入库思想、验证模块与现有 R3 Checkpoint，不重写训练器，也不依赖 Kaggle Output 的历史版本下载。**

## 2. 目标与非目标

### 必须达到

1. Kaggle 每形成**已提交的本地 Checkpoint**，能将完整恢复状态持久上传 Hugging Face。
2. 新 Kaggle Session 直接经 HF API 下载指定 Experiment / Run / Arm / Step，校验身份和 SHA256，恢复 Optimizer、RNG、数据游标、Registry、Shadow Candidate 和 Event 游标。
3. GitHub 保留现有结果分支及汇总指标发布机制；新增一个小体积的 `checkpoint-index.json`，以 HF repo、**精确 revision**、相对文件路径、SHA256、Step 和实验身份链接大文件。
4. 可执行 `train / resume / evaluate / fork` 四种不同操作；`resume` 不允许静默改成从头训练，`evaluate` 不得更改模型。
5. HF 临时不可达时，不得虚报成功持久化；可以向 GitHub发布可靠的失败/部分结果证据；不得将未持久化的运行标记为 `DURABLE_COMPLETE`。

### 本次不做

- 不修改 MiniCells 的 Q、LCG、STOP、CTX、Cell 分化训练算法；不修改 R3 的冻结科研数据及已发布结果。
- 不为迁移旧 R3 强行下载 Kaggle 产物，不把未证实存在的旧 Checkpoint 当成已恢复。
- 不引入数据库、Google Cloud、独立 GPU 服务器或复杂调度平台。
- 不把全部原始训练 Oracle 标签、私有 Probe 内容或访问令牌推送到 GitHub。

## 3. 系统架构

```text
GitHub 源码 + 冻结协议 / Config
              │ 固定 SHA
              ▼
        Kaggle GPU Worker
              │
      保存本地 Checkpoint
              │
        完整性与身份检查
              │
              ▼
   Hugging Face Private Repo
   ├─ Checkpoint (.pt)
   ├─ Checkpoint SHA256 sidecar
   ├─ Immutable asset / probe bundle
   └─ Per-arm committed pointer
              │
              ├─ 提供下一会话 Resume / Evaluate / Fork
              │
              ▼
    远端提交结果 (HF revision)
              │
              ▼
 GitHub 结果分支（现有发布器）
   ├─ 指标和对照研究报告
   ├─ checkpoint-index.json（小文件）
   ├─ storage-status.json（小文件）
   └─ 失败或部分完成证据
```

GitHub 的引用只描述**可验证的已上传产物**，不得提前发布尚未完成的 HF URL。

## 4. Hugging Face 仓库选择

建议新建**独立的私有 Model Repository**，示例名 `archelabs-org/minicells-research-checkpoints`（须核实组织权限与可用名称）。不要直接向以前发布 Native CLM 模型的仓库灌入新的实验 Checkpoint，以免旧版本和新实验混杂、公开与私有权限不一致。

现有历史 `archelabsxyz/native-clm-v0` 继续保持原样，仅作为旧工具链设计及历史研究来源。不擅自迁移、覆盖或修改其可见性。

HF Hub 的 Model Repo 使用 Git revision 和 Xet/LFS 管理二进制文件，适合我们初期几十个至数百个可追溯 Checkpoint 的实验。未来如果极高频地保存大量临时状态，可评估 Hugging Face Storage Buckets；Buckets 是非版本化对象存储，切换需要另行定义版本和索引原语，不作为本次恢复旧机制的前置条件。

## 5. 数据布局和版本身份

```text
<HF_PRIVATE_MODEL_REPO>/
  clm-001-r3b/
    <run-id>/
      immutable/
        runtime-config.json
        experiment.lock.json
        assets/manifest.json
        assets/tokenizer.json
        assets/train.npy
        assets/calibration.npy
        assets/admission.npy
        assets/development.npy
      arms/
        Q-R3b/
          step-000050/
            checkpoint.pt
            checkpoint.sha256.json
            checkpoint-manifest.json
          step-000100/...
          latest.json
        QC-R3b/...
        QS-R3b/...
      probes/
        Q-R3b/frozen-probe-bank.pt
        Q-R3b/frozen-probe-bank.json
        QC-R3b/...
      progress/
        <arm>-step-<N>.json
```

注：目录为规范目标结构，并非现有 R3 的已发布 HF 结构。真实的 Runner 可通过 **mapping adapter** 将已有 `checkpoint-step-000400-event-000005.pt` 映射至上述不可变路径。**同一个 Step 可能有不同 Event Revision；必须将 `(step, event_revision, event_cursor)` 一并纳入不可变对象 ID，不能只按 Step 去重。**

`run-id` 应冻结 Experiment ID、Seed、Source SHA、Runtime Config SHA256、Protocol SHA256 与 Assets Manifest SHA256；不同训练身份不得共用可写目录。

如果资产已在另一个受控且内容不可变的远端仓库，可以只存资产 Manifest + 固定 revision + 哈希引用；前提是必须通过完整恢复测试确认可获取。Frozen Probe Bank 在模型跨阶段比较中应保留原始字节，不得重建为“等价”探针。

## 6. 发布协议：两阶段提交，远端持久化之后再记账

**阶段 A：本地一致性提交**

1. 继续使用既有 R3 `checkpoint.save()`：先完成本地 `.pt` 和 `.sha256.json`，再以 `latest.json` 为本地提交点。
2. 不改变既有 Event 语义：`events.process()` 对每个生命周期动作都写 Event Revision Checkpoint；这是防止 Shadow Candidate / Promotion 中断后重复执行的关键。
3. 只把本地已提交、标识完整的 Checkpoint 放入上传队列。队列需保护单个文件不在上传时被修改；不得引用 `.orphan` 临时文件。

**阶段 B：远端发布与验证**

4. 将大文件与 sidecar、Checkpoint Manifest 上传至 HF 独立路径；每个 Step/Event Revision 都是不可变的。
5. 上传成功后获取目标 HF commit OID，验证目标路径在该 revision 下存在；抽样/首次真实演练进行远端下载后 SHA256 复核，所有 Resume 时都做完整 SHA256 验证。可以将完整远端回读作为严格门禁，但需计入网络与时间成本。
6. 同步更新该 Arm 的 `latest.json`，其中只允许指向**已提交的** Checkpoint。可以将 Pointer 与 Checkpoint 在单次 `create_commit` 中提交以减少中间不一致窗口，或按先对象、后 Pointer 的顺序分次提交。更新 Pointer 前严格检查 Step/Event Cursor 单调前进。
7. 多 Worker 同仓库写入须防竞态：每个 Arm 独立路径，使用仓库提交前置 SHA（`parent_commit`）做乐观并发控制并在冲突时重新读取后重试，或者设单一远端发布队列；**绝不允许旧 Worker 将较新的 latest 回退**。
8. 远端成功的 HF commit OID、路径、内容 SHA256、bytes、run-id 才进入 GitHub 的 `checkpoint-index.json`。

失败策略：上传失败可限次退避重试，且应保留本地文件。任何需要“已持久化才能继续”的长跑模式，不可在存储持续失败时无限继续训练；在安全边界停止该 Segment 并发布 `STORAGE_BLOCKED`，而不是把运行标为完成。

## 7. 恢复协议及四种运行模式

- **Train**：新实验；明确 `--resume none`，必须证明该 run-id 没有已经提交的可恢复历史，除非用户显式选择创建新 Run。
- **Resume**：指定 run-id + `arm` + `step` 或 `latest`。通过仓库、revision、路径、哈希取回 `.pt`，加载完整训练状态；Config/Source/Tokenizer/Assets/Protocol 和 Event 语义必须兼容，必须继续同一科学实验。不允许找不到数据时自动从零开始。
- **Evaluate**：指定不可变 HF revision + Step/Event；读取模型并设置只读评估，不执行优化器步进，不修改原始科学证据；结果使用独立评估协议版本登记 GitHub。
- **Fork**：以历史模型创建新 `experiment_id` 或 `run_id`，记录 `parent_checkpoint` 的完整身份；改变训练配置/算法必须被识别为新研究，不得伪装原实验 Resume。若不能兼容复用旧 Optimizer，则需明确重建与科学含义。

建议 CLI 作为统一入口（以下为期望接口，**不是现有 R3 已提供的选项**）：

```bash
python run.py --mode storage-preflight --hf-repo "$HF_REPO"
python run.py --mode train-dev --arm QC-R3b --storage hf --segment-end-step 200
python run.py --mode resume --arm QC-R3b --checkpoint-ref "$REF" --segment-end-step 350
python run.py --mode evaluate-checkpoint --arm QC-R3b --checkpoint-ref "$REF"
python run.py --mode fork --parent-checkpoint-ref "$REF" --new-experiment CLM-001-R3c
```

`segment-end-step` 是执行控制参数，不能修改冻结研究配置中的 `training.steps`，否则 Checkpoint 的 Config SHA 会失配。

## 8. GitHub 元数据契约

沿用 R3 `results/clm001-r3-...` 结果分支 / 白名单发布方式，增加新实验对应的白名单文件，并保持单文件大小限制；不可直接 `git add` `.pt`。

最小 `checkpoint-index.json` 示例：

```json
{
  "schema": "minicells.checkpoint-index.v1",
  "experiment_id": "CLM-001-R3b",
  "run_id": "<frozen-run-id>",
  "source_sha": "<40-hex>",
  "status": "DURABLE_PARTIAL",
  "checkpoints": [
    {
      "arm": "QC-R3b",
      "step": 400,
      "event_revision": 5,
      "event_cursor": "<committed-cursor>",
      "hf_repo": "archelabs-org/minicells-research-checkpoints",
      "hf_repo_type": "model",
      "hf_revision": "<full-hf-commit-sha>",
      "hf_path": "clm-001-r3b/<run-id>/arms/QC-R3b/step-000400-event-000005/checkpoint.pt",
      "sha256": "<checkpoint-sha256>",
      "bytes": 12345678,
      "runtime_config_sha256": "<sha256>",
      "assets_manifest_sha256": "<sha256>",
      "protocol_sha256": "<sha256>"
    }
  ]
}
```

若私有 HF 仓库未公开，GitHub 的链接只是**可追溯指针**，公众无法直接下载。不要在 GitHub 公开私有 Train/Probe 内容或敏感运行参数。建议记录状态：`NOT_ATTEMPTED`、`UPLOAD_IN_PROGRESS`、`DURABLE_PARTIAL`、`DURABLE_COMPLETE`、`STORAGE_BLOCKED`、`VERIFICATION_FAILED`，并将存储状态与科研结论分为不同字段。

特别保留以前 M2 Publisher 的可取之处：**HF 上传失败不能抹去已完成的合法研究数据**；但对于希望通过 HF 实现下次恢复的 R3b，完整跑完而未上传成功时不能宣称恢复能力已经建立。

## 9. Secrets 与权限边界

- 在 Kaggle Secrets 中使用 `HF_TOKEN` 和 `GITHUB_TOKEN`；GPU Worker 不打印、上传或持久化这两个值。
- HF token 必须具有**目标私有仓库内容写权限**，最好使用仓库粒度的细分权限；`GITHUB_TOKEN` 只需结果分支写权限。
- Preflight 在占用正式 GPU 预算前验证：HF 仓库真实存在、读写可用、GitHub 发布预检可用；不能用“能下载公开 TinyStories”替代“能写私有 HF Checkpoint”。
- PyTorch `.pt` / pickle 加载只来自明确授权且 SHA256 匹配的可信实验产物；禁止把任意公开仓库的 `.pt` 作为可信训练状态直接执行反序列化。
- 在日志和 GitHub 记录发生的权限错误类别、状态码及仓库名称，不回显 token、签名 URL 或秘密。

## 10. 实现定位与最小代码变更

1. **复用历史实现**：参考 `publish_native_clm_v0_m2.py` / `publish_native_clm_v0_m3.py`（`HfApi`、`auth_check`、`--require-hf-upload`）、`fetch_native_clm_v0_m1_checkpoint.py`（精准 revision 与 SHA256 验证）、`fetch_native_clm_v0_m3w0_checkpoints.py`（受协议锁定的恢复）。
2. **新增通用模块**：例如 `shared/python/minicells_artifacts/hf_storage.py`，支持 `preflight`、`publish_committed_checkpoint`、`fetch_verified_checkpoint`、`list_verified_checkpoints`、`promote_latest`、`publish_immutable_assets`、`generate_index`。避免将存储代码耦合到某一套模型结构。
3. **修改 Runner**：`r3/.../runner.py` 保存和生命周期事件提交后触发远端持久化；兼容原来的本地 `latest.json`。支持 `--resume-source hf --checkpoint-ref`、训练边界及只读评估。
4. **修改 Automatic**：`r3/.../automatic.py` 的 HF 写权限预检应早于正式训练。保留 GPU Pilot、beta 冻结和研究门禁；Pilot/Assets 可缓存并以身份锁定，防止每次 Session 重复准备。
5. **修改 Publisher**：`r3/.../publication.py` 继续保留小文件白名单；新增 `checkpoint-index.json`、`storage-status.json` 及状态合并；不得重新发布 Checkpoint 二进制。
6. **Notebook**：只需要 Kaggle Secrets、固定 GitHub Source SHA、运行参数和 HF Repo ID。Notebook 不使用 `Add Input → Notebook` 或 `kaggle kernels output` 进行恢复。
7. **向后兼容**：R3 原始版本继续可重复计算结果，不因为新 Storage 默认行为改变已冻结旧实验；将新机制应用到单独版本或有明确授权的下轮实验。

## 11. 工程验收门禁（必须真实执行）

**Gate A：CPU / 模拟存储测试**
- 序列化、SHA256、run-id、Config/Source/Protocol/Data/Tokenizer 身份检查；损坏/缺失/错误权限拒绝加载。
- Step 100 事件过程中断与恢复；确保 Event Cursor 与 Shadow Candidate 一致，没有重复 Promotion；多个事件修订版不会相互覆盖。
- `latest` 幂等、单调；两个上传者并发，旧 Step 无法覆盖较新 Step；同名不同哈希拒绝静默覆盖。

**Gate B：HF 真实 I/O 烟雾测试（不消耗正式训练配额）**
- 使用私有测试仓库，在 Kaggle 或安全 CI 环境上传一个小的完整 `.pt`、sidecar、manifest。
- 在**全新进程/会话**中通过固定 HF revision 下载并验证 SHA256、身份与模型参数；确认无 Kaggle Output 依赖。
- 使 HF token 无写权限或断网，验证正确返回 `STORAGE_BLOCKED`、且 GitHub 状态不被误标为完成。

**Gate C：分段训练对照**
- 连续训练 10 Step vs 训练 6 Step → HF 持久化 → 新会话恢复 → 训练至 10 Step；在相同环境的确定性配置下对比模型、优化器、RNG、数据游标、事件状态的逐字段一致性。
- 如果 GPU 算子非确定，先给出 CPU 严格一致证明，再明确 GPU 可达成的容差，不能用“最终能跑”冒充逐位相等。
- Evaluate 历史 Step，不允许发生梯度更新、原始 Checkpoint 覆盖或科学数据重新定义。

**Gate D：真实双通道回传**
- HF 保存实验 Checkpoint；GitHub 只保存实际存在且验证过的 HF SHA+revision+路径及轻量结果。
- 使用干净 Kaggle Session，从 GitHub 的 Checkpoint Index 找到 HF 产物并实际恢复。
- 检查 GitHub git diff 无 `.pt`、Secrets、原始 Oracle 私有数据；Notebook 记录真实 HF 测试证据。

**禁止仅以 `CPU_PASS`、已生成目录或 Hugging Face SDK 调用返回成功推断全链路验收通过。**

## 12. 提交与交付顺序

1. **P0-A：旧机制复核与通用 HF 存储适配器**——提交旧源码复用映射表及模拟测试。
2. **P0-B：Checkpoint 远端提交/恢复、身份与 Event 语义**——完成自动上传/下载并支持历史选择，双 Worker 并发测试。
3. **P0-C：Kaggle 最小真实往返**——新会话恢复、6→10 与连续 10 Step 对比；只有这个 Gate PASS 才允许正式 R3b。
4. **P1：GitHub index + 独立评估 + Fork**——两个后端状态分离、产物可追溯、历史评估不重新训练。
5. **P1：R3b 纵向诊断**——在工程通过后，单独评审/冻结新的科学实验协议，开启结构化 Snapshots。

交付材料应包含：Commit SHA / PR、实际创建或使用的 HF repo ID（禁止含 token）、HF 演练 revision、示例 Checkpoint Index、Kaggle 验证版本及日志、恢复一致性测试结果、尚未解决的风险。

## 13. 决策与风险

- **仓库权限**：历史的 `archelabsxyz/...` 和 `archelabs-org/...` 都出现在源码/记录中，真实新目标应通过当前账号权限确认；不能假设两个命名空间通用。
- **时间预算**：每 50 Step 上传会消耗 Kaggle 网络时间。Pilot 先测单份 `.pt` 大小和上传耗时，再决定上传间隔与压缩策略；不要依赖未实际测量的吞吐量。
- **配额**：HF 私有免费额度及超额政策应按当前官方文档确认；不要假定 100 GB 足够无限期保留多轮实验。
- **历史证据**：暂无法读取的 R3 Kaggle Output 不等于已经迁移 HF；除非找回后实际上传且验证，否则 GitHub 的 R3 Checkpoint Recovery 一律标记 `UNAVAILABLE/NOT_VERIFIED`。
- **发布隔离**：科学结论是否成立和模型是否已可靠持久化是两个维度，两个状态都要公开记录，但不互相篡改。

## 14. 给 Codex 的执行指令

> 请在 `ArcheLabs/mini-cells` 中复用既有 Native CLM M2/M3 Hugging Face 模型文件上传/下载实现，为下一轮 MiniCells 实验恢复“HF 保存可恢复 Checkpoint + GitHub 保存轻量研究证据”的双通道模式。优先审阅并复用过去的 `publish_native_clm_v0_m2.py`、`publish_native_clm_v0_m3.py`、`fetch_native_clm_v0_m1_checkpoint.py`、`fetch_native_clm_v0_m3w0_checkpoints.py`；保留当前 R3 `checkpoint.py` / `events.py` 的原子提交、身份校验和 Event Cursor 语义，不改变科学算法，不重新训练 R3。先完成本地/模拟测试，然后使用独立私有 HF 仓库完成真实上传、跨 Kaggle Session 下载与 6→10 Step 等价恢复实验。大型 `.pt` 与固定探针/资产存 HF，GitHub 回传 `checkpoint-index.json`、HF 精确 revision、SHA256 与指标报告，绝不提交模型二进制。恢复失败不得从零训练，远端发布失败不得声称可恢复完成。分阶段提交，报告源码 SHA、HF 验证 revision、GitHub Commit SHA、真实 Kaggle 验收证据；在门禁通过前不得启动 R3b 正式实验。

## 15. 参考位置

- GitHub 既有 HF M2 Publisher：`research/projects/cellular-language-models/context/06-native-clm/scripts/research/publish_native_clm_v0_m2.py`
- GitHub 既有 HF M3 Publisher：`research/projects/cellular-language-models/context/06-native-clm/scripts/research/publish_native_clm_v0_m3.py`
- GitHub 既有验证式下载：`research/projects/cellular-language-models/context/06-native-clm/scripts/research/fetch_native_clm_v0_m1_checkpoint.py`
- GitHub 已冻结 revision 下载：`research/projects/cellular-language-models/context/06-native-clm/scripts/research/fetch_native_clm_v0_m3w0_checkpoints.py`
- GitHub R3 新出版器：`research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3/implementation/clm001_r3/publication.py`
- GitHub R3 Checkpoint / Event：同目录 `checkpoint.py` / `events.py`
- HF 上传文档：https://huggingface.co/docs/huggingface_hub/guides/upload
- HF 仓库和私有存储：https://huggingface.co/docs/hub/repositories ；https://huggingface.co/docs/hub/storage-limits
- HF Storage Buckets（未来备选）：https://huggingface.co/docs/hub/storage-buckets
