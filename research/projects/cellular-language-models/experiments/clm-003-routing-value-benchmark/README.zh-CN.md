# CLM003 阶段 0

本阶段只交付协议与 CPU 机制工程，不是 A/B/C/D/E 的完整训练实现。

已实现：使用真实原生 CLM 类的合成 CPU 夹具；A/B/C 共享合法随机 Connection；保持 Cell 与公共参数初值；隔离初始化随机流；C 事件仅匹配 A 的实际提交；配对训练 Seed 作为分析单位；JSON Manifest、元数据白名单和现有 GitHub 回传组件；独立 CPU Notebook。

未实现或未冻结：真实父 Checkpoint 与优化器/游标恢复、训练 Seed 和预算、实际 A 更新算法版本、独立数据角色身份、统计门槛与多重比较、MoE/Dense 实体模型与匹配容差、非劣界限及硬件测量方案。`config.json` 保留十项明确阻断，不擅自填写科学参数。随机重连幅度未匹配，不声称已排除幅度混淆。

既有 FAST002 的终点评价是历史 Confirm，不能静默改称独立 Development。新 Confirmation 未建立，因此不能声称确认性实验就绪。父模型已适应历史路由的事实必须继续披露。

Kaggle 导入 `notebooks/kaggle-phase0-cpu.ipynb`，选择 CPU、Internet 并勾选已有 GITHUB_TOKEN，Run All 即完成合成预检及 GitHub 回传。无需 HF_TOKEN、KAGGLE_API_TOKEN 或历史 Output。权重不进入 Git；`--publish-existing` 只重试结果回传；`--run-science` 明确拒绝。

本次实施不使用 GPU、不执行 Optimizer 更新、不读取模型质量、不改写历史实验。不对 MoE/Dense 的性能或架构可行性作结论。
