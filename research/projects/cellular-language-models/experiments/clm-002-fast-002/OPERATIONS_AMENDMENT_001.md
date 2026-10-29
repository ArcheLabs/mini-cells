# 操作修订001：从 Kaggle 云端 Output 运行时恢复

本修订替代 README/Notebook 中必须手动挂载 Input 的操作要求；不修改冻结科学协议、配置SHA、数据身份、Seed、校准尺度、信任阈值或训练预算。

1. 优先核验现有本地文件与恢复缓存，完整时不需要 Kaggle 网络认证。
2. 缺失时复用 `r3b/kaggle_posthoc/kaggle_source.py` 与 CF-001 `RecoverableKaggleOutputAPI`，从既有 `bingjiangli/minicells` Notebook 的已保存版本读取 Output。
3. 支持既有认证；在 Kaggle 运行时优先尝试现有 `KAGGLE_API_TOKEN` Secret。会话已启动不等于版本化 API 已获得私有文件读取授权。GITHUB_TOKEN 仍仅用于结果回传。
4. 自动检查当前 SDK 是否支持原版本选择接口；不支持时安装前次恢复已使用的固定 Kaggle API 提交 `6894864ff4095cf98e9e36d3a694ff5fc9ba3818`。旧版缺失私有方法不会被当成兼容。
5. 默认只查最近12个已保存版本，按固定 FAST-001 Run ID 与31个精确路径（manifest+30个权重）匹配。每个候选版本只分页一次；找到全部必需路径即结束该版本枚举，不逐权重重复列举。
6. 先下载小 manifest，核对其完整 identity 与固定 GitHub 结果包一致，之后才下载权重。权重SHA来自固定历史发布提交，全部匹配后才允许模型加载及校准。不会根据文件名相似或版本相邻推断同一次运行。
7. 下载沿用 CF 的连接/读取期限、10秒进度心跳、响应复用和官方续传。不保留旧适配器外层多重退避；认证/网络失败直接停止并保留完成文件。只下载所需权重，不下载全部旧代码、日志或 Token 数组。签名URL不写入证据。
8. 找不到匹配时报告已检查范围，不无限扫描历史，也不补训。CLI 的可选 `--kaggle-version` 可精确指定已知版本；正常 Notebook 不要求用户输入。
9. `KAGGLE_RECOVERY_PROOF.json` 记录选择版本、Run 身份、文件SHA/大小、传输和缓存统计。30个Checkpoint仍须通过原本的完整状态与参数验收；此操作修订不代替科学门禁。
10. 本轮只做模拟 API 的 CPU 测试，不声称在本机完成真实 Kaggle 身份认证、云端下载或GPU校准。实际凭据和已保存 Output 的存在由首次 Kaggle 运行验证。
