# 最小本地验证

Python3.11.18、PyTorch2.6.0+cpu、einops0.8.2。未启动真实GPU开发训练、正式训练、CPU性能Benchmark或完整历史测试矩阵。

三个必要回归检查通过（最新核心修改后：3 passed，12.65秒）：

1. 数据Oracle、答案不回填、split隔离、拒绝formal-test访问；篡改HOLE回填答案会失败。
2. 微型合成fixture完成两个真实AdamW更新，参数改变且目标计数精确；完成Checkpoint恢复不再调用Optimizer.step；错误source/protocol身份拒绝恢复。
3. 真实Script Version ID绑定规则：当前ID缺失、与保存版本不符、源码协议标记不符均停止，不允许用最近版本补位。

完整开发数据生成/校验通过，4096训练Episode×8=32768查询Targets；384开发验证Episode，未生成或打开正式测试。

本地数据指纹：
- train ids: aedc9350a19ddca01f4df1ddde48ac33dc0b538bd9200f3a3c934dfaf0685cc3
- train labels: 4e93fe27fb21ef773ea3c9fd99bd6c50d19fdb8f7731c2879fbdda39ecae1c8d
- validation ids: c7dfb74ced08700d1d28bc3da22a90cba2a9df751e0a2408d75de31df9213d3b
- validation labels: d7aa43b18ecf9fd32194d95aca873b7142110d1cf2757e68ae699dbc8ddc9a83

Python/Notebook语法、协议标记、预算算术、JSON与Git格式检查通过。复用恢复设施时发现其导入另一个通用model模块；本入口以明确文件路径绑定既有M3C-002 Model，避免模块名碰撞，不修改历史恢复实现。

真实Kaggle当前Script Version ID获取、GPU耗时/显存/学习曲线与结果发布尚未在本机实测。公开SDK的ApiKernelMetadata并未保证提供ScriptVersionId，不能把current_version_number或kernel id替代。原4113ec9入口依赖运行端实际提供并可核对的ID，取不到则停止；该旧行为已由下文用户明确授权的开发延后策略取代。平台获取仍未在本地实测。

## 运行后修复：身份缺失时发布

真实运行在训练前因平台未暴露Script Version ID停止；随后publish对execution=None调用get导致异常。已修复nullable execution处理，增加一项定向回归验证：缺失身份的失败包仍能进入已有发布流程，metadata的ID保持null，不伪造ID。定向检查1 passed，3 deselected（6.01秒）；未重跑完整训练或测试矩阵。实际GitHub凭据/网络由运行时验证，fixture仅验证失败分支不会再发生AttributeError。

## 用户授权后的开发身份延后

仅本次开发校准允许ID_PENDING继续。缺失ID仍记null，协议标记不匹配与身份冲突不放行；正式训练/测试未授权。增加定向检查：实际缺字段分支在开发授权下返回pending、未授权/标记不符时拒绝；补录URL拒绝错误Notebook/版本序号并保留用户来源标签；已完成且ID待补录的运行仅重发结果，不进入训练。未执行额外GPU或完整本地训练。

本次定向回归结果：4 passed，3 deselected（6.86秒），未重跑原优化器训练fixture或完整测试矩阵。兼容上一版Notebook启动标记仅用于此次身份策略修订；已核对除身份字段外模型、数据、Seed与预算完全相同，当前协议hash仍单独记录。

补充冲突边界：两个定向检查再次通过（2 passed，5 deselected，1.28秒），覆盖补录ID不得与已记录真实ID冲突，以及不同源码不能覆盖已完成运行的原decision。
