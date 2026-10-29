# 方法定义

研究名称：单初始 checkpoint 的 NICE 型评分验证。2026-10-28 已核对 [ICML 2025 论文](https://proceedings.mlr.press/v267/wang25bm.html)及[作者官方源码](https://github.com/JTWang2000/NICE/tree/9516c140158299e5d5c3ba82e235ed2d69132ed0)。本项目采用此次需求约定的单初始 checkpoint 变体，不能称为原论文的完整复现。

源码对照：`obtain_gradients` 的候选 token 平均 CE、`obtain_policy_gradients` 在不做长度归一化时的奖励加权 CE 总和，与本项目的梯度方向一致；作者 `calculate_influence_score` 对归一化向量做矩阵乘法，本项目对任务余弦取均值。CPU 上实际执行这三个作者函数并作数值对照已通过，外部 tokenizer/reward 使用明确的测试替身。作者原设置使用 warmup、多 checkpoint、已有 Adam moments 和 TRAK Rademacher 投影；本项目固定初始 checkpoint、fresh AdamW、CountSketch、30 个 SWE-smith 多轮任务及每题四次 rollout。作者 Adam 辅助函数的 epsilon 位置及偏置修正规则也与本项目不同，不能直接把两种 Adam 特征视为同一实现。本项目 Adam 方向按实际首次 AdamW 更新验证，不为追随原仓库而修改既定研究定义。

2026-10-27 按用户确认更新：主 validation 为 SWE-smith 任务源中与训练候选分开的 30 题；Verified 不再作为 NICE 评分源。另固定 Verified、Pro Public V2、Live Python full 快照、Terminal-Bench 2.0 各 50 题，作为初始模型与全部 12 个分支的独立迁移评测。具体来源、隔离与 native harness 见 TRANSFER.md。

基座为固定 revision 的 `Qwen/Qwen3.8-27B`。实际架构、官方加载类、库版本、文本注意力模块和非 thinking 设置，须在服务器核对。没有预热状态，只使用共同初始 checkpoint 的 LoRA 可训练参数空间。基座冻结，LoRA rank=16、alpha=32、dropout=0。评分时关闭 dropout，不创建优化器更新。

候选和主验证两端的梯度都只取同一组LoRA A/B参数，不是全量基座梯度。候选端为assistant token均值CE，验证端为奖励加权生成token NLL总和；后者用固定四次rollout分母。实际官方小模型测试检查两端反传后基座梯度均为None、参数均未被评分修改。12个正式SFT分支同样只更新LoRA。完整加载基座用于前向和反向传播，不等于优化全部基座参数。

## 候选梯度与验证方向

每条完整训练轨迹的 loss，是全部有效 assistant 目标 token 的平均交叉熵：

```text
g_i = grad L_i(theta0)
L_i = sum(valid assistant token CE) / number of valid assistant target tokens
```

用户、system、工具观察和 padding 不监督。`assistant-rendered-delta-v3-tool-groups` 用真实 chat template 的逐轮 prefix 差异定位 assistant body、工具调用序列化和结束标记，排除 assistant header；连续工具观察按完整分组处理。已知的空 non-thinking scaffold 属于模板 header。真正的推理 token 不会被删除。模板若会改写此前消息、无法定位稳定区间，清洗直接失败，要求核对目标模板。跨边界 token 不监督。不截断超长轨迹。当前后端使用完整轨迹，不做片段平均或 packing。

每个 validation 任务生成四条完整交互轨迹。每步保存实际 prompt IDs 和 generated IDs，因此梯度回放不会把工具观察误作输出：

```text
log pi(tau | v) = sum(all model generated token log probabilities)
h_v = -(1/4) * sum_m resolved_vm * grad log pi(tau_vm | v)
```

`resolved` 为官方判分的 0/1。梯度端实现等价的正 NLL 梯度，不除以轨迹长度。不能删掉错误动作。零奖励轨迹仍保留在四次 rollout 的固定分母。基础设施失败不是奖励 0；不完整 rollout 阻止评分。

NICE 采样使用 temperature=1、top_p=1、top_k=0，无额外语法约束、重复惩罚或 forced token。模型 EOS 定义停止条件。保存实际 `GenerationConfig`。本地推理服务使用相同的官方 HF 模型和 LoRA，返回精确 token IDs、原始输出和 logprob sum。梯度端逐步核对模型/adapter/tokenizer/模板/精度/库版本身份，并检查 logprob 差异。初始容差为每个输出 token 0.005 nats，记录总差异及总容差；这不是对生成器无偏性的独立证明。真实 smoke 必须先检查该对齐。

## 主分数与投影

在原 LoRA 参数空间中，先按 SFT 的全局 grad clipping 规则裁剪候选梯度。主方向和诊断方向分别为：

```text
u_i(fresh Adam) = clipped_g_i / (abs(clipped_g_i) + 1e-8)
u_i(raw) = g_i
s_i = (1/30) * sum_v cosine(u_i, h_v)
```

fresh Adam 来自共同的 m0=v0=0，weight_decay=0；正学习率标量不改变余弦。该方向近似符号向量，不能代表多步 Adam 路径。主分组只使用 `fresh_adam_cosine`，`raw_grad_cosine` 只作诊断。

范数不超过 `1e-12` 的向量记作零。零向量对应的余弦为 0，分母仍为 30。若全部 validation 方向为零，评分和训练停止；覆盖低于 20% 时标记 `LOW_REWARD_COVERAGE`。高效实现使用 `q = mean_v normalize(h_v)`，然后 `normalize(u_i) dot q`，不再归一化 q。

使用 `countsketch-v1` 流式投影，默认 8192 维，seed=42。每个原参数坐标用固定 uint64 hash 映射到一个桶和一个符号。参数按完整名称排序，双方共享偏移和 hash。先构造 fresh Adam 方向，再投影；投影后的余弦是近似估计。CPU 测试核对流式块边界、精确聚合和误差，服务器 smoke 对真实 LoRA 梯度保存多个投影种子的误差。投影误差可能改变近邻排名，必须随结果报告。

## 分组与训练

300 条候选按主分数、data_id 升序排序，低/中/高各 100 条；全分数相同则阻止分层。记录分界分数、间隔及并列数量。每段和全池各按 11、22、33 抽 30 条。

共同监督 token 目标 B，来自全池 seed=2026 的 1000 个随机 30 条预算探针的中位数。正式组从其来源均匀、无放回抽样，选择确定性重试序列中首个处于 B±10% 且不与已选组完全相同的候选，最多 10000 次。不按均分优化成员。抽样分布是**条件于 token 预算**的分布。跨组允许重叠，记录交集和 Jaccard，不强制仓库比例相同。

每个分支重置初始 adapter、优化器和 RNG，使用固定 seed=2026 的数据顺序。一遍、30 次更新、常数 lr=1e-5、AdamW betas=(0.9,0.999)、eps=1e-8、weight_decay=0、max_grad_norm=1。保存每个完整步骤的 adapter 参数、优化器、数据顺序、Python/NumPy/Torch/CUDA RNG 和校验值。中断重算的训练意图和原始成本另存，不隐去重复计算。

## 评测与解释

正式基线独立于 NICE 的 120 次 rollout。初始及 12 个训练后模型各运行同一 30 题一次，seed=2026、贪心生成、共同工具/上下文/时间预算。预算耗尽和无效动作是任务失败。环境/服务/判分故障保留 INFRA_ERROR，只有这类错误可在干净环境重试。

`R = resolved_count / 30`，`Delta_k = R(M_k) - R(M0)`。缺题、重复题或基础设施错误使正式成功率为 null，另报已完成数量。子集预测值是成员主分数均值。

报告平均秩 Spearman 和两两一致率，单列可比较对数、并列对数。常数序列的相关性为 null。不默认做显著性检验或 bootstrap。12 个子集有重叠且共享 30 道主题，不能视为独立重复。三个抽样种子不代表训练稳定性。主集单题为 3.33 个百分点，这部分同题关系不支持未见任务泛化结论。另将四个外部 benchmark 各 50 题作为持出迁移证据，每题为 2 个百分点；分别报告初始与 12 个分支的配对提升及描述性相关，不混合不同任务分布。总计 3110 次 Agent episode，12 份正式 LoRA 与 360 次训练曝光不变。
