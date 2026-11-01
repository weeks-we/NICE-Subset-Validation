# 当前方法定义

当前配置见 configs/main.yaml，执行与参数完整说明见 [HANDOFF](HANDOFF.md)。2026-11-01 用户授权：2000 SWE-smith 候选；Verified 20 个 NICE 任务、每题 8 rollout；高低二等分各选 512；两个学习率、三个阶段、四份选中权重；SWE-smith 100 题选权重，独立 Verified 100 题最终测试。

NICE 和 SFT 都只对 LoRA 参数求梯度，基座冻结。候选损失为一条完整轨迹的 assistant 有效 token 平均 CE；用户、system、工具观察和 padding 不监督。验证方向 h_v=(1/8) sum_m resolved_vm * grad NLL(generated tokens | actual input)，不按轨迹长度再除；失败为零但保留八次固定分母。每个验证任务的方向独立归一化，零方向保持零，最终在全部 20 题上平均余弦。

主评分为按 SFT 全局范数裁剪后、fresh Adam 首次更新方向的余弦；原始梯度余弦作诊断。共享确定性 CountSketch 投影与参数顺序；评分不创建 optimizer.step。不以 DDP 混合不同候选梯度。保持 eps、clip 与后续 SFT 一致；两种标量学习率不改变归一化的 fresh-Adam 方向，因此共用一份 NICE 排名。

按分数升序分成 1000/1000 的低高半池，各以 seed=11 抽 512；共同监督 token 预算 ±10%。高低成员各固定一次，两种学习率复用同组成员和相同顺序。每分支 1 epoch，有效 batch=8，微 batch 实测冻结；损失先按轨迹 token 取均值，再按有效 batch 的轨迹取均值。每次有效 batch 仅裁剪和更新一次。

每 (组别, 学习率) 保存 128/256/512 曝光节点；在同一 SWE-smith 100 题上选 resolved 最大节点，平分选更早节点。各学习率下高低各选一份，共 4 份；初始模型与这四份才接触最终 Verified 100 题。不存在用最终 benchmark 调学习率或选节点。

各组最终节点可以不同；比较的是同学习率、同最大训练预算和同选权重规则下的最佳节点，而非严格相同更新次数的单因素对照。三个进度的完整验证曲线同时保留。

冻结四份选择后，initial 与四份权重还各自对原 NICE 20 题做一次 greedy 复测，共 100 次。预算和官方判分与 benchmark 相同，结果独立存放在 nice_retest，禁止据此重选权重；initial 重新评测，不拿随机 rollout 作 greedy 基线。原题已参与选样，因此仅为诊断，不能合并成 120 题独立 benchmark。总正式工作量为 2060 episode。

75 步等交互预算在 NICE、选权重和测试一致；NICE temp=1/top_p=1/top_k=0，评测 greedy。最近五个工具观察参与输入，原始完整观察落盘；NICE 重放实际 token IDs，并核对生成 logprob。零 dropout 下以 training mode 启用 HF 梯度 checkpointing，不代表更新基座；有效目标位置输出层与分块 CE 在数值测试中对照完整损失及 LoRA 梯度。

方法是单初始 checkpoint 的 NICE 风格变体。原论文涉及 warmup、多 checkpoint、已有 Adam moments、不同投影和任务，不能宣称完整复现。单 seed 不提供稳定性估计；选权重验证成绩具有适配性，最终测试用共享题目的逐题配对比较；预训练污染未知。
