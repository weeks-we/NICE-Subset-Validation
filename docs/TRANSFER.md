# 当前 benchmark 范围

2026-11-01 用户将正式流程改为只运行 SWE-bench Verified：20 题用于 NICE，另外互斥的 100 题作为最终 benchmark。初始模型和各 (组别, 学习率) 内选中的四个 checkpoint 共享这 100 题。

旧 Verified/Pro/Live/Terminal 各 50 题的四套迁移实验已经关闭，旧实现仅供历史回归，不应运行。当前选择节点使用独立 SWE-smith 100 题；不能把 benchmark 放进节点选择。执行见 [HANDOFF](HANDOFF.md)。

冻结权重后还对原 NICE 20 题执行 initial+四份选中权重的单次 greedy 复测，单独记录为 nice_retest，共 100 episode，仅用于诊断。这 20 题不计入独立 benchmark，也不用于重选权重。
