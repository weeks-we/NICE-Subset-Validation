# 项目协作规则

本文件是本仓库唯一权威运行规则。`CLAUDE.md` 指向本文件。

1. 先阅读 `README.md`、`docs/METHOD.md`、`docs/DATA.md`、`docs/HANDOFF.md`。
2. 固定 300 候选、30 个 SWE-smith 主 validation 任务、每题 4 次 NICE rollout、12 个 30 条子集、抽样种子 11/22/33。训练与主验证按任务分开。另固定 SWE-bench Verified、Pro Public V2、SWE-bench Live（Python full 快照）、Terminal-Bench 2.0 各 50 题作为迁移评测，seed=42，初始模型和 12 个分支共享这 200 题。迁移题不参与 NICE 评分、训练或选子集。各分支独立训练一遍，effective batch=1，无中途评测。
3. 不能根据成绩换题、换成员、删失败组、增加 epoch、调整各分支学习率或改主指标。高分组不必获胜。
4. 本机只运行 CPU 测试及显式 mock。不下载目标模型权重，不启动真实训练。真实执行必须使用明确的 `execution: server` 配置和授权的完整 GPU UUID。
5. 初始模型、LoRA、tokenizer、模板、mask、数据和服务配置必须可追溯。评分不能执行 optimizer.step。所有训练分支从同一个初始 adapter 开始。
6. 未知指标用 null。明确区分合成验证、真实实验和未执行项。不能把 mock 结果写成目标模型结果。
7. 允许修复依赖、环境和不改变研究定义的工程问题。保留错误、成本、原始证据及变更说明。源码身份变化后不得篡改旧运行的快照来强行恢复。
8. 完成步骤恢复必须校验 adapter、optimizer、RNG、数据顺序、源码和 checksum。中断的 Agent episode 只在干净容器重启，不能重放未知工具动作。同种基础设施错误最多重试两次；模型失败不重试。
9. 不提交数据、权重、检查点、凭据、运行状态或缓存。用户负责 GitHub 提交；除非另有明确授权，不执行 commit/push/merge。
10. 修改研究代码后运行 `python -m pytest -q`。流程变化后另运行一次新的显式 mock 输出目录。检查生成的 audit、表格及 PNG。未测试的目标模型、GPU和容器项目必须如实列出。
