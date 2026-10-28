# 同源验证与独立迁移评测

2026-10-27 按用户确认更新：NICE 评分和同题正式评测改用 SWE-smith 任务；四个外部 benchmark **各 50 题，共 200 题**。Pro 使用 Public V2，Terminal 使用 2.0。

| 角色 | 来源 / 版本 | 使用数量 |
|---|---|---:|
| 训练候选 | SWE-bench/SWE-smith-trajectories，tool | 清洗后固定 300 条成功轨迹 |
| 主 validation | SWE-bench/SWE-smith，train | 30 题 |
| 独立开发题 | SWE-smith 未进入主集的任务 | 1 题 |
| 迁移 Verified | SWE-bench/SWE-bench_Verified，test | 50 题 |
| 迁移 Pro Public | scaleapi/SWE-bench_Pro-os，v2/tasks | 50 题 |
| 迁移 Live | SWE-bench-Live/SWE-bench-Live，Python full 固定快照 | 50 题 |
| 迁移 Terminal | Terminal-Bench 2.0 注册版本指向的固定 Git commit | 50 题 |

## 冻结与隔离

先冻结迁移题，再选主 validation 与开发题，最后清洗训练候选。四个外部集各按固定 seed=42 派生的独立随机流无放回抽 50 题，输入先按 ID 排序；按固定 suite 顺序排除已选迁移题的重复 ID/规范化问题。记录源 commit、种群/可抽数量、选择 ID 和 checksum，不参考模型成绩、环境是否好跑或模型支持的题型。

主 30 题按仓库比例、seed=42 选择；开发题为余下 ID 最后的任务。主任务先排除迁移题的相同 ID 和重复/近似问题。训练候选排除主 30 题、开发题和迁移 200 题的相同 ID、显式 source_instance_id 关系及重复/近似问题。SWE-smith 的其他任务仍可提供训练轨迹，不能把整个同源任务源都过滤掉。

环境故障的同仓库主集替换只允许在最终冻结和任何主模型接触之前，候补必须排除训练任务、开发题和相同/近似训练问题。外部 50 题不自动替换；缺题/基础设施错误保留，成功率记 null 并先修复环境。文本与显式元数据规则不保证清除所有同源变体或模型预训练污染。

## 训练与评测计数

12 个分支保持原设置：低/中/高/全池随机各 3 个 30 条子集，抽样种子 11/22/33，共同初始 adapter，各训练一遍、30 步。主 30 题每题 4 次 NICE rollout，共 120 次；初始与 12 个训练后模型各做 30 次同题评测，共 390 次。

全部 12 个分支训练完成后，初始与 12 个模型在四套相同冻结样本上各评测一次：13×4×50=2600 次。总计 **3110 次 Agent episode**，另计 smoke、明确基础设施重试和 Pro 的纯补丁 verifier replay。迁移评测不执行 optimizer.step，不新增训练 adapter。

正式训练后 LoRA 仍为 **12 份**，另有共同初始和 smoke 各 1 份。主训练步骤检查点仍为 360 份。

## 执行器与证据

- SWE-smith 在可信准备阶段从 image_name 拉取 instance_id 分支，记录实际 commit SHA，生成只保留公开 tip 的浅历史镜像；移除其他 refs/reflog、不可达历史对象和 origin，避免 Agent 通过历史读取已删除的隐藏测试。正式 Agent 仍断网。`swesmith_grade.py` 复用官方执行与解析函数，在独立容器按同一冻结 SHA 恢复测试并判分。[官方 harness](https://github.com/SWE-bench/SWE-smith/blob/main/swesmith/harness/eval.py)。
- Verified 使用固定官方 SWE-bench harness。
- Live 使用官方推荐的 Python-only 分支，安装在独立 Python venv，避免覆盖主环境的 swebench 包；数据和 harness 分别固定 commit。[Live 官方说明](https://github.com/microsoft/SWE-bench-Live/tree/python-only)。
- Pro Public 使用 V2 的原生 Harbor 任务，固定 instruction、task.toml、环境、verifier 和参考文件 checksum。生成补丁后在全新环境只应用捕获的 patch，再运行未修改 verifier，以重判结果为准，同时保留原环境分数。[Pro V2 说明](https://github.com/scaleapi/SWE-bench_Pro-os/blob/main/v2/README.md)。
- Terminal 使用 2.0 注册版本的原生任务和 verifier，不转换成 SWE patch 题。[Harbor](https://github.com/harbor-framework/harbor)、[Terminal-Bench 2.0](https://www.tbench.ai/?version=2.0)。

准备阶段以 Harbor install_only 为选中的 Pro/Terminal 任务创建环境，不调用模型；记录实际 Docker 服务镜像 ID，在正式运行及 Pro 全新环境重判时逐次核对。主集和 Verified/Live 同样固定 Docker digest、核对干净起点并隔离隐藏判分材料。模型只接收公开问题/instruction 与工具观察，不把宿主上的 verifier 或参考解拼入消息。

所有评测使用同一冻结 LoRA 服务和项目配置中的 token/action/墙钟预算；同一 benchmark 内所有模型设置相同。Pro/Terminal shell 调用独立，SWE runner 使用持久 shell；这些 scaffold 差异随报告说明。结果属于固定预算下的随机子集实验，不是完整官方 leaderboard。Terminal 如有部分奖励，单列 native_mean_reward；完整成功定义为 reward=1。

## 在线流程与结果

同一 ONLINE.md 入口会固定所有来源与执行器版本。`install_harness.py` 安装主 SWE-bench、SWE-smith、Harbor，并在独立 venv 安装 Live fork。服务器使用 **Python 3.12+**。`prepare-online --download-model` 获取主数据和迁移源，冻结抽样、清洗并预检全部选中任务环境；启动服务后 `run-online` 验收 smoke，再运行主实验与迁移。

结果位于 `<run>/transfer/evaluation/<benchmark>/<model>/`，每次基础设施重试使用新 attempt/容器，沿用固定重试上限，模型失败不重跑。可独立执行 `evaluate-transfer --model initial` 或指定已训练子集；增量汇总会归档旧分析。

`transfer/summary.json`、`transfer/tables/summary.csv` 和 `paired_tasks.csv` 分别报告四个 benchmark 的 baseline、resolved/50、paired delta、原生奖励、完整性与 NICE 分数对外部提升的描述性关联。两类四面板 PNG/PDF/灰度图展示分数与提升、每组全部三个点，不做均值柱或默认显著性推断。50 题下每道二元成功对应 2 个百分点，不合并不同 benchmark 成一个成绩。

主 summary/report 继续报告 SWE-smith 同题关系，并链接迁移报告。audit 核对 4×50 抽样、训练/主集隔离、52 组结果与完整分母；回传 bundle 只导出汇总、图、ID 和 checksum，不含 benchmark 题面、gold patch 或完整 verifier。

本次改变数据角色、依赖和缓存身份，必须使用新配置、新存储/输出目录；旧运行与报告保留。真实模型、真实样本及四套 Docker/harness 链路仍须服务器实测，本地 CPU/mock 不能替代。
