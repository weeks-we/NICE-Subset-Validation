# 数据流与代码验证映射

推荐从 `scripts/run_server.py` 进入，由 `nice_validation/supervisor.py` 管理生命周期。以下路径相对于服务器 `<storage>`，configure_server 将其生成绝对路径；main/example 配置含占位项，不能直接真实执行。

| 块 | 来源与输出 | 代码 | CPU 验证 |
|---|---|---|---|
| 版本与路径 | HF 模型/数据 + GitHub harness → data/source_revisions.json、configs/*.local.yaml；两张独立GPU | online.resolve_sources/configure_server | test_online、test_supervisor：commit固定、配置幂等、绝对路径、已有模型证据 |
| 在线获取 | SWE-smith trajectories/tool → data/training.jsonl；SWE-smith/train → data/validation.jsonl；来源/SHA → data/source_lock.json | acquire.hf_split、data.fetch_data | test_online：split范围、分片、断线/尾行/发布恢复、防改 |
| 迁移来源 | Verified/test、Live/full → JSONL；Pro v2/tasks、Terminal2.0注册路径 → data/benchmark_repos；data/transfer_sources.json | transfer_sources.acquire_transfer/checkout | test_transfer、test_supervisor：Git首次检出/中断恢复、注册集边界、源与task文件SHA |
| 清洗与隔离 | 先冻结外部4×50；排除任务/家族/近似文本重叠；主30、开发1、成功候选300 → runs/main/manifests | data.prepare、protocol | test_data_state、test_protocol_regressions、test_online、test_transfer |
| 模型与初始状态 | Qwen/Qwen3.8-27B固定revision → models/Qwen3.8-27B、model_lock.json；tokenizer同路径；models/initial_adapter | model.fetch_model/initialize_adapter/Runtime | test_online、test_math、test_supervisor；实际目标模块由服务器smoke验收 |
| 监督与梯度 | assistant有效token均值CE；奖励轨迹NLL求和/固定4次分母；Adam初步方向→CountSketch | protocol、model、scoring | test_math、test_protocol_regressions：CPU有限差分、符号/归一化/零梯度、实际Runtime方法 |
| 子集与训练 | 300→3个100分层+全池；每类3个30条条件抽样；独立12×30更新 → training | subsets、training | test_subsets、test_training：token预算、重叠、共同初始、optimizer/RNG恢复逐值一致 |
| 主任务执行 | 独立容器、隐藏测试隔离、官方SWE-smith grader → 120 NICE+390正式同题episode | environments、harness、scripts/swesmith_grade.py | test_harness、test_swesmith、test_server_guards：协议、期限、隐藏历史、判分接线 |
| 外部执行 | 13模型×4套×50；Verified/Live官方harness；Pro全新环境补丁重判；TB原生reward | transfer、harbor_agent、harbor_runner | test_transfer：原生环境/镜像/奖励分流、重试、不完整固定分母 |
| 服务与托管 | 服务GPU→loopback；启动、身份核对、失败收尾、停止后分析、audit/zip → controller | service、supervisor、scripts/run_server.py | test_supervisor、test_server_guards：GPU路由、代理绕过、拒绝错误服务、顺序与失败清理 |
| 分析与图 | 主结果 + transfer/summary → CSV/9类PNG/PDF/灰度/审计 | analysis、transfer_analysis | test_cli_pipeline、test_transfer：完整mock、52行、固定50、正式null、resume、增量分析 |

原始数据在服务器在线准备，未加入 Git；本机没有正式数据、目标模型或镜像。HF/GitHub main 只在首次解析元数据时使用，此后下载、安装和恢复使用完整 commit。Terminal2.0 的任务列表与 commit 来自固定 Harbor registry，不混入同仓库未注册任务。Live 使用 `<storage>/envs/live/bin/python`，隔离同名 swebench 包。

默认GPU池8张，可用--num-gpus调整（至少2），gpus.select_gpus遵守授权池或可见设备顺序；online生成1份主配置和N−1份服务配置，supervisor逐个管理独立GPU/端口/日志。主任务和外部benchmark按原任务索引取模分片，排序汇总且不改随机种子；评分和训练仍单卡顺序执行。test_gpu_pool覆盖2/4/8卡、设备不足、可见范围、启动失败清理，以及四套200题的真实线程并行路由。

服务与梯度端共享模型、tokenizer、adapter、精度、模板和工具定义，GPU UUID分开。健康状态包含源码/环境/模型/运行身份，不能把任意ready进程当正确服务。完整长度更新、真实模型logprob对齐和Docker grader仍需服务器证据，失败阻止正式实验。
