# 扩大数据量的独立 NICE 探索实验

这是给第二位师兄运行的独立项目。原项目源码、配置和 `docs/HANDOFF.md` 保留不变。
完整服务器交接说明见 [新的 handoff](../../docs/HANDOFF_SCALED.md)。

| 项目 | 默认配置 |
|---|---|
| 模型（暂定） | `Qwen/Qwen2.5-Coder-7B-Instruct`，BF16，`AutoModelForCausalLM`；师兄端最终确定 |
| 硬件 | 4×72GB；先通过实际服务器容量和环境验收 |
| 训练来源 | `SWE-bench/SWE-smith-trajectories` 的 `tool` split |
| 主验证来源 | `SWE-bench/SWE-smith` 的 `train` split |
| 候选池 | 清洗、去重、排除验证/开发题后选 2000 条完整轨迹 |
| 主验证 | 100 题，另留 1 题开发验收；按任务与训练隔离 |
| NICE | 每题 4 次随机 rollout，共 400 次；只对 LoRA 求梯度 |
| 子集 | low/middle/high/random 各 3 份，每份 150 条，即候选池的 7.5%；分层为 666/667/667 |
| 训练 | 12 分支，每分支 1 epoch、有效 batch=4、38 次更新，共 456 次；1800 条次曝光 |
| 学习率 / LoRA | 恒定 `1e-5` / rank 16、alpha 32、dropout 0 |
| 评测 | 初始模型及 12 个 adapter 各评 100 题，共 1300 次 |
| 外部 benchmark | 关闭；本轮不能得出迁移效果结论 |
| 总正式 agent episode | 1700 次，不含开发验收及基础设施重试 |

训练阶段先停止全部推理服务，再四卡同时训练四个独立分支，分三批完成；
随后重启四个推理服务。NICE 梯度评分也已使用全部授权卡分片：每卡独立计算自己的
候选和验证方向，每条样本仍为 batch=1，不做 DDP 梯度平均。单个训练分支无需 DDP。
验证方向先汇总，全部为零则停止；各卡的共同参考和参数顺序校验通过后才计算候选池。
逐样本投影缓存和冻结分片计划位于 `scores/shards/`，支持校验后续算。
控制器会在正式 SFT 前，停止推理服务并在四卡分别实测 micro-batch 1/2/4，
用真实的同时多轨迹前反向和梯度累积完成固定有效 batch=4。所有卡都通过容量验收的
方案才可选；吞吐距离最佳不超过 5% 时优先较小物理 batch，留出显存空间。
测速权重用后丢弃，初始 adapter 不变。计划写入 `training_batch/plan.json`，
所有分支使用同一计划，恢复不会重新选择。最后 2 条仍更新一次，不丢弃。
最多交互 75 轮，其他 token/时间/上下文预算仍生效；预算耗尽且有非空补丁时
送官方测试判分，保留耗尽原因。这个协议只影响新项目。

服务器使用新的 Python 3.12+ 环境，进入**本目录**后执行：

```bash
python scripts/run_scaled_server.py --storage-root /data/nice-scaled-v1 --execute-server
```

默认发现当前可见 GPU 的前四张；可增加 `--num-gpus N --gpu-uuids GPU-... GPU-...`。
当前 7B 是暂定建议，最终由师兄端 Codex 按资源与工具调用验收确定。
更换模型可用 `--model-id ID --model-loader AutoModelForCausalLM`，须在首次运行前确定，
已经冻结的运行不能换模型续跑。只看计划时去掉 `--execute-server`。

本地验证使用显式 mock，不下载模型、不触及 GPU：

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m nice_validation --config configs/scaled-mock.yaml --output runs/scaled-check-01 run --profile mock
```

mock 的合成分数不代表研究结果。真实权重、Docker 镜像、官方任务测试和四卡显存
都必须在服务器完成 preflight 与 smoke；失败时控制器停止，不应绕过验收。
最终自动输出 `<storage>/controller/results-*.zip` 和 `completion.json`。

这是扩大规模并更换模型/交互预算的新探索实验，不能与第一轮直接作单因素因果比较。
主验证同时参与 NICE 评分，属于适配目标集，不是独立测试集。
