# NICE-Subset-Validation

完整服务器交接见 [HANDOFF](docs/HANDOFF.md)，配置权威为 [main.yaml](configs/main.yaml)。

| 项目 | 当前设计 |
|---|---|
| 基座 | 复用原固定版本 Qwen/Qwen3.8-27B 与 tokenizer，BF16 LoRA |
| 候选 | 原 SWE-smith trajectories/tool 来源，清洗后 2000 条 |
| 选样 | NICE 分数升序二等分，各 1000；高低各抽 512，sampling seed=11 |
| NICE | Verified/test 20 题，每题 8 次，共 160 rollout；仅 LoRA 梯度，8 卡独立分片 |
| SFT | 高低 × lr=1e-6/5e-6，共 4 分支；每分支 512 条、1 epoch、有效 batch=8 |
| 阶段权重 | 1/4、1/2、完整训练，即 128/256/512 条曝光、16/32/64 次更新，共 12 份 |
| SFT 选权重 | SWE-smith 未入候选池的成功轨迹 ID 对应的独立 100 题；每分支选 1 份，平分选更早节点 |
| 最终 benchmark | 与 NICE 20 题隔离的另外 100 个 Verified 任务；initial+4 个选中权重，共 500 episode |
| NICE 原题复测 | initial+4 个选中权重对原 20 题各 greedy 运行一次，共 100；单独报告，仅作诊断 |
| 总正式工作量 | NICE 160 + SFT 验证 1300（含 initial）+ benchmark 500 + NICE 原题复测 100 = 2060 episode；256 次 SFT 更新 |
| 协议 | 75 轮，32768/262144 输出 token，上下文 131072，墙钟 3600 秒，单工具 300 秒 |
| GPU | 默认 8×H200-144GB；4 分支并行训练并同步验证节点，结束后全卡推理；每卡一个服务、两个任务槽 |

SWE-smith 数据卡正文提及最初 5017 条训练轨迹，但当前 tool split 元数据已更大；本实验优先沿用服务器此前冻结的原始版本，不会冒称当前所有数据都是最初 5k。原四套迁移 benchmark 关闭。

服务器从新的 clone 目录、新 venv 启动，复用旧模型和原始数据：

```bash
python scripts/run_server.py --storage-root /data/nice-a-verified-v2 --reuse-from /data/旧实验存储根 --config-dir configs/server-a-v2 --execute-server
```

`--reuse-from` 指旧 storage 根，不是旧 Git 目录。必要时另加 `--model-path /实际模型目录`。复用核验包含 model_lock.json、固定 revision 和原始数据 sidecar/checksum。新实验的初始 LoRA、清洗/抽样、评分、训练和结果重新生成。旧模型目录不写入，权重不复制也不重复下载；新进程仍需从磁盘装入 GPU。

旧路径搬迁时按旧 storage 内的可信锁和 checksum 定位资源；旧数据若只导出了前缀，则在新目录补齐同 revision 完整 split。实际模型/数据路径与复用决策记录在新 storage 的 `data/reuse.json`，三类来源的最终版本/条数见 `data/source_lock.json`。

本机验证，不触及目标权重或 GPU：

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m nice_validation --config configs/mock.yaml --output runs/a-mock-check run --profile mock
```

mock 所有成绩均为合成验证。完整真实权重、8 卡显存、任务容器与官方判分尚需服务器 preflight/smoke；自动停止具体失败，不以 mock 顶替。

完成后回传 `<storage>/controller/results-*.zip` 与 `completion.json`。报告/表格/图在 `<storage>/runs/main/`；模型、12 份阶段 adapter、优化器断点和完整 trace 留服务器。详见 [数据流](docs/DATAFLOW.md)、[方法](docs/METHOD.md)、[数据](docs/DATA.md)。
