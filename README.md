# NICE-Subset-Validation

检验单个初始 checkpoint 上的 NICE 型数据评分，能否预测短期 LoRA SFT 后的 Coding Agent 收益。项目自包含，不依赖本机的 RSI 仓库。

固定设计：从 SWE-smith trajectories 选 300 条成功轨迹，从 SWE-smith 任务源独立保留 30 道主 validation，每题 4 次 NICE rollout；高、中、低、全池随机各 3 个 30 条子集。12 个分支从同一 adapter 独立训练一遍。另冻结 SWE-bench Verified、Pro Public V2、Live（Python full 快照）、Terminal-Bench 2.0 各 50 题，用于初始模型与 12 个训练后模型的迁移评测。合计 510 次主集 + 2600 次迁移 = **3110 次 Agent episode**，360 次训练曝光。smoke、基础设施重试和纯补丁重判另计。详见 [TRANSFER.md](docs/TRANSFER.md)。

本地实现包含数据获取与冻结、消息校验、assistant 掩码、奖励梯度、流式投影评分、预算条件抽样、独立 SFT、步骤检查点、LangChain 工具路由、持久 Docker shell、官方 SWE-bench 判分、CSV/PNG/PDF、状态和审计。**目标模型、真实数据和容器链路尚待服务器实测。** Mock 的分数、训练和成功率均为合成值，不能用于研究结论。

## 本地运行

Python 3.11 或更高版本。在仓库根目录执行：

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m nice_validation --config configs/development.yaml doctor
python -m nice_validation --config configs/development.yaml run --profile mock
python -m nice_validation --config configs/development.yaml status
```

CPU 小模型测试使用本地随机初始化的 PyTorch 模型，不下载 checkpoint。需要 PyTorch 的测试在缺少该依赖时应先安装 CPU 版 PyTorch；本地已测环境见 [requirements/local-tested.txt](requirements/local-tested.txt)。普通 mock 流程不需要模型库、GPU、Docker或网络。

产物位于 `runs/mock/`。打开 `report.md`、`audit.md`、`tables/` 和 `figures/`。审计的 `formal_evidence: WARN` 表示这是显式合成验证。`success_rate`、`delta` 和正式相关性保持 null；演示值使用单独的 `synthetic_demo_*` 字段。图上也有合成标记。

重复运行会验证已完成产物的 SHA256。配置、源码或冻结输入变化时拒绝复用原运行目录。分析不完整运行时，缺失题不会被当作失败凑分母。补齐运行后使用 `analyze --refresh` 保留旧报告并重新汇总。

## 服务器运行

默认使用 **8张GPU**（目标8×H200），可用 `--num-gpus 4` 等调整，至少2张。自动分配1张梯度/训练卡及N−1张独立推理卡；8卡时主任务与四套benchmark使用7个推理副本并行。梯度评分和12分支训练仍单卡顺序执行。可用 `--gpu-uuids <UUID1> <UUID2> ...` 指定授权范围，未指定则发现可见设备，不足卡数会报错。详见 [HANDOFF.md](docs/HANDOFF.md)。

完整托管优先按 [HANDOFF.md](docs/HANDOFF.md) 使用 `python scripts/run_server.py --storage-root <目录> --execute-server`，默认8卡；调整卡数增加 `--num-gpus N`。该入口自动安装、在线准备、管理所有推理服务、运行smoke与全部实验，再停止自建服务、审计和打包。Claude Code可按handoff连续执行与排错。代码分块及测试映射见 [DATAFLOW.md](docs/DATAFLOW.md)。

首次在线准备由单入口串联 [在线流程](docs/ONLINE.md)：配置生成、固定来源、下载、清洗、adapter与镜像准备、服务、真实smoke和主实验。提供实际存储目录与授权GPU范围，默认8卡，无需手工复制数据或逐项抄版本号。

下列为分步操作方式，用于排查或已有配置的服务器；首次在线准备按上述 ONLINE 文档执行即可。手工配置时先读 [docs/HANDOFF.md](docs/HANDOFF.md)，复制 `configs/server.example.yaml` 为被 Git 忽略的 `configs/server.local.yaml`，填写真实路径、完整 GPU UUID、三个 Hugging Face revision 和 SWE-bench commit。占位配置会被执行器拒绝。

```bash
python -m pip install -e '.[server,dev]'
python scripts/resolve_sources.py --output data/source_revisions.json
python -m nice_validation --config configs/server.local.yaml doctor
python -m nice_validation --config configs/server.local.yaml fetch-data
python -m nice_validation --config configs/server.local.yaml fetch-data --mode stream --limit 3000 --execute
python -m nice_validation --config configs/server.local.yaml fetch-model
python -m nice_validation --config configs/server.local.yaml init-adapter
python -m nice_validation --config configs/server.local.yaml prepare-data
python -m nice_validation --config configs/server.local.yaml prepare-environments
```

`--limit 3000` 是原始读取上限示例，清洗后不足 300 条会停止。原始读取前缀会写入来源记录，不能描述成对完整数据集的均匀抽样。省略 limit 才会读取 tool split 的全部记录。`fetch-data` 默认只预览目标分片和字节数；只有 `--execute` 才写入数据。`fetch-model` 必须显式调用且仅允许 `execution: server`；已有模型则先核对来源，再执行 `freeze-model`，不会重复下载。

模型服务用另一张已授权 GPU 启动：

```bash
python -m nice_validation --config configs/service.local.yaml serve --port 8765
```

服务器先执行 `preflight --network` 汇总配置、依赖、GPU、Docker 和网络缺口。数据/模型/镜像/服务就绪后执行 `preflight --prepared`；本地 tokenizer 可用 `python scripts/check_tokenizer.py --config configs/server.local.yaml` 独立检查。预检不会下载或加载权重。分步操作详见 ONLINE，完整托管使用 HANDOFF 的单入口。

此次协议与缓存身份修复后，请使用新的运行目录。正式运行前的真实 smoke 包含完整 32768 token 容量更新验收；本地 mock 不能替代。服务切换 adapter 支持失败恢复，残留容器可通过 `cleanup-containers` 先列出，再用 `--execute` 按精确运行标签清理。

主进程随后执行：

```bash
python -m nice_validation --config configs/server.local.yaml smoke
python -m nice_validation --config configs/server.local.yaml run --profile main
python -m nice_validation --config configs/server.local.yaml status
python -m nice_validation --config configs/server.local.yaml resume
```

独立命令还包括 `rollout-nice`、`score`、`make-subsets`、`train --subset high_11`、`evaluate --model initial`、`analyze` 和 `audit`。参数顺序是先给全局 `--config`/`--output`，再给子命令。

每个模型副本使用一张完整 UUID 绑定的 GPU。多个推理服务可按固定题目分片并行；配置 `agent.workers` 和对应 `service_endpoints`。不支持把一份训练模型通过配置数字变成八卡 DDP/FSDP。单卡长序列是否可行，必须由真实 smoke 验证。

## 方法与交付

- [方法定义与研究边界](docs/METHOD.md)
- [数据来源、格式与冻结](docs/DATA.md)
- [服务器交接与恢复](docs/HANDOFF.md)
- [验证记录](docs/VALIDATION.md)
- [参考来源与复用边界](docs/REFERENCES.md)

SWE-smith 主 validation 同时参与评分和同题评测；迁移能力由四套独立冻结的外部 50 题分别衡量。不以主集成绩证明泛化，不把子集成绩当成官方全量榜单，不声称完整复现 NICE。没有要求高分组获胜。

用户自行提交 GitHub。本项目没有自动 commit、push 或合并步骤。数据、权重、状态和运行日志由 `.gitignore` 排除。
