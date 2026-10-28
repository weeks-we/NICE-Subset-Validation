# 服务器交接：单入口执行

先读仓库根目录 `AGENTS.md`；研究定义见 [METHOD.md](METHOD.md)，数据见 [DATA.md](DATA.md)，迁移集见 [TRANSFER.md](TRANSFER.md)，代码分块见 [DATAFLOW.md](DATAFLOW.md)。本机仅做 CPU 与显式 mock；正式 Qwen/GPU/Docker 验收在服务器完成。

## 交给服务器 Claude Code 的指令

复制以下文字，并提供实际存储目录、授权GPU范围及要使用的卡数（默认8）；若服务器已有授权配置，优先读取。

> 读取 AGENTS.md 和 docs/HANDOFF.md，在本项目完成在线准备、真实 smoke、全部固定实验、审计和结果打包。先检查现有资源、配置、Python、网络、磁盘、Docker 和实际交付 commit。使用下方单入口，不需要我逐步操作。允许安装项目依赖、联网获取固定版本数据/模型/镜像、在授权设备上执行和恢复、修复工程缺陷并保留证据。对可自行解决的依赖、进程和路径问题持续处理，不要只给计划或要求我代跑命令。不得根据成绩换题、改指标、加 epoch、删失败组，不得用 mock 代替正式结果。修复导致源码或冻结环境变化时保留旧证据，使用新存储目录，记录原因。遇到缺少授权、凭据、容量或不可用外部服务等无法自行解决的条件，报告具体阻塞。完成后核对 audit、逐题分母、表格和实际图片，交付报告及结果 zip。不要提交或推送 Git。

## 唯一推荐入口

要求 Linux、Python 3.12+ 独立虚拟环境、Git、可访问的 Docker daemon、NVIDIA 驱动和至少两张授权 GPU。每份模型单卡运行；单卡容量由 smoke 验证，当前没有已验证的 TP/FSDP 后端。

默认使用 **8张GPU**，目标服务器为8×H200；`--num-gpus N` 可调整（N≥2）。分配为1张梯度/训练卡 + N−1张独立推理卡，自动生成N−1份服务配置、连续端口和N−1个任务worker。默认8卡时7个推理副本并行完成NICE rollout、主评测和四套benchmark。梯度评分与12个训练分支仍单卡顺序执行，未实现多卡评分或并行分支训练；不能声称所有阶段8卡同时计算。其他型号也须通过真实smoke。运行冻结后调整卡数、设备或并行配置，应使用新存储根与配置目录。

```bash
python scripts/run_server.py \
  --storage-root /srv/nice-subset-20261028 \
  --execute-server
```

改为4卡：增加 `--num-gpus 4`。指定授权设备池：增加 `--gpu-uuids <UUID1> <UUID2> ...`，按顺序选前N张，第一张训练，其余推理；未指定设备池时读取 nvidia-smi，并遵守已有 CUDA_VISIBLE_DEVICES 的完整UUID或物理索引顺序。服务器必须是已授权资源；不足N张、重复UUID或非法设备范围直接报错，不静默缩减或占用范围外设备。仍支持旧的 `--train-gpu/--service-gpu` 两卡方式，该方式显式选择2卡。单卡模式暂不支持两个独立模型副本的运行。

不带 `--execute-server` 只输出计划，不探测GPU或下载。首次默认在线下载模型，无需手填模型或数据文件路径。`--config-dir` 默认指向本仓库 `configs/`；并列实验需不同目录。`--port` 默认8765，8卡需8765–8771共7个连续空闲端口；不匹配服务占用则拒绝启动并关闭本次已启动的服务。已有模型可加 `--model-path`，必须先具有可信 `model_lock.json`，代码核对实际版本和文件 checksum。

入口自动串联：

1. 安装项目依赖，解析并冻结 HF/GitHub 来源，生成主配置及N−1份绝对路径服务配置。
2. 安装固定 commit 的 SWE-bench、SWE-smith、Harbor；Live Python harness 安装到独立 venv。
3. 预检；在线获取全部来源；先固定四套各 50 个迁移任务，再隔离、清洗并冻结 300 候选、30 主任务和 1 独立开发任务。
4. 下载并锁定模型，创建共同初始 adapter，准备和预检所选任务环境与镜像。
5. 自动启动所有独立GPU推理服务，逐个核对服务身份；运行真实 smoke（含32768 token更新和logprob/梯度链路），通过后执行全部实验。题目按固定索引取模分片，保持原题目与随机种子，结果统一排序汇总。
6. 停止自己启动的全部服务并记录成本，刷新分析、audit、打包报告。匹配的既有服务只复用，调用者负责其生命周期；明确记录 `service_reused`、gpu_count和inference_services。中途启动失败也清理此前已启动服务。

## 在线下载清单

| 内容 | 来源/范围 | 后续用途 |
|---|---|---|
| 基座模型与tokenizer | Qwen/Qwen3.8-27B 固定commit的模型分片、config、tokenizer及模板文件 | 本地模型服务、梯度和LoRA训练 |
| 训练原始记录 | SWE-bench/SWE-smith-trajectories 的 tool split | 清洗后冻结300候选，分支各选30 |
| 主任务原始记录 | SWE-bench/SWE-smith 的 train split | 持出30主任务、另1开发任务 |
| Verified与Live数据 | SWE-bench/SWE-bench_Verified test；SWE-bench-Live/SWE-bench-Live full | 各固定抽50题 |
| Pro与Terminal原生任务 | scaleapi/SWE-bench_Pro-os 的固定仓库/v2/tasks；Terminal-Bench 2.0注册表及固定任务仓库 | 各固定抽50题，保留原生环境和verifier |
| Python库和评测代码 | PyPI项目依赖；固定commit的SWE-bench、SWE-smith、Harbor、Live python-only | 训练、工具路由、绘图、官方判分；Live独立venv |
| 任务执行环境 | 选中任务所需Docker镜像、SWE-smith任务分支；原生任务构建/安装声明的依赖和资源 | 准备干净Agent容器及独立grader |

默认完整获取所需HF split和原生任务来源后再清洗/抽样，并非只下载最终300/30/50条。已有可信缓存和镜像可复用；LoRA、优化器检查点、图表和报告在服务器生成，不需要下载。Git、Docker daemon和NVIDIA驱动是服务器前置条件，入口安装Python依赖，不负责安装系统驱动或授予Docker权限。下载固定版本和checksum均留证，数据/权重/镜像不上传Git。

阶段日志在 `<storage>/controller/<阶段>/attempt_NNN/`；服务日志和PID分别在 `controller/service-<port>.log/json`；实验和收尾状态在 `controller/completion.json`。失败退出保留日志，Claude应读取具体错误并修复。配置解析或依赖安装失败发生在实验前，也须检查入口输出，不能仅依赖completion文件。

## 固定实验与结果

模型 `Qwen/Qwen3.8-27B`，按官方 config 使用 `AutoModelForImageTextToText`。训练源 `SWE-bench/SWE-smith-trajectories` tool，固定 300 条成功完整轨迹；主 NICE/同题评测源 `SWE-bench/SWE-smith` train，持出 30 题，每题四次 NICE rollout。低/中/高/全池随机 × 11/22/33 共 12 分支，各 30 条训练一遍，均从共同初始 adapter 开始。

迁移：Verified test、Pro Public V2 原生 Harbor、Live Python full 固定快照、Terminal-Bench 2.0 注册任务，各随机 50 题，seed=42。初始及 12 分支共用 200 题，不参与评分和训练。主 510 + 迁移 2600 = 3110 个 episode；独立 smoke、基础设施重试和 Pro 补丁重判另计。训练后 LoRA 12 份；加共同初始和 smoke，共 14 个 adapter 目录。主训练步骤检查点另计 360 份。

结果在 `<storage>/runs/main/`：`report.md`、`audit.md/json`、`summary.json`、`transfer/`、`tables/`、`figures/`。核对 52 个迁移汇总行、各组固定 50 分母、9 类图和指标完整性。缺失结果为 null。不能当作全量榜单，不要求高分组获胜。回传 `controller/results-*.zip`；服务器保留原始数据、权重、检查点和详细 trace。

## 中断与工程修复

确认原进程退出后，以相同参数重跑入口；仅有残锁时加 `--recover-lock`。锁核对主机/PID，拒绝抢占活跃进程；完成步骤复用 checksum，训练恢复 adapter/optimizer/RNG/顺序。中断 episode 从新容器重启，不重放未知工具动作；同类基础设施错误最多两次重试，模型失败不重试。

源码或冻结依赖变更必须使用新存储根与配置目录；禁止改旧 snapshot 绕过校验。残留容器先用 `cleanup-containers` 列出，再用 `--execute` 按精确运行标签清理。分步排错见 [ONLINE.md](ONLINE.md)。网络需实际探测 HF、GitHub、PyPI 和镜像站，不能从“可上外网”推断全部可达。

本地证据见 [VALIDATION.md](VALIDATION.md)。服务器 Claude 的交付标准是实际完成证据或具体不可自行解决的阻塞；本地测试不能保证目标 GPU 容量、外部可用性和整轮实验必然成功。
