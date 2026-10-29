# 服务器交接：单入口执行

先读仓库根目录 `AGENTS.md`；研究定义见 [METHOD.md](METHOD.md)，数据见 [DATA.md](DATA.md)，迁移集见 [TRANSFER.md](TRANSFER.md)，代码分块见 [DATAFLOW.md](DATAFLOW.md)。本机仅做 CPU 与显式 mock；正式 Qwen/GPU/Docker 验收在服务器完成。

## 交给服务器 Claude Code 的指令

复制以下文字，并提供实际存储目录、授权GPU范围及要使用的卡数（默认8）；若服务器已有授权配置，优先读取。

> 读取 AGENTS.md 和 docs/HANDOFF.md，在本项目完成在线准备、真实 smoke、全部固定实验、审计和结果打包。先检查现有资源、配置、Python、网络、磁盘、Docker 和实际交付 commit。使用下方单入口，不需要我逐步操作。允许安装项目依赖、联网获取固定版本数据/模型/镜像、在授权设备上执行和恢复、修复工程缺陷并保留证据。对可自行解决的依赖、进程和路径问题持续处理，不要只给计划或要求我代跑命令。不得根据成绩换题、改指标、加 epoch、删失败组，不得用 mock 代替正式结果。修复导致源码或冻结环境变化时保留旧证据，使用新存储目录，记录原因。遇到缺少授权、凭据、容量或不可用外部服务等无法自行解决的条件，报告具体阻塞。完成后核对 audit、逐题分母、表格和实际图片，交付报告及结果 zip。不要提交或推送 Git。

## 唯一推荐入口

要求 Linux、Python 3.12+ 独立虚拟环境、Git、可访问的 Docker daemon、NVIDIA 驱动和至少两张授权 GPU。每份模型单卡运行；单卡容量由 smoke 验证，当前没有已验证的 TP/FSDP 后端。

首次可由Claude创建并激活独立环境，例如 `python3.12 -m venv .venv` 后执行 `source .venv/bin/activate`；解释器尚未具备时，先检查服务器已有conda/uv等工具并准备Python3.12+。恢复已冻结运行时使用原环境，不重新创建或升级依赖。入口自动安装项目Python依赖，但不自动安装系统驱动或授予Docker权限。

评测依赖明确固定：SWE-bench使用官方 **v4.1.0** 发布引用解析得到的完整commit（包约束swebench==4.1.0、ghapi<2），不追踪重构后的5.x主分支；SWE-smith/Harbor/Live按各自锁定commit安装。安装后主环境及Live环境均执行pip check，再检查官方函数/常量接口，预检检查Git、Docker Compose与全部所选GPU。Verified镜像准备和正式判分统一使用swebench命名空间。兼容性失败先修复环境，禁止绕过检查继续下载权重或跑正式实验。

模型下载先单独取得固定 revision 的 config.json，用实际安装的 AutoConfig/AutoModel 映射导入官方模型类；配置与 loader 不匹配或模型依赖不能导入时，在下载权重前停止。数据准备检查真实字段：SWE-smith 在抽取主30题和开发题前排除空题面，写 `manifests/main_task_filter_report.json`；Verified 要求原生 version，Live 保留 test_cmds 列表和 log_parser，不伪造 version。缺失必需字段或 benchmark 题面为空时应修复来源，不能补造数据继续。

主任务补丁固定与启动时记录的真实commit比较，允许Agent在容器内自行commit；不能改回 `git diff HEAD`。Pro在setup冻结干净仓库root/commit并记录patch_base.json，重判环境必须具有相同起点。Pro补丁通过容器文件下载保留原始字节，主任务也保留CRLF；不要经文本换行转换后重放。模型破坏Git仓库导致收集失败记任务失败，Docker连接故障保留基础设施错误。只有实际执行了独立重判才标记pristine_regrade，超时/任务失败不能伪称重判完成。

使用示例中的较短存储根，如 `/srv/nice-subset-20261028`，避免把数据缓存放到多层冗长目录。datasets会把缓存绝对路径编码进锁文件名；本地Windows测试已遇到长路径限制，使用仓库内独立短缓存目录后通过。服务器仍须核对实际存储和缓存权限。

安装后先运行 `python -m pytest -q tests/test_official_model_interface.py tests/test_parquet_interface.py tests/test_source_schemas.py tests/test_patch_baseline.py`。这些测试覆盖官方混合注意力 Qwen/实际PEFT、真实Parquet库、数据字段及Git补丁，在CPU验证加载、梯度、更新、重载和补丁重放；若显示 skipped，说明目标依赖未装好，不能算接口验收通过。不下载27B权重，不替代真实GPU、Docker与完整Agent smoke。

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

恢复时还会检查全部已发布训练检查点的identity、步骤编号和checksum，最后一步的optimizer边界必须与连续检查点链一致。若一次episode已经完成、仅最终汇总文件发布中断，则直接恢复该次已完成结果（包括失败结果），不重新调用模型。只有尚未完成的episode才从新容器重启。

共同初始adapter先写入独立 `.initializing-*` 目录，文件和identity齐全后才原子发布；中断的临时目录保留作证，重跑会创建新的临时目录。已有正式目录缺identity时拒绝覆盖，需保留旧目录并使用新路径。复用初始adapter也核对基座锁和LoRA模块。手动run-online同样核对服务的模型/源码/运行目录/GPU身份，不能只凭ready状态接入；loopback连接绕过代理。学习率、训练epsilon/裁剪阈值和各整数预算必须为有效正值，非法配置在执行前拒绝。

源码或冻结依赖变更必须使用新存储根与配置目录；禁止改旧 snapshot 绕过校验。残留容器先用 `cleanup-containers` 列出，再用 `--execute` 按精确运行标签清理。分步排错见 [ONLINE.md](ONLINE.md)。网络需实际探测 HF、GitHub、PyPI 和镜像站，不能从“可上外网”推断全部可达。

cleanup-containers只处理主Docker执行器标有nice.run的容器；Harbor原生任务残留需根据具体trial/session的Compose标签单独核对清理，不能扩大到其他项目容器。

本地证据见 [VALIDATION.md](VALIDATION.md)。服务器 Claude 的交付标准是实际完成证据或具体不可自行解决的阻塞；本地测试不能保证目标 GPU 容量、外部可用性和整轮实验必然成功。
