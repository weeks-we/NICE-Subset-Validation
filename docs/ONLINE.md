# 服务器在线下载到完整实验

首次完整托管优先使用 [HANDOFF.md](HANDOFF.md) 的 `scripts/run_server.py --execute-server`，自动管理依赖、准备、服务、实验与收尾。本文保留分步排错方式。

本流程用于服务器上从Git克隆后的首次准备。单入口默认使用8张已授权可见GPU（1训练+7推理），用 `--num-gpus N` 调整，`--gpu-uuids` 可限制设备池；其他来源和版本自动固定。首次主实验前必须通过真实smoke。以下两卡分步命令仍用于兼容与排错；主入口自动管理N−1个服务，不需要手工启动。

当前数据角色以 TRANSFER.md 为准：训练来自 SWE-smith trajectories，主 30 题来自 SWE-smith 任务；Verified、Pro Public V2、Live Python full 与 Terminal-Bench 2.0 各持出 50 题。新增任务、依赖与缓存身份要求使用新存储/输出目录。服务器 Python 使用 3.12+。

## 1. 安装依赖并生成本机配置

在仓库根目录和独立 Python 环境中执行。Linux 需要 Git、Docker daemon、NVIDIA 驱动及可用 CUDA PyTorch。先用 `nvidia-smi -L` 查 UUID，再确认设备授权。以下目录和 GPU 是示例，须换成实际资源：

```bash
python -m pip install -e '.[server,dev]'
python scripts/configure_server.py \
  --storage-root /srv/nice-subset \
  --train-gpu GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa \
  --service-gpu GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb
python scripts/install_harness.py --config configs/server.local.yaml
python scripts/lock_environment.py --output /srv/nice-subset/environment.json
python -m nice_validation --config configs/server.local.yaml preflight --network
```

配置生成器只访问 Hugging Face 和 GitHub 元数据，不下载模型权重、不加载 GPU。生成：

- `configs/server.local.yaml`：梯度、训练、评测调度配置。
- `configs/service.local.yaml`：相同模型与输入、另一张 GPU 的推理服务配置。
- `<storage-root>/data/source_revisions.json`：训练、SWE-smith 主任务、四个迁移集、模型与各执行器的完整 commit。

两份本机配置都使用绝对路径，数值类型保持一致，并被 Git 忽略。重复运行相同命令复用已冻结版本，不重新追踪 main；已有配置与参数不同则拒绝覆盖。不要手工刷新旧实验的版本锁来复用结果。`install_harness.py` 从该锁的官方 commit 安装并核对 direct_url；不需要手工复制 commit。

已有模型时，生成配置可额外指定 `--model-path /actual/model/snapshot`。必须已有可信model_lock.json；代码核对实际固定revision和checksum。无锁目录优先在线下载。服务端口可通过--port指定；多服务从该端口连续分配。上述旧两卡示例生成1个worker，--num-gpus N方式生成N−1个worker及服务配置。

## 2. 在线下载和准备

首次需要下载模型时：

```bash
python -m nice_validation --config configs/server.local.yaml prepare-online --download-model
```

已核实来源并执行过 `freeze-model` 的现成模型：

```bash
python -m nice_validation --config configs/server.local.yaml prepare-online
```

命令顺序固定为：

1. 核对服务器依赖、GPU UUID、Docker 和官方 benchmark 安装身份；失败则停止。
2. 保存固定版本目标分片与字节数计划到 `online-download-plan.json`。
3. 按显式 `--download-model` 获取完整固定模型快照；已有来源不明目录必须先核实。
4. 下载 SWE-smith trajectories **tool**、SWE-smith 任务 **train**，以及迁移 Verified **test** / Live **full**；Pro Public V2 和 Terminal 2.0 使用固定 Git commit 的原生任务目录。发布来源 sidecar、SHA256 与 source_lock，不下载轨迹的 xml/ticks 分片。
5. 初始化共同 LoRA adapter，然后才创建运行快照，避免把缺失来源文件或 adapter 冻结进身份。
6. 使用本地真实 tokenizer 做消息校验、监督 mask、去重和跨集合检查，冻结 300 候选与 30 validation。合格样本不足时停止，不复制补数。
7. 主 validation 和开发题来自 SWE-smith；训练清洗排除主题、开发题及已冻结迁移题。开发题写入 development_task.json 并记录 checksum，同仓库候补替换排除开发题和训练候选。
8. 下载所有选中任务的 Docker 镜像，主集/Verified/Live 固定 digest；Pro/Terminal 原生环境以 install_only 预检并记录实际服务镜像 ID。不会为未抽中的全部 benchmark 任务准备镜像。

默认 `download` 模式，原始 Parquet 缓存在 `<storage-root>/cache/hub`，转换缓存位于 `cache/datasets`。可用 `--mode stream` 改为流式读取，但仍获取完整所需 split，不按前缀改变候选来源。实际原始数据、模型和运行目录都保留在 storage-root，不进入 Git。Docker 的存储目录由 daemon 管理，须另行确认它位于容量足够的盘。

下载使用 Hugging Face 官方的固定 revision / snapshot API，Parquet 转换使用 datasets 的显式 data_files 与 cache_dir；接口依据见[官方 Hub 下载文档](https://huggingface.co/docs/huggingface_hub/en/guides/download)和[datasets 加载文档](https://huggingface.co/docs/datasets/en/package_reference/loading_methods)。

中断后重复相同命令和参数即可恢复。完整数据先核对 sidecar/校验值，已完成文件不重复转换；JSONL 中断尾行先保留证据再重新获取该行；模型下载有独立来源标记，即便 config.json 已到达也会继续未完成的快照。来源、内容、配置或依赖变化时拒绝混用旧结果。

若进程在有运行锁的阶段被终止，确认旧进程退出后使用 `prepare-online --recover-lock`，并保留原有 `--download-model`/`--mode` 参数。同主机活跃 PID 或异主机锁不会自动回收。环境预检失败时保留报告，按 DATA/HANDOFF 修复；不因模型成绩自动换题。

## 3. 启动服务，继续 smoke 与主实验

准备完成后，终端一启动推理服务并保持运行：

```bash
python -m nice_validation --config configs/service.local.yaml serve --port 8765
```

终端二执行：

```bash
python -m nice_validation --config configs/server.local.yaml run-online
```

`run-online` 核对准备阶段、开发题身份、文件、依赖和服务健康，然后独立真实 smoke → 120 次 NICE rollout → 梯度评分 → 12 组抽样 → SWE-smith 基线 → 12 个分支训练及同题评测 → 四个迁移集的初始与 12 模型评测 → 独立迁移图表/主分析/审计。主集 510 次 + 迁移 2600 次 = 3110 次 Agent episode，训练实现与分支设置不变。不会根据迁移成绩换题或重新选训练子集。

smoke 包含完整 32768 token 更新、adapter 保存重载、真实工具/容器/官方判分、生成与梯度回放对齐。任一验收失败都会阻止主实验；任务是否 resolved 不作为工程 smoke 的成功条件。当前单卡后端的长序列容量仍须实测，不能用缩短输入替代验收。

执行中断后，保持相同代码、依赖和配置，再运行 `run-online`；需要恢复残锁时加 `--recover-lock`。已完成的 smoke/评分/训练步骤按现有 checksum 和恢复规则复用。具体重试、检查点、容器清理与结果回传见 HANDOFF。

## 网络和常见失败

Hugging Face、GitHub、PyPI 和 Docker registry 需要分别可达。使用团队批准的代理/证书设置；本地服务需要 `NO_PROXY` 覆盖 `127.0.0.1,localhost`。本流程不会关闭 TLS 校验、切换不明镜像源或把认证信息写进仓库。元数据可达不等于权重和镜像的大文件传输完成，实际下载异常会保留证据并停止。

出现依赖/API 不兼容时先在主实验前修复并重新验收，记录实际环境锁。若配置生成时官方 main 已不符合本项目 server 依赖约束，可在开始下载与冻结运行前选择兼容的官方 commit 并统一配置/来源锁；不得在已开始的实验中偷偷升级后继续复用缓存。
