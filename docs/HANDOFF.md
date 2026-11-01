# 师兄 A：Verified NICE / SWE-smith 选权重实验交接

本文件取代此前 A/B 交接。当前只有根项目，运行 `scripts/run_server.py`；B 目录已移除。`configs/legacy.yaml` 与 development.yaml 只保留历史工程回归，不能用于这次正式实验。

## 交给服务器 Claude Code 的执行指令

> 阅读根目录 AGENTS.md、README.md 和本文件，使用新的代码 clone、独立 Python 环境与新存储根，完成环境核验、旧基座及原始数据复用、新数据划分、真实 smoke、NICE、四个 LoRA 训练分支、十二个节点验证、四份权重选择、最终 benchmark、审计及打包。用户授权在分配的 Linux/8×H200-144GB 上安装依赖、获取公开固定版本资源、使用 Docker、执行与恢复实验及修复工程问题。正常依赖、路径、下载、进程问题自主解决并保留日志，不只给出计划或要求师兄逐条代跑。先查找旧实验的 storage 根、config snapshot、source_revisions 和 model_lock，核验可复用资源；不要重复下载或复制已有大模型权重。不得修改旧证据、覆盖旧 snapshot、拿旧选样/梯度/权重冒充新实验，不按 benchmark 成绩选权重或重试模型失败。代码/环境变化导致身份不兼容时保留旧目录，新建存储继续。只在确有权限、凭据、容量或外部服务阻塞且无法自行解决时报告具体原因。完成后检查实际 CSV、PNG、audit 与逐题计数，返回 results zip 和 completion.json。不要 commit/push Git。

## 固定研究定义

| 内容 | 定义 |
|---|---|
| 模型 | 原冻结 Qwen/Qwen3.8-27B，AutoModelForImageTextToText，BF16、SDPA、LoRA r=16/alpha=32/dropout=0，q/k/v/o_proj |
| 训练原始来源 | SWE-bench/SWE-smith-trajectories 的 tool split，优先复用此前固定 revision |
| NICE/benchmark 原始来源 | SWE-bench/SWE-bench_Verified 的 test split，完整获取后固定抽样 |
| SWE-smith 配套任务 | SWE-bench/SWE-smith 的 train split，按 instance_id 对应轨迹 |
| 候选池 | 合格、成功、完整、tokenize 后 ≤32768 token 的独立任务轨迹 2000 条 |
| NICE 主集 | Verified 20 题 × 8 rollout，共 160；与最终 benchmark 隔离 |
| NICE 抽样 | 分数升序二等分为 1000/1000；各抽 512，只有 sampling seed=11；监督 token 总量满足共同预算 ±10% |
| SFT 验证 | 不在整个 2000 候选池内的合格轨迹 ID，对应 SWE-smith 独立任务 100 题 |
| 开发验收 | 另 1 道 SWE-smith 任务，不参与评分、训练、选权重或 benchmark |
| 四个 SFT 分支 | low_lr1、low_lr2、high_lr1、high_lr2；lr1=1e-6、lr2=5e-6 |
| 训练预算 | 每分支一遍 512 条；有效 batch=8，64 次更新；128/256/512 条曝光保存 3 份阶段 adapter |
| 选权重 | 各 (组别, 学习率) 分别在 3 个节点中选择：官方 resolved 数最大，平分选更早节点；共 4 份 |
| 最终 benchmark | 从 Verified 剩余任务中固定抽 100；initial+四份选中权重，共 500 episode |
| NICE 原题最终复测 | 复用原 20 题，initial+四份选中权重各 greedy 跑一次，共 100 episode；诊断集，不是独立测试 |
| 旧外部 benchmark | Pro、Live、Terminal 与原四套 50 题迁移实验全部关闭 |

数据划分 seed=42（角色使用固定派生值），训练顺序 seed=2026；它们不是多次 seed 复现实验。同组的两个学习率必须使用完全相同的 512 成员和数据顺序，独立从同一初始 LoRA 开始；不能接续另一个学习率的训练。

SWE-smith 官方数据卡正文提到最初用于训练的 5017 条轨迹，但当前 split 元数据已增大。这里沿用原下载版本，记录实际 revision、原始条数和清洗条数；不要自动换成上游最新或按“正好 5k”截断。需至少 2101 个合格独立任务供候选、SFT 验证与开发使用，泄漏排除后仍须够数；不足则说明原因，不能复制补齐或按答案筛选容易题。

Verified 20 题用于 NICE，并在最后单独复测；另外 100 题只用于最后独立测试。SWE-smith 100 题用真实多轮 Agent 与官方测试的 resolved 选择 checkpoint，不用 teacher loss 代替。先固定所有集合，再接触模型；按任务 ID、显式 source_instance_id 家族和精确/近似题面排除重叠。不能由这些检查声称模型预训练绝无污染。

家族排除同时读取轨迹侧和任务侧的显式关联，并计算多级关联的闭包，避免连接配套任务后丢失仅在轨迹中出现的来源信息。`split_family_ids.json` 单独保存跨集合排除证据；原始轨迹 source_sha256 仍来自未改写的源记录。

## 新 clone 与旧模型/数据复用

推荐布局：

```text
/srv/code/nice-a-v2/          新 Git clone 与代码
/srv/venvs/nice-a-v2/         新 venv，避免 editable install 指向旧仓库
/data/old-experiment/        原模型、原始数据、缓存和旧结果，保留
/data/nice-a-verified-v2/     新 LoRA、划分、梯度、训练、评测和结果
```

`--reuse-from` 必须是原 storage 根，内有 `data/source_revisions.json`，通常另有 `runs/main/config/snapshot.json`、`data/source_lock.json`、`data/transfer_sources.json`。Claude 可根据旧 completion、配置和授权磁盘目录自行定位；不要把旧 Git clone 根误当 storage。

代码按旧来源记录映射：training→原 tool；旧 validation/SWE-smith→新 smith；旧 verified→新 validation/Verified。SWE-smith 数据集和同名 Git harness 的 commit 是不同命名空间，必须分别保留。原始 JSONL 必须保留 `.source.json`，核验 ID/revision/split/无前缀 limit/内容 checksum 后引用旧绝对路径；原始文件不改写。

模型优先读取旧 snapshot 的真实 model.path，也可显式传 `--model-path`。必须具有可信 model_lock.json，核对 ID、revision、全部文件 checksum 与官方 context 容量。已有合格模型直接使用，不重新下载或复制；校验会读取大文件，不代表正在下载。缺可信锁时先依据原下载记录、固定 Hub commit 与内容核实来源，不能随手造锁跳过检查。确有缺失才获取对应固定版本。

只复用基座/tokenizer、原始数据、下载缓存与已有 Docker 镜像层。重新建立初始 LoRA、数据清洗与划分、NICE rollout/梯度、选样、SFT、选权重与最终结果。不要整体链接旧 data、models 或 runs：这些目录还包含旧 adapter、任务映射和来源快照。

## 唯一正式启动入口

Linux、Python 3.12+、Git、Docker daemon、NVIDIA 驱动和兼容 CUDA PyTorch 必须可用。确认授权 GPU 可用；不要通过 kill 所有 Python 或全局 Docker prune 清资源。新 clone、新 venv、新 storage/config-dir；旧目录继续保留。

```bash
cd /srv/code/nice-a-v2
python3.12 -m venv /srv/venvs/nice-a-v2
source /srv/venvs/nice-a-v2/bin/activate

python scripts/run_server.py \
  --storage-root /data/nice-a-verified-v2 \
  --reuse-from /data/old-experiment \
  --config-dir configs/server-a-v2 \
  --execute-server
```

上述是示例路径，以服务器查到的实际目录为准。必要时加 `--model-path /实际模型目录`；可用 `--gpu-uuids GPU-... ...` 指定授权池，默认选择可见范围内 8 张完整 UUID。默认端口 8765–8772；被其他实验占用时加 `--port 8875` 等选择连续空闲端口。调整卡数用 `--num-gpus N`（至少 2），不能冻结后改卡数续旧运行。

入口安装本项目 server/dev 依赖及固定 commit 的 SWE-bench/SWE-smith harness，生成绝对路径配置并准备镜像。若服务器需专用 CUDA wheel 源，先在新 venv 装匹配 CUDA 的 PyTorch，检查 torch.cuda.is_available；不能用 CPU wheel 冒充。公开依赖、GitHub/Hugging Face 与镜像源都要实际核验，不从“可以上外网”推断全部可达。Loopback 自动加入 NO_PROXY。

原始数据不齐时下载完整 tool、Verified test、SWE-smith train 所需 split；不下载 xml/ticks，也不准备未抽中全部 benchmark 任务的镜像。已存在 Docker 镜像层会复用，新抽到的任务可能仍需拉取/构建。

启动前还要检查实际模型架构的 CUDA 快速路径。当前官方模型 config 使用 Qwen3_5 混合注意力；[Transformers 官方说明](https://huggingface.co/docs/transformers/model_doc/qwen3_5)指出其 DeltaNet 的 `fla`、`causal_conv1d` 快速内核缺失时可能退回更慢、更占显存的 PyTorch 实现。Claude 应根据实际冻结的 Transformers/PyTorch/CUDA 与 H200 架构，自主安装并验证兼容实现；先有匹配的 CUDA PyTorch，再构建相应扩展，不盲目升级已冻结环境。`preflight` 会记录相关包和 WARN；包存在也不等于内核实际被启用，要检查模型后端日志及 GPU 前反向吞吐。可用的优化在环境冻结前完成，不自动引入来源不明远程代码；无法使用时如实记录性能限制，并按真实 smoke 判断容量与可行性。

## GPU 分工与加速

默认 8 卡：

1. 环境/初始 adapter 与 smoke：GPU0 按需，开发题推理用 GPU1；服务惰性加载，避免 smoke 时占用训练卡两份模型。
2. NICE rollout 与 initial 的 SFT 验证：8 个推理服务，每卡一份底座，共 16 个任务槽；单卡模型调用串行，在工具等待期间交错两题，不是每卡加载两份底座。
3. NICE 梯度：停止全部推理服务，8 卡独立分片；候选逐条 batch=1，验证题的 8 rollout 在同一卡累积。无 optimizer.step、无 DDP/all-reduce，不平均掉不同候选的梯度。先汇总验证方向，全部为零就停止；各卡共享参考、参数顺序和投影校验通过后才计算候选。
4. 物理 batch 测速：仍无服务，各卡实测 micro-batch 1/2/4，始终有效 batch=8；只能从全部授权卡容量验收通过的方案中选统一计划，吞吐在最佳 5% 内优先较小物理 batch。临时测速权重丢弃，不改变初始 LoRA。
5. SFT：GPU0–3 各训练一个独立分支；GPU4–7 验证原子发布的阶段权重。每个分支完成前，其他进程不能读半写入的 adapter。
6. 四个训练进程全部退出并核验后，GPU0–3 也启动推理服务，加入正在进行的验证任务池；最终 benchmark 使用全部 8 卡/16 槽。只有模型评测阶段结束后才切换到下一份 LoRA，不混用权重。

少于 8 卡时按同一逻辑分训练卡与验证卡，分批训练；最终都转为推理卡。它不是 DDP/FSDP，单个训练分支不会把多张卡的 batch 再相乘。

推理服务保持基座驻留，只切换 LoRA；失败切换会清理并允许重新加载，不沿用不确定的模型状态。SFT 和 NICE 支持模型的 logits_to_keep，在所有有效目标位置计算输出层；分块 checkpoint CE 避免保存整段 FP32 softmax，损失定义和 token 覆盖不变。NICE 在零 dropout 条件下启用梯度 checkpointing，不创建优化器更新。

只有 NICE 随机 rollout 计算用于梯度重放的精确 logprob；greedy 的 SFT 验证和 benchmark 跳过不需要的额外 teacher-forcing 前向，保留实际输入/输出 IDs、输出结束原因与 token 数。

实际物理 batch、吞吐和显存必须以服务器测得值为准，不能因 H200 显存足够就跳过验收。四个正式分支使用同一有效与物理 batch，损失为每条轨迹 assistant-token 平均 CE 再按轨迹平均，padding/工具观察不监督；学习率恒定，不随 batch 自动放大。

## 交互协议与验收

| 限制 | 所有 NICE/SFT 验证/最终 benchmark 统一值 |
|---|---|
| 模型回复次数 | 75 |
| 每次生成量 | 最多 32768 token |
| 整题累计生成量 | 最多 262144 token，仅累计模型输出 |
| 输入＋本次输出上下文 | 最多 131072 token |
| 交互墙钟 | 3600 秒 |
| 单次工具执行 | 300 秒 |
| 官方判分超时 | 3600 秒 |

官方依据为 SWE-smith 仓库的 `agent/swesmith_infer.yaml`，其中 `instances.subset: verified`、`split: test`；明确规定 `per_instance_call_limit: 75`、`execution_timeout: 300`、最近 5 次工具观察与评测 `temperature: 0`。这些项目已对齐；服务器保存所用固定 harness commit 下该文件的原文作为来源证据，若与这里引用的公开配置不同先报告差异，不能静默改预算。

该官方 YAML 还设置 `per_instance_cost_limit: 3.5` 美元，但未明确指定单次/累计输出 token、上下文或整题交互墙钟上限。API 费用额度无法直接等价转换为本地 Qwen 推理 token 数，不将它伪装成某个官方 token 限制。Verified 官方 harness 对生成后的补丁执行测试，与 Agent 的交互预算是不同环节；因此这里采用 32768/262144 输出 token、131072 上下文和 3600 秒交互墙钟作为预先固定的本地补充限制，不能称为完整复刻官方协议，也不能把补丁测试超时当成交互时限。旧工具观察用明确标记替换后参与模型输入，完整原文仍在 trace/tool 文件；实际模型输入与输出 token IDs 用于 NICE 重放。预算相互独立，可能先触发上下文或墙钟，不能保证每题用满 75 轮。

本轮按用户要求扩大 token 上限：单次输出较此前 8192 提高到 32768（4 倍），累计输出由 65536 提高到 262144（4 倍），上下文由 65536 提高到 131072（2 倍）。累计输出跨多次回复相加，允许大于单次上下文窗口；每次调用仍按真实剩余上下文缩减可生成量。上限不代表必须生成到该长度，EOS/submit 会正常提前结束；墙钟仍为 3600 秒。SFT 教师轨迹清洗上限仍为 32768，不随 Agent 输出预算扩大而改变训练集。

ACTION/WALL/TOKEN/CONTEXT_LIMIT 且工作区非空补丁时自动送官方测试，保留原 stop_reason、automatic_submission、patch SHA256。自动送测不代表通过；空补丁、工具超时、格式错误不被自动改成成功。NICE 用 temperature=1、top_p=1、top_k=0；选权重和 benchmark 用 greedy，同一协议和固定预算。

逐调用 response/trace 保存生成额度与 finish_reason（eos/length/other），事件日志另保存本次调用前整题已用/剩余输出量。EOS 恰好落在最后允许 token 时仍归 eos。

真实 smoke 必须通过：32k 长度容量更新；131072 总长度且 32768 输出位置的 NICE 反向容量；独立开发题的模型→工具→Docker→补丁→官方判分链路；logprob/梯度重放对齐；adapter 保存重载一致。所有 221 个已选任务（20+100+100+1）的环境预检失败要修环境，不按答案换题。扩大后的 NICE 反向容量是必须实测的目标，不能凭模型原生最大上下文或 144GB 显存推定可用。先确认 CUDA 快速内核、梯度 checkpointing 和有效位置分块 CE 真正启用；容量不足应保存峰值显存和具体失败，不能静默缩上下文、量化或改训练定义续同一运行。

## 保存节点、选权重和成本

每分支一次训练，固定在第 16、32、64 次更新保存 12 份可加载阶段 adapter。额外的逐更新 adapter/optimizer/RNG/order 断点用于恢复，不是增加候选训练节点。选中 4 份只写 selection.json 指针，不额外复制权重。基座、共同初始和 smoke adapter 分开保存。

训练前重新核验冻结候选/子集/job 清单。`training_identity` 绑定实际 token/labels 等训练行、学习率、batch 计划与 run 身份，写入训练报告和各节点；最终审计重新计算对照。不能只凭任务 ID 相同就复用改过数据的断点。

12 个节点各验证 100 道 SWE-smith 题，共 1200 次；initial 验证另 100 次。每个组别和学习率内独立选阶段，平分选更早曝光，不能以 benchmark 定平局。只有 12 组验证和 4 分支完整后发布 model-selection，再放行 initial+4 个选中模型的 Verified benchmark。

### 必须完成 NICE 原 20 题的最终复测

服务器 Claude Code 不得遗漏此步骤：先冻结 `selection.json` 的四份权重，再使用 initial 与这四份权重，对 `manifests/validation.json` 中原封不动的 NICE 20 题各执行一次 greedy 多轮 Agent 和官方判分。不得另抽题，不额外跑 8 次，不评测未选中节点，也不根据复测结果重新选择节点、学习率、训练集或刷成功率。

无需增加启动参数：`scripts/run_server.py` 已在每个模型的 100 题 benchmark 完成后自动调用 `evaluate(run, "nice_retest", model_id)`，同一模型驻留时再跑这 20 题，避免反复加载。使用全部 8 卡和相同的 75 轮、32768/262144 输出 token、131072 上下文、3600 秒墙钟与 300 秒工具预算；同样记录超限、非空补丁自动送测、token 和时间成本。结果写入独立的 `evaluation/nice_retest/<model_id>/`，不得覆盖原 `rollouts/nice/`。

initial 也必须重新 greedy 评测，不能将原 NICE temperature=1 的 8 次 rollout 成功率当作 greedy 基线。原随机 rollout 的 160 条记录仍保留在 `nice_task_results.csv`。复测属于已参与 NICE 选样的诊断结果，不能称为独立 benchmark，也不能与另 100 题混合计算 120 题的独立测试成绩。失败保留在每模型固定 20 题分母，基础设施错误先按既定策略修复，缺失结果不允许完成审计。

正式工作量：160 NICE + 1300 SFT 验证 + 500 benchmark + 100 NICE 原题复测 = 2060 episode，256 次训练更新、2048 条次训练曝光。开发 smoke、batch 测速、工程参考、基础设施重试及中断成本另计。耗时主要可能在多轮验证/判分/容器和模型装载，不能用训练更新数推算整场耗时。NICE 每题失败 rollout 贡献零，但仍固定除以 8；零方向任务仍在 20 题分母中，全部为零停止，低 coverage 告警并报告。

## 恢复与工程修复

同源码、依赖、输入和配置重跑原命令；确认旧控制器已退出，仅有残锁时加 `--recover-lock`。训练 worker 租约和 PID 也要检查，不能只删 controller.lock。父进程退出后训练 worker 在下次更新前停止；正在执行的 CUDA 操作仍需确认进程退出。

已完成阶段核对 checksum；坏的完成证据必须保留并停止，不静默重算覆盖。SFT 恢复 adapter/optimizer/RNG/顺序和有效 batch；中断节点发布从可信 optimizer 边界修复。中断 episode 从干净容器重启，不重放未知工具动作。同类基础设施错误最多两次重试，模型失败不重试刷成功率。

修复改变研究源码或冻结环境时保留失败证据，使用新 storage/config-dir；可继续引用同一套核验过的旧原始输入，但不得改旧 snapshot、改 checksum 或拷贝旧梯度伪装续跑。只清理本控制器拥有的进程和本运行标签容器，不全局 prune。

## 结果检查与交付

新 storage 下：

- data/source_revisions.json、source_lock.json、reuse.json：实际输入版本、路径和复用记录。
- models/initial_adapter：共同初始 LoRA；基座可能仍在旧目录，按生成配置为准。
- runs/main/manifests：候选、NICE/SFT 验证/benchmark/开发 ID、两组成员、四个训练 job、十二个节点定义。
- runs/main/scores：分数、覆盖率、8 卡参考与投影缓存；缓存约 0.26GB＋元数据，不含原始全梯度。
- runs/main/training：四分支逐步断点、十二份阶段 adapter、checkpoint.json、训练摘要和 worker receipt。
- runs/main/evaluation/selection：initial+12 节点的 SWE-smith 逐题证据。
- runs/main/selection.json：各 (组, lr) 的选中节点及验证数，不读取 benchmark 成绩。
- runs/main/evaluation/benchmark：initial+4 份选中权重的 Verified 逐题证据。
- runs/main/evaluation/nice_retest：initial+4 份选中权重对原 NICE 20 题的 greedy 复测证据，与原随机 rollout 隔离。
- runs/main/tables、figures、summary.json、report.md、audit.json/md：可读结果。
- controller：命令/服务日志、source 身份、completion 与最终 zip。

完成前严格核对：2000 候选、两组各 512；20/100/100/1 三类验证/测试与开发任务互斥且与候选隔离；每题 8 NICE rollout；4 分支各 64 更新；12 个节点曝光与 checksum 正确；每分支从同一初始 LoRA；4 个独立的 (组, lr) 选择；1300 个 SFT 验证、500 个 benchmark 和 100 个 NICE 原题复测结果分母完整；总计 2060；mock 不混入正式结果；失败不删题。审计不得有 FAIL，低 coverage 的 WARN 要如实保留。

默认只回传两个小文件：

1. `<storage>/controller/results-<UTC时间>.zip`
2. `<storage>/controller/completion.json`（status=completed）

zip 包含报告、CSV、图、配置/ID 清单、selection 与 checksum，不含基座、LoRA 权重、优化器断点、原始数据、完整 trace 或投影缓存。它们留服务器；有具体失败诊断需求再按任务导出。失败时交付具体错误和证据路径，不伪造 completed 包。

优先看 benchmark_summary.csv、high_low_comparison.csv、paired_tasks.csv；选权重过程看 checkpoint_validation.csv 和 selection.json。只有一个 seed，不能估计 seed 稳定性；四个模型共享任务，必须做配对描述，不当作四次独立复现。SWE-smith 选权重成绩不是独立测试成绩。

NICE 160 次随机 rollout 逐题结果另见 `nice_task_results.csv`。最终复测看 `nice_retest_summary.csv`（5 个模型，各分母 20）、`nice_retest_task_results.csv`（100 条）、`nice_retest_paired_tasks.csv`（20 行模型配对）与 `nice_retest_high_low_comparison.csv`（2 个学习率的高低配对比较）。这些表自动进入最终 zip；汇报时明确标为“参与 NICE 选样的原题复测”，与独立 benchmark 分开。`summary.json/nice_retest` 记录诊断结果及非独立测试标记；`task_results.csv` 以 scope 区分所有评测。

`summary.json/logical_token_usage` 分别汇总 NICE、SFT 验证、benchmark、nice_retest 与训练的已知 token 数，未知项为 null；这些是逻辑 token，不包含 activation checkpointing 的重算，不能当作 FLOPs。阶段成本 CSV 汇集控制器、训练与已退出推理服务的成本记录，完整事件/服务日志留在服务器。

## 外部依据与本地验证边界

- SWE-smith 数据：https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories
- SWE-smith 配套任务：https://huggingface.co/datasets/SWE-bench/SWE-smith
- Verified：https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified
- SWE-smith 官方推理：https://github.com/SWE-bench/SWE-smith/blob/main/agent/swesmith_infer.yaml
- 原模型官方 config：https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json
- NICE 官方源码：https://github.com/JTWang2000/NICE

本项目仍是单初始 checkpoint、fresh-Adam/CountSketch、LoRA 参数空间上的 NICE 风格变体，不是原论文 warmup/多 checkpoint/原 Adam 状态的完整复现。本地只运行 CPU、微型随机模型和显式 mock；实际大模型、8 卡 H200、Docker 和公网完整性必须在服务器验收。最新本地检查记录见 docs/VALIDATION.md。
