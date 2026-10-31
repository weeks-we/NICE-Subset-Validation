# 第二位师兄：独立扩大规模实验交接

请把这份文档和完整仓库交给服务器上的 Claude Code。第一轮继续使用原来的
`docs/HANDOFF.md`；不要修改或中断第一位师兄的任务。本文件是第二轮的唯一运行交接入口。

## 给 Claude Code 的执行指令

你负责把下面的独立实验从环境准备执行到最终结果打包。用户已授权在分配的 Linux
服务器下载公开数据、模型及容器镜像，配置独立环境，运行验收、NICE、SFT 和主评测。
先读 `experiments/scaled_nice/AGENTS.md` 和 README，再按本文件执行。
常规环境、依赖、下载与可恢复工程问题自主修复并留下记录；不要只输出计划让师兄逐条操作。
不修改第一轮，不提交 Git，不根据结果更换任务或训练配置。协议、模型或数据规模发生
实质变化时保留旧证据并换新存储根目录，不能修改 snapshot 伪造续跑。

**工作目录固定为 `experiments/scaled_nice/`，不是仓库根目录。**
这是源代码隔离的第二套项目，模块名仍为 `nice_validation`，必须创建独立 venv，
不得复用第一轮的 editable install 或同一存储目录。复制仓库到另一台服务器即可；
不需要从第一轮拷贝模型、数据、缓存或结果。

## 选定配置与理由

选择 **4×72GB** 这一组，不同时占用 8×48GB。GPU 具体型号、互联、CPU、磁盘及
Docker 性能未知，不能仅凭显存容量承诺运行速度。72GB 为 32k 上下文的 BF16 LoRA
及梯度重放留出较多空间；最终以实际最长序列 smoke 为准，不能保证任何 72GB 卡都通过。

模型**暂定**为 **`Qwen/Qwen2.5-Coder-7B-Instruct`**，官方 `AutoModelForCausalLM`，
BF16、SDPA、gradient checkpointing。它的官方 config 声明 `Qwen2ForCausalLM` 和
32768 最大位置长度。代码先检查 config→loader 映射，再加载本地已冻结权重。
不默认量化，不用非官方远程代码，tokenizer 与模型使用同一冻结 revision。
用户将最终模型选择交给师兄端处理。请 Claude 根据实际资源、模型工具调用能力和
工程验收确定最终模型，并在正式启动前记录选择理由；配置中的 7B 是可执行的暂定起点。
若要换模型，在第一次启动前通过参数指定，并确认该模型的正式 loader、LoRA 模块、
工具调用模板和完整上下文容量；不要拿不兼容架构直接替换 ID 后强行训练。

| 项目 | 本轮定义 |
|---|---|
| 训练原始数据 | `SWE-bench/SWE-smith-trajectories`，`tool` split，全 split 下载 |
| 配套任务集 | `SWE-bench/SWE-smith`，`train` split，全 split 下载 |
| 候选池 | 清洗、tokenize、去重和泄漏排除后 3000 条完整轨迹 |
| 主验证 | 按任务持出 100 题；另选 1 题仅开发验收，均不进训练 |
| NICE rollout | 每题 4 次，共 400 次，使用初始 adapter |
| score 分层 | 对候选分数升序排列，三等分，每层 1000 条 |
| 子集 | low/middle/high/random × seeds 11/22/33，共 12 份，每份 150 条 |
| 选样比例 | 每分支占候选池 5%；三个分层内部的抽样比例各为 15% |
| token 控制 | 各分支监督 token 总数满足共同预算 ±10%；不可通过截断凑数 |
| SFT | 一遍、effective batch=1、每分支 150 更新，共 1800 更新 |
| 优化器 | AdamW，恒定 lr=1e-5，betas=(0.9,0.999)，eps=1e-8，weight_decay=0，clip=1 |
| LoRA | q/k/v/o_proj，r=16，alpha=32，dropout=0；12 个最终 adapter |
| 主评测 | initial + 12 分支，各 100 题，共 1300 episode |
| 正式 episode 总数 | NICE 400 + 主评测 1300 = 1700；不含 smoke 和 infra retry |
| 外部四套 benchmark | 全部关闭，不下载任务集，不执行正式评测 |

5% 参考 NICE 官方示例脚本的选样比例，并非声称原论文所有实验统一用此比例。
本项目依然是单初始 checkpoint、每题 4 rollout 的 NICE 风格变体；官方示例包含
warmup 后多个 checkpoint 和 MC=20，不能把本轮描述为原论文完整复现。
上述学习率是当前统一实验起点，没有声称是 7B 模型的最优超参。

训练和验证方向都只对 **LoRA 可训练参数**求梯度，底座冻结；评分不更新权重。
正式训练的 12 个分支从同一初始 adapter 独立开始，互不接续。使用四张训练卡是
同时训练四个不同分支，不会把每个分支的 batch 改为 4，也不是 DDP/FSDP。

## 新交互协议

本轮 `agent.max_actions=75`，一次模型回复计一个 action，回复中的工具调用可能有多个。
还同时限制每回复最多 2048 新 token、每 episode 合计 16384 新 token、上下文 32768、
交互墙钟 1200 秒、单工具 120 秒、官方判分 1800 秒。先触及哪个预算就按哪个结束，
75 是上限，不意味着每题都能使用满 75 轮。

`budget_exhaustion_policy=grade_patch`：ACTION/WALL/TOKEN/CONTEXT_LIMIT 时，
若工作区已有非空补丁，自动送官方测试判分，保存 `automatic_submission`、
`patch_sent_for_grading`、补丁字节数与 SHA256，同时保留原 `stop_reason`。
没有补丁仍失败；工具格式错误、工具超时等不因此自动变成成功。自动送测也不代表
测试通过，`resolved` 必须来自官方测试。NICE reward 和全部主评测使用同一规则。
第一轮的严格提交规则和 30 轮上限完全保留在父项目。

这是为避免已有正确补丁仅因没有显式结束而丢失的预先协议调整。训练数据量、模型
与执行预算均有变化，不能凭两轮差异判断究竟是哪一项带来改进。

## 启动与 GPU 调度

Linux、Python 3.12+、Git、可用 Docker daemon、可访问 Hugging Face/GitHub/容器镜像源。
需要服务器账户具有相应目录和 Docker 权限。代理环境要放行 loopback；程序会为服务
子进程加入 `127.0.0.1,localhost` 的 NO_PROXY。若公开上游被网络拦截，记录实际错误，
修复连通性后继续；不要假定“能上外网”就等于所有大文件和镜像源可用。

示例路径可按服务器实际磁盘调整，保持为全新的专用目录：

```bash
cd /path/to/NICE-Subset-Validation/experiments/scaled_nice
python3.12 -m venv /data/venvs/nice-scaled-v1
source /data/venvs/nice-scaled-v1/bin/activate

# 默认模型、4 张当前可见 GPU、端口 8875—8878。
python scripts/run_scaled_server.py \
  --storage-root /data/nice-scaled-v1 \
  --execute-server
```

入口自动安装本目录的 `[server,dev]` 依赖及冻结 commit 的 SWE-bench/SWE-smith harness，
解析并冻结公开源版本、下载原始 split 和模型、生成绝对路径配置、准备任务镜像，
执行 preflight/smoke，再执行正式实验。无需手动把原始数据拷进 Git 项目。
基础 Python/OS/Docker 若不存在或无权限，Claude 可以先安装或报告具体权限阻塞。
若服务器采用独立 CUDA PyTorch 软件源，应先在该 venv 安装兼容的 CUDA 构建，
检查 `torch.cuda.is_available()`，再启动入口；不能用 CPU wheel 冒充 GPU 环境。

自定义模型、GPU 池或端口：

```bash
python scripts/run_scaled_server.py \
  --storage-root /data/nice-scaled-custom \
  --num-gpus 4 \
  --gpu-uuids GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
              GPU-yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy \
              GPU-zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz \
              GPU-wwwwwwww-wwww-wwww-wwww-wwwwwwwwwwww \
  --model-id Qwen/Qwen2.5-Coder-7B-Instruct \
  --model-loader AutoModelForCausalLM \
  --port 8875 \
  --execute-server
```

示例 UUID 是占位，真实值来自 `nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv`。
使用完整 UUID。不给 `--gpu-uuids` 则从当前 `CUDA_VISIBLE_DEVICES` 允许范围发现前 N 张；
卡数至少为 2，可通过 `--num-gpus` 改变。不要指定本实验无权使用的卡。
默认 4 卡调度如下（GPU0 指选定池的第一张卡）：

| 阶段 | 占用方式 |
|---|---|
| 下载、准备、初始 adapter | GPU0 按需加载；尚无推理服务 |
| smoke | GPU0 做最长上下文训练/梯度验收，GPU1 做开发题推理；其他服务惰性加载 |
| NICE rollout 与 initial 主评测 | GPU0—3 四服务，一卡一模型副本、各一路任务 |
| NICE 梯度评分及选样 | 先退出全部推理服务；GPU0 评分，其余空闲 |
| 12 分支训练 | 四卡并行，每卡一独立分支，三批；无推理服务占显存 |
| 12 分支主评测 | 重启四推理服务，逐模型四路并发；分支之间隔离 adapter |
| 报表、审计、压缩 | 停止全部自有推理服务后执行 |

服务必须由本控制器创建并拥有，以便阶段切换释放显存；不复用外部现存服务，
端口被占时换新的连续空闲端口。不要手动启动 `serve` 后再跑入口。
不要用父目录的 `run_server.py`，也不要绕开控制器直接执行真实 `run-online`。

## 数据质量、验收与失败处理

下载的原始 split 可以大于最终使用量；3000 是清洗后的候选数，不是下载上限。
全轨迹保留，最长 32768 token，超长样本按既定规则排除，不静默截断工具调用。
按任务 ID、源信息及重复内容检查训练与验证/开发泄漏；可用样本不足、文件为空、
tokenizer/工具消息非法、分层不可行或 token 预算不可满足时明确停止，不能复制补齐。
源 revision、下载内容 checksum 和本地导出路径进入冻结证据。

开发题不参与主结果，只检查多轮消息→工具→容器→补丁→官方测试链路；验收不要求
模型一定解对，但要求框架、判分和失败归因有效。完整 32k 前反向与单步更新必须
实际通过才能继续。OOM 时保留日志，可修复释放服务、依赖或设备绑定；若必须改
上下文、模型、量化或训练定义，换新存储根并在报告说明，不能悄悄降配续同一运行。

NICE 全部方向为零时阻止评分；低 reward coverage 会告警，并在结果报告实际有效任务数。
不要因为 7B 初始成功数少而按答案筛选“容易题”替换。固定 100 题共享给 initial 和
12 分支，置信区间和逐题配对结果一起看，不能把 12 个重叠子集当 12 次独立复现。
如果数据/模型选择确实不支持形成方向，报告阻塞证据并结束，而非伪造完整结果。

训练由一个父协调器写全局 state，每个子进程只写自己的分支与 receipt；receipt 记录
实际 GPU、源身份、训练产物和 worker 日志 checksum。父进程等待整批退出并验证后发布
各分支完成状态。普通失败会回收其余子进程。恢复前检查孤儿进程，尤其是控制器被
SIGKILL/节点故障的情况；仅解除过期锁不能证明 GPU 子进程已经退出。

保持相同源、依赖环境、数据、模型和参数重启时，可使用相同命令加 `--recover-lock`。
程序只回收已确认同机 PID 不存在的锁；已完成阶段会校验产物后跳过，未完成 SFT 从
保存的 adapter/optimizer/RNG/数据顺序恢复，中断 episode 在干净容器重启。
不要手工编辑 state、receipt 或 checksum。基础设施错误最多按既定规则重试两次；
模型本身失败不重试刷成功率。不能因恢复失败删除原证据。

## 存储、日志与交付

`--storage-root` 下统一保存：

- `data/`：原始导出、源版本锁及环境映射；`cache/` 保存 Hub/datasets 下载缓存。
- `models/base/`：固定 revision 的模型与 tokenizer；`models/initial_adapter/` 为共同初始 LoRA。
- `runs/main/`：config 快照、manifests、rollouts、scores、training、evaluation、CSV、图和审计。
- `controller/`：各命令 stdout/stderr/时间记录、服务日志、完成状态和最终压缩包。

以实际生成配置为路径权威。模型配置写入本项目 `configs/scaled-server/*.local.yaml`，
全部关键数据/模型路径为绝对路径，不能拿示例目录替代生成配置。
在线下载包括 Python 依赖、两个原始数据 split、7B 权重/tokenizer、SWE-bench/SWE-smith
官方源码和所选任务容器镜像。主流程不拉取四套外部任务快照，不创建 Live 独立环境，
也不从 Git 安装 Harbor。通用 server 依赖仍包含 Harbor Python 包，但不启动其
benchmark 任务；不能把“关闭 benchmark”理解为零网络访问。
需预留下载缓存、解压后模型、Docker 镜像层和训练 checkpoint 空间；磁盘实际占用通过
下载计划、`df -h`、`docker system df` 检查，不把 7B 权重大小当作全部存储需求。
清理只能针对本运行可重建的缓存/自有容器，保留正式证据，不能全局 prune 别人的镜像。

成本记录见 `logs/events.jsonl` 和 `tables/stage_costs.csv`；训练子日志另在
`logs/parallel-training/<subset>/<attempt>/`。token 为真实调用记录的逻辑 input/output，
不是 FLOPs，也不包含 checkpointing 重算。分支独立进程记录实际训练 GPU UUID；
父进程发布阶段不重复声称占用训练卡。

完成后必须检查：审计无 FAIL；12 个分支各 150 更新；候选 3000、主验证 100；
400 NICE rollout、1300 主评测、正式总数 1700；未出现 transfer 正式任务；模型身份一致；
全部分支 adapter 和逐题结果可追溯；mock 没有混入正式指标。缺失指标用 null。
100 题每多解一题为 1 个百分点。报告“同目标集上的选样效果”，不能宣称迁移有效。
主验证被用于 NICE 评分，不是独立测试；日后追加 benchmark 需要重新核验与全部训练
和开发任务的交叉泄漏，不能默认旧轮的 200 个任务仍然独立。

控制器成功后自动生成：

1. `<storage>/controller/results-<UTC时间>.zip`
2. `<storage>/controller/completion.json`，其中 `status` 应为 `completed`。

师兄默认只需把这两个小文件发回。zip 含报告、CSV、图、配置与 ID 清单，不含底座、
12 个 adapter 权重、optimizer checkpoint、全量数据和完整轨迹。权重与原始日志留在服务器；
若要继续分析特定失败，再按任务导出 trace。若执行失败，返回具体失败说明和路径，
不要交付伪称完成的 zip。

## 已核对的外部依据与本地验证边界

- [Qwen 官方模型](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
  与 [官方 config](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/raw/main/config.json)。
- [NICE 官方示例](https://github.com/JTWang2000/NICE/blob/main/running_commands.sh)：
  5% 选样、多 warmup checkpoint、MC=20；本项目不冒称完全相同。

本地全量 CPU 回归 **249 项通过**，包含微型随机 Qwen2 模型的正式 Transformers/PEFT
接口测试；完整 3000/150/100 显式 mock、恢复和压缩包检查也已通过。
真实 7B 权重、四张 GPU、Docker 任务和公网下载
完整性仍需上述服务器验收；不能把本地通过解释为服务器已真实训练成功。
