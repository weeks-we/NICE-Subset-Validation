# Claude Code评测复核：服务器部署编排

最新接入已改为Python Claude Agent SDK；构建/复用/启动优先阅读 [SDK部署说明](reevaluation/DEPLOYMENT_SDK.md)，不要按下文历史CLI命令描述直接开跑。scripts/smoke_reevaluation.py已切换SDK，实际容器入口为/opt/nice-sdk/bin/python。

本文件补充 docs/HANDOFF.md 的2026-11-12条目。最新完整SDK执行入口为 `scripts/run_reevaluation.py`，按 [SDK部署说明](reevaluation/DEPLOYMENT_SDK.md) 的preflight/run命令执行。包含八卡服务、600题调度、逐题恢复、预算监督及打包；必须经过真实服务器验收，不以旧run_server.py代替。

## 代码、旧数据与新结果

- 新代码clone到独立目录（或确认干净、停用的独立工作副本后更新），不要直接覆盖正在执行或用于旧成果溯源的代码。用户自行commit/push后才能取到变更。
- 原storage/runs/main、模型、五份adapter和原任务镜像只读复用；新部署输出与旧run不得相同、互相嵌套。不编辑旧snapshot、stage或checksum，不复制旧正式结果作为新结果。
- 原冻结Verified100题＋NICE20题、initial和原四个selection.json选中adapter，共600次。无需SWE-smith100题重选权重或重新生成NICE160。
- configs/reevaluation.yaml是独立评测参数清单，不与原snapshot合并。vLLM固定0.22.0、Claude Code固定2.1.153，禁止latest和运行中更新。参数来源、工具范围与服务器填写边界见 [固定重测协议](REEVALUATION_PROTOCOL.md)，研究参数不交由服务器自由选择。

## 第一步：盘点，优先复用

2026-11-12增强：盘点必须核对原 initial/identity.json 与各 checkpoint.json 的 adapter_sha256，核对训练 run/model 身份及 initial 的基座锁身份；基座锁不得为空或只含 config，必须覆盖 tokenizer、权重及索引引用的所有分片。缺证据时阻止，不在旧目录重新生成锁冒充原证据。

在拥有原storage及原Docker daemon访问权限的Linux环境执行：

```bash
python scripts/prepare_reevaluation.py \
  --old-run /原storage/runs/main \
  --output /新storage/deployment \
  --execute-server
```

默认不授权时仅打印计划，不触碰Docker/GPU。执行时读取原snapshot、selection、两套完整任务manifest及environments/nice、benchmark的task_images和lock；核验基座锁内文件、五份adapter文件、原image map及120题本地Agent/grader镜像/官方标签。生成resource_inventory.json，缺项集中列出，不pull、不构建、不执行模型，不写旧目录。adapter散列是本次读取指纹，仍须与原交付证据核对，不把“存在”当作原训练真实性证明。跨节点时同一镜像可能不可见：根据旧记录从内部仓库/镜像导出恢复同内容，不重新拉公网latest顶替。

原120题镜像通常已具备；它们没有因新clone失效。新的Claude Code程序及运行库仍需提供：不要在正式任务里在线npm安装。可为原镜像制备派生运行层或挂载隔离的只读CLI运行目录，保持原仓库checkout、Python/conda测试环境和官方grader镜像不变。验证CLI可在原镜像的glibc/架构下执行；不能假设宿主机二进制复制过去必定可用。Claude Code无自动批准时会卡住、root下绕过权限可能受限，这些要在无人值守smoke中解决，不临时给主机全权限。必要的可写会话目录仅供该任务使用。

## 第二步：运行环境和推理服务制备

独立制备兼容vLLM0.22.0的CUDA/PyTorch镜像，不在旧训练venv内升级。提前准备Claude Code固定发行物、模型缓存、依赖、任务运行层、官方grader和公司证书。官方CLI许可按平台要求核验。构建阶段可访问受控外网，正式计算只用已有资源。

以原固定BF16基座加载rank16 adapter，工具解析qwen3_coder、思考解析qwen3。同一时段五个模型之一占据各服务；所有Claude主请求、辅助请求必须路由同adapter，不使用云端小模型。initial仍挂原共同初始adapter。实际生成参数按配置草案，262144总窗口、32768单次输出、thinking/history thinking开启；后端参数、代理转换及CLI最终请求需捕获核验。不能只靠CLI界面显示的模型名称证明路由正确。

默认8卡、TP1/每卡1槽；262K真实容量不够则先降低并发或验收TP2布局，不悄悄量化/缩上下文。GPU UUID从平台授权分配读取并落盘。此处只评测，无optimizer.step、无NICE梯度。

接入依据：[vLLM 0.22.0 Claude Code](https://docs.vllm.ai/en/v0.22.0/serving/integrations/claude_code/)、[同版本模型支持表](https://docs.vllm.ai/en/v0.22.0/models/supported_models/)、[实际评测CLI适配示例](https://github.com/Tencent/workbuddy-bench/blob/main/src/workbuddy_bench/agents/cc_agent.py)。不把第三方项目的默认值冒称Verified官方协议，也不保证未经验收的CLI版本组合可用。

## 第三步：替代network none，先验收受限内网

原DockerTask和其network=none不修改。新Claude Code任务需要调用本地推理服务，故用**单独专用网络**，不使用host网络或Docker默认bridge。专用bridge本身不限制外网，必须配合平台NetworkPolicy、防火墙或受控网关：只允许任务容器访问指定推理IP/端口和必要DNS，禁其他外网及任务容器之间访问。规则及验收证据保存在deployment目录。agent不能伪称加了一个Docker网络就实现了隔离，也不擅自修改公司防火墙；缺权限先报告平台管理员需做的最小动作。

单节点优先用明确的宿主机bridge网关/服务地址（容器内localhost不是宿主机）。跨节点用公司提供的私有service DNS/IP，确保任务容器可访问；计算节点地址随作业变化时由部署发现并冻结到本次配置，不写死他人服务器IP。

应用层对推理端点取消HTTP_PROXY/HTTPS_PROXY/ALL_PROXY及小写变量，NO_PROXY/no_proxy写入服务主机名；容器构建所需代理与正式推理代理配置分离。清代理不等于网络路由一定可达。公司若只提供代理出口但内部通信正常，不影响模型请求；若跨节点流量被禁止，应使用平台允许的服务/网关或同节点布局。网关若要求TLS和认证，用公司CA与凭据，不能关TLS验证。当前探测不传凭据，仅检查通常公开的health，需认证的health由agent按安全机制补验，不把密钥塞URL。

用同一个原任务镜像，**从容器里面**验收（endpoint为服务根地址，不带/v1）：

```bash
python scripts/prepare_reevaluation.py \
  --old-run /原storage/runs/main \
  --output /新storage/deployment \
  --probe-image sha256:已存在任务镜像内容ID \
  --network 公司已创建并限制出口的专用网络 \
  --endpoint http://实际私有推理服务地址:8000 \
  --execute-server
```

脚本不会新建网络或修改防火墙，不挂模型/数据，使用pull=never；在目标容器里解析DNS，拒绝公网/loopback/不确定地址，再绕过HTTP代理请求/health，失败落盘。network_probe.json PASS仅证明DNS/TCP/HTTP健康检查，不证明工具调用、LoRA、thinking、隔离或完整600题评测。所有实际服务端点均须探测，而非只检查一张卡。还应实测一个禁止的公网请求失败、其他任务不可达，保存网络策略证据。预检失败不分配正式评测任务。

## 第四步：执行器接入验收清单

### 已提供的真实CLI/工具smoke

smoke要求至少三条assistant事件、Read/Write/Bash及随机工具反馈。完整入口另做真实开发题、LoRA数值、容量和压缩验收；不能仅凭smoke PASS开始正式评测。

Claude Code是下载并安装的程序，不是远程Claude模型；固定发行物、依赖和必要运行库在公司构建阶段装进本地任务运行层，正式计算不在线安装。新入口scripts/smoke_reevaluation.py支持在授权Linux服务器进行真实接入验收（本机只做显式mock/CPU测试）：

```bash
# 用平台secret注入ANTHROPIC_API_KEY和ANTHROPIC_AUTH_TOKEN，切勿写进Git。
python scripts/smoke_reevaluation.py \
  --image 已提前准备且包含python和claude的本地镜像 \
  --network 已限制出口的专用网络 \
  --endpoint http://实际私有推理地址:8000 \
  --model 当前服务实际adapter别名 \
  --claude-version 2.1.153 \
  --output /新storage/deployment/smoke-唯一编号 \
  --execute-server
```

先做容器内DNS/health检查，再以非root用户运行真实CLI（仅在无敏感挂载的隔离容器内自动批准工具），核验CLI版本及真实工具调用事件，并读取随机内容、写文件、核对结果。所有模型别名指向同一指定服务模型。8192单次输出、12轮和600秒是短工程测试预算，不替代正式32768/250轮/7200秒。保存stdout、stderr及status.json；FAIL退出非零，保留证据，不进入正式任务；清理仅限本次随机命名容器。不要直接把旧任务镜像当作已安装CLI的运行层，先验证UID1000可执行CLI且拥有独立可写会话目录。需要企业证书的场景，预装受信CA，不关TLS校验。

每个实际推理服务都要测；五份模型（含initial）都要在最终路由下测。status PASS只证明这条CLI/模型/文件工具链，不证明adapter权重实际不同、thinking保留、262K容量或官方判分正确。相同输出也不能单独证明LoRA没加载，反之输出不同也不能单独证明加载正确。

### 正式任务前的硬性关口

configs/reevaluation.yaml的acceptance节是必须实现并通过的要求清单，不是已通过回执。后续600次执行器必须读取并核验下列实际证据；目前准备/smoke脚本不会启动正式任务，也不提供“跳过验收直接开跑”的选项：

| 关口 | 通过条件 | 失败处理 |
|---|---|---|
| 固定软件与资源 | vLLM精确版本、CLI版本、原基座revision、五份adapter及120题清单/镜像身份均可验证 | 输出缺项或不匹配项，停止 |
| CLI/工具接入 | 新smoke通过，并核实请求实际落到指定服务及模型名 | 停止，保存客户端和服务端日志 |
| LoRA权重 | 核对rank16及q/k/v/o目标映射、实际加载权重；用同一固定输入/模板的HF PEFT与vLLM比较可比的logprob/输出，记录容差依据，不只看模型名 | 映射/数值差异未解释则停止，不自动合并/量化换方案 |
| 生成协议 | 捕获真实请求和服务有效配置，thinking及历史保留、temperature1/top_p.95/top_k20/min_p0/penalties均生效 | CLI或协议覆盖参数时修正并重测，不静默用默认值 |
| 长度和资源 | 输入+输出262144，单次输出最多32768；在声明的TP及任务并发下做真实推理容量验收 | 优先降低并发/验收TP2；仍失败则反馈，不静默缩预算 |
| 真实开发题 | 独立原开发题完成模型→工具→仓库修改→补丁→原官方grader；核验环境与参考补丁的判分链路，禁止替换正式题 | 工程链路失败停止；模型未解出题须与环境失败分开，不伪造成功 |
| 安全网络 | 目标端点可达，禁止公网和任务横向访问；无外部模型或模型名fallback | 保存实际策略/测试证据，停止 |
| 停止与恢复 | 轮数、墙钟、累计输出、工具时间和预算末补丁提取真实触发；中断后可恢复，不能重复记成绩 | 停止修实现，不把配置字段存在当限制已执行 |

仅改变checkpoint不改变这套评测参数。任何关口失败，agent返回“失败关口、命令、错误、证据路径、可行修复”，不启动600次评测。基础设施重试有上限，不能无止境尝试或重试模型失败。容差和并发在正式计算前决定，不能根据benchmark表现倒改。

配置复核：原模型/五份权重/120题固定；vLLM0.22.0，Claude Code固定2.1.153，禁自动更新；BF16、rank16；thinking开启、保留历史；262144上下文、32768单次输出、262144累计输出；250轮、7200秒交互、1200秒工具、3600秒独立判分；8卡TP1/单槽为起点，容量不够先验收TP2；500 benchmark + 100 NICE诊断，均采样单次，不重选checkpoint。新上下文策略用CLI原生管理，不叠加原五观察裁剪。不假定Claude effort枚举会自动成为Qwen xhigh，记录实际支持和映射。

在600题执行器编写/接入时逐项落实，不让agent只运行Hello就开始正式评测：

1. 固定原任务和镜像身份，每题全新工作区/会话；不挂原数据、参考答案、其他题目或权重到Agent容器。只有推理服务能读取模型文件。
2. Claude Code实际模型请求包含正确adapter名；验证五份模型及基础PEFT接口，确认无静默fallback、错误参数映射或外部模型调用。
3. CLI工具读/搜索/编辑/测试/结束链路成功，完整记录thinking与tool-use反馈，采用CLI原生上下文管理，不叠加原五观察裁剪。
4. 固定250轮、7200秒交互、1200秒工具上限、262144累计输出（含辅助请求），确认对应CLI版本支持何种控制；没有可靠计数器时不能声称限制已实现。禁子agent/人工询问/网络搜索，不允许外部答案访问。
5. 用原官方判分工具和grader镜像，判分3600秒独立计时。预算停止前保存容器工作区，提取非空补丁送判；停止原因不改写。正式模型失败不重试刷分，基础设施错误有限重试并留全部证据。
6. 先使用独立开发题完成端到端smoke，资源/协议未通过不开始正式题。新的开发smoke只属工程验收，不重算原NICE或训练。
7. 新结果目录独立冻结代码/版本/环境/参数/任务/adapter/网络身份。每个episode原子提交结果及checksums；恢复跳过新协议已完成任务，不能复用旧协议成绩。完成500+100任务才汇总，两个scope分开，导出失败及成本明细和结果zip。

## 给师兄agent的简短指令

本地准备层验证：370项CPU/微型模型测试通过，其中新增17项资源与网络准备测试；新完整2060次mock、串行评分对照、恢复、audit、CSV/PNG和zip检查通过，6张图与此前核验版本字节一致。记录保存在忽略目录runs/reeval-deploy-*20261112*。这些是旧流程回归和新准备层验证，**不是Claude Code/vLLM真实接入或新600次评测验收**；本机没有运行Docker、目标权重或GPU。

补充CLI smoke后的全量回归为376 passed（285.94秒），含6项新增CLI命令/凭据/固定版本检查；新完整2060次mock、串行评分、恢复和结果包检查再次通过。记录为runs/reeval-smoke-pytest-20261112.log、runs/reeval-smoke-verification-20261112.log及runs/reeval-smoke-full-20261112/validation_checks.json。真实CLI、模型、LoRA、容量及官方判分未在本机执行，仍需服务器逐关验收。

读取docs/HANDOFF.md的2026-11-12补充及本文件。使用独立新代码和评测输出目录，只读复用原模型、五份adapter和120题镜像。先运行prepare_reevaluation.py盘点，只补缺失的CLI/依赖/镜像层；制备阶段完成网络、代理、端点和真实小样本验收。旧原实验源码及snapshot不改，不重训、不重算NICE、不重选权重。现有入口只做准备，不是完整评测器；按第四节完成并验证评测执行器后再运行600次新协议评测，不能用旧run_server替代。公司权限阻塞列明最小配置需求，不凭猜测开跑，不commit/push。
