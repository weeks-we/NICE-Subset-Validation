# 师兄B：独立服务器从零实验

## B硬件覆盖：8×H20

用户已确认B使用8张NVIDIA H20，不能继承A的H200-144GB显存与吞吐假设。服务器agent先用nvidia-smi核验8张实际可见且获授权的卡、单卡显存、驱动、互联及UUID；不从“H20”名称猜单卡显存。构建CUDA扩展的架构必须读取实际device capability，通常Hopper为9.0，但须实测后显式传TORCH_CUDA_ARCH_LIST，不再使用Dockerfile默认值。

完整模型、数据、harness和221任务环境的制备范围不变。B以环境诊断为目标，不要求达到A的32并发吞吐。GPU短测从单请求、单任务槽开始；BF16基座+LoRA训练需在本机真实验收。加载失败或OOM时先报告显存/日志，不能擅自量化底座、缩短正式上下文、改学习率或有效batch来宣称原协议通过。可将更小输入的工程检查作为单独标记的测试，但正式长度容量必须另列NOT_TESTED。

vLLM先验证TP1单槽，完整上下文容量不足时可在正式计算前配置TP2并重新验收，须有至少两张授权卡；实际卡数/显存不足就报告阻塞，不借用不存在的卡。原HF/PEFT训练入口不因vLLM支持TP就自动支持跨卡训练；多卡训练分片若需要，属于另行实现验证的工作，不用device_map冒充已实现训练并行。CUDA版Torch/内核选择以H20实机驱动及构建栈为准。

下文H200、八卡、每卡4槽等数字仅为A历史/未来配置背景，不作为B短测执行参数。B的机器事实与工程测试配置保存在独立诊断目录，不修改A的main.yaml或reevaluation.yaml。

## 最新范围（覆盖下方先前的“仅少量环境”限制）

本次需要**全量环境制备＋短期训练/测评诊断**，不是只准备一题。下文“无需全部任务镜像”“只取少量数据”“一题即可”的旧范围由本节替代；仍禁止无条件启动完整实验。

### 全部准备完成后才开始短测

- 下载并锁定完整基座/tokenizer和三类完整数据split，保留revision、完整条数、来源sidecar和checksum。诊断样本从完整数据中取，不能用部分下载冒充完整来源。
- 按既定划分规则清洗并冻结2000候选、20 NICE、100 Smith验证、100 Verified benchmark和1开发题；覆盖全部221个独立任务，保存ID/来源家族/近似题面隔离证据。冻结不代表完成正式NICE或训练。
- 预备全部221题的public clean工作环境及对应官方grader，生成“任务ID→工作镜像/判分镜像/digest/commit”清单，内部仓库发布后在计算节点提前load/check，官方标签可解析。共享镜像可去重，不要求221份物理重复内容。
- 主训练/判分运行镜像、独立vLLM0.22.0、SDK0.2.87+Claude2.1.153工具箱，以及全部221题的SDK派生层均制备并固定digest/父镜像ID。SDK层只用于工作环境，grader保持原环境。依赖、模型、任务镜像的联网动作全部在制备阶段完成，计算短测时禁止补下载。
- 数据冻结/初始LoRA及Run身份有GPU前置依赖时，安排明确的制备验收步骤；现有prepare-online不等于纯下载入口。服务器agent须先核对入口行为，必要时在B独立诊断目录实现阶段适配，不直接启动整条run_server流水线。

### 全量检查＋有界短测

1. **全部镜像检查**：逐任务核对digest、base/checkout commit、public clean标记、工作区洁净、无答案/判分脚本泄漏；检查testbed解释器可执行、SDK程序/包版本、UID权限、挂载与内部服务可达性。结果按221题逐项记录，不能以抽测代替镜像清单完整检查。不要为此启动221个GPU服务。
2. **训练短测**：从清洗候选固定取8条样本，执行最多2次有效batch=8的LoRA优化更新；若需要第二次曝光显式记录样本复用，仅作工程测试。保存/重载临时adapter和优化器/RNG状态，做恢复对照；检查底座冻结、LoRA梯度有限且非零、loss/mask/token长度、累积batch正确。正式候选和训练权重不能被这些临时更新污染。
3. **NICE短测**：固定2条候选及2个开发诊断任务做少量梯度/投影检查，最多各1次rollout；评分过程不optimizer.step，检查零方向、有限值及串行/分片一致性。诊断任务必须明确用途，不能用benchmark模型表现选配置。未完成全部2000评分/160 rollout，不得标记正式NICE阶段完成。
4. **任务测评短测**：在看到输出前，按数据集/仓库分层固定最多8题（覆盖Smith与Verified），每题只运行一次agent→工具→补丁→官方判分；协议预算保持，20分钟仍未完成的题可作为明确engineering timeout停止，不计正式成绩。官方判分另用参考补丁正例/空补丁负例验证，参考答案不得进入工作容器。模型答错与环境失败分开报告。
5. **SDK/vLLM短测**：使用临时初始/训练adapter验证路由、实际thinking/采样、Read/Write/Bash随机反馈和usage；短测不要求存在五份正式checkpoint，不伪造selection.json。若准备正式每卡4并发，必须独立测试4请求满上下文容量、显存/缓存抢占、延迟及主机RAM；未跑该压力测试标NOT_TESTED，不能把短输入通过当容量通过。使用工程探针，不凭benchmark resolved调参。
6. **恢复和收尾**：至少中断恢复一个诊断任务和一次短训练，检查不重跑已完成项目、不会误用临时权重；检查CSV/PNG/zip。完成上述短测即停止本次服务，保留全部制备资产。全部诊断输出置独立engineering目录，不写正式研究完成标记。

报告environment_diagnosis.md及逐任务环境CSV：全部资源准备数量、镜像/资产checksum、PASS/FAIL/NOT_TESTED、硬件及峰值RAM/显存、测试命令、停止原因、修复记录、仍缺项。任何模块未实测不可宣称通过。遇到环境问题优先修复后重测同一工程样本，同类基础设施最多两次重试；模型失败不挑结果重试。**不跑完整NICE、四分支正式训练、12节点选择或600题正式测评。** 后文完整实验流程仅作为未来背景。

## 当前任务覆盖：仅构建与短程环境诊断

**本次只授权打一份主运行镜像并做短程验收，不运行完整实验。以下完整实验设计仅作依赖与后续工作背景；不得直接执行完整 run_server.py、run-online 或 SDK 600题 run。** 不需要下载全部任务镜像或预跑正式NICE/选样/SFT/benchmark。诊断完成即停止，保留报告和镜像供后续使用。

执行范围按以下顺序，失败模块报告原因、日志和最小修复，不用正式实验不断试错：

1. **构建主镜像**：按BUILD.md解析B自己的依赖锁、固定基础镜像和两个harness commit，构建本目录Dockerfile。检查pip check、关键import、Python/CUDA版torch及来源信息，保存digest和完整freeze。没有旧环境也能新建，不能要求A提供原锁。
2. **平台连通与挂载**：确认主容器能看到授权GPU、实际可写资产/结果路径、同一个Docker daemon；主容器调用Docker创建一个无GPU小容器并清理。只用平台批准权限，不全局prune。
3. **资产最小准备**：先核验模型/数据源连通；为真实加载测试下载固定revision的完整基座/tokenizer。数据先取少量真实tool轨迹、Smith任务和Verified任务检查结构/对应ID，明确标记诊断样本，不能写成正式完整来源锁或混入后续正式数据。无需为本次诊断下载所有任务镜像。
4. **GPU短测**：在一张明确授权GPU加载基座，初始化全新临时LoRA；对几条真实清洗样本执行前向、loss、backward，检查底座冻结、LoRA梯度非零且有限、padding/工具观察不监督；少量optimizer更新并保存/重载临时adapter，验证加载接口与数值正常。接着验证NICE评分不执行optimizer.step及投影维度/有限值。所有输出标为engineering-only，不能当正式研究结果。出现OOM或内核回退记录实际显存/后端，不用缩小正式预算掩盖。
5. **单任务环境**：准备一题独立开发任务的public clean工作镜像和官方grader，核对commit、工作区、testbed解释器及工具执行。用参考补丁正例和空补丁负例检查官方判分（参考答案不能进agent容器）。可选再做一题Verified，以覆盖Smith与Verified两种环境；不以模型是否解对作为环境是否正常的标准。
6. **SDK/vLLM接缝（具备资源才做）**：按../reevaluation/构建独立vLLM与SDK派生层，不把它们安装进训练venv。使用本次临时adapter在单个服务做健康、模型路由和Read/Write/Bash随机反馈测试，检查thinking和实际usage。不能启动要求五份正式checkpoint的完整重测preflight或伪造五份adapter。缺少该独立层时主镜像验收仍可先完成，报告SDK/vLLM为NOT_TESTED，不能把它计为PASS。不要索取或使用聊天中的DeepSeek密钥。
7. **停止与报告**：停止并清理本次创建的服务/容器，保留镜像、资产和日志。生成 `environment_diagnosis.md`，逐模块列PASS/FAIL/NOT_TESTED、实测版本/硬件、命令、证据路径、问题原因、已修复项及仍缺资源；另保存镜像digest、freeze、临时配置及日志索引。不得写“全流程通过”来代替未测GPU容量/SDK/官方判分。

硬性停止边界：只处理少量诊断样本、一次短训练与单任务测试；不生成正式划分、2000候选梯度、160 rollout、12节点、100题选权重或600题评测。某模块受权限/网络/资源阻塞时记录后继续可独立进行的检查，不无限重试，同类基础设施错误最多两次重试。本次不要求8卡满负载或每卡4请求262K容量；若未实测必须保留NOT_TESTED。

现有一键入口将准备与完整实验串联，不能用于本次诊断。服务器agent可在B自己的诊断工作目录编写最小验收脚本，复用项目函数并核对函数签名；不修改A配置、不伪造研究snapshot或阶段完成状态。测试范围与修复记录必须可复查。

本目录是新的部署交接，不恢复师兄A的运行，不依赖A的磁盘、训练权重或旧环境锁。A的代码/config/main.yaml/交接保持不变。B必须从下载基座、数据、初始化LoRA开始重跑完整实验；不能直接启动要求五份现成adapter的SDK重测入口。

## 实验定义与入口

研究定义沿用仓库 configs/main.yaml：SWE-smith成功tool轨迹清洗后2000候选，NICE升序二等分各随机512、seed11；Verified20题每题8次rollout；另100题Verified最终测试；SWE-smith独立100任务选节点及另1开发题。两组×学习率1e-6/5e-6、有效batch8、一遍512条、128/256/512曝光保存节点，4份选中adapter。原完整训练流程使用原agent和75轮协议，不能把后续Claude SDK的250轮协议偷偷代入NICE/SFT验证。最新SDK/vLLM协议只在B完整训练和选权重后进行独立600次重测。

若用户希望从NICE起就统一Claude SDK，需要另做研究代码适配；本交接没有宣称已实现。不要拼接两个harness的成绩作同一协议比较。

## 目录与阶段

```text
/srv/nice-b/code/                 固定commit源码
/srv/nice-b/build/                联网构建上下文、完整依赖锁
/data/nice-b/assets/              原始数据、基座、下载锁（B自行下载）
/data/nice-b/preparation/         来源revision、冻结任务清单与镜像catalog
/data/nice-b/experiment/          新训练完整storage
/data/nice-b/sdk-reevaluation/    新harness独立结果
/private/nice-b/                 公司路径、GPU UUID、网络配置与凭据
```

### 1. 联网制备阶段（不启动正式训练/评测）

服务器agent先检查硬件和公司Docker/GPU/挂载权限。8×H200-144GB是默认设计，真实硬件不足时正式冻结前报告并确认可行配置，禁止冒充已通过容量。构建节点须能访问公司镜像源、PyPI/受控wheel源、GitHub、Hugging Face及任务镜像来源。

按Dockerfile构建Python3.12 CUDA devel主运行镜像；另按 ../reevaluation/ 的 Dockerfile.vllm、Dockerfile.sdk、Dockerfile.task-sdk 构建后续重测层。依赖版本与来源在构建后冻结，代码不COPY进镜像，挂载固定源码。SDK/vLLM不装入训练venv，任务SDK不升级testbed，grader保持官方固定commit。

B无旧环境：runtime.lock.txt必须在联网依赖解析环境新生成完整精确锁，先由当前pyproject.toml的server/dev约束、CUDA版torch及两个官方harness固定commit解析并做pip check；去除项目editable和swebench/swesmith条目，保留其余精确版本/VCS来源。不能复制A的恢复文档要求的“原锁”，不能把空文件当锁。构建参数见BUILD.md。固定模型/data/harness的真实完整revision，不用main作为最终身份，不编造sha。B无需与A字节相同，但如不同版本必须报告，不宣称复现实验输入完全相同。

数据使用SWE-bench/SWE-smith-trajectories tool、SWE-bench/SWE-smith train、SWE-bench/SWE-bench_Verified test。下载完整所需split，记录实际条数、清洗数量、revision、sidecar/checksum。模型Qwen/Qwen3.8-27B，AutoModelForImageTextToText，固定revision并完整下载tokenizer和权重；不从Git拉大文件。需要至少2101个清洗后独立成功任务，泄漏排除后不足即停止。

**先冻结任务清单，再制备221个任务环境**：20 NICE、100 Smith验证、100 Verified benchmark、1开发。任务ID/来源家族/近似题面隔离；准备public clean工作镜像与官方grader，不能把答案塞入工作镜像。用 scripts/prepare_task_images.py prepare 发布内部digest catalog，计算节点提前load/check；SDK派生层还需原父镜像ID标签。

现有prepare-online并非纯下载：还初始化adapter、冻结Run、准备环境且会调用GPU预检。因此纯制备阶段不能直接运行它并声称“无GPU”。agent先使用下载/来源冻结与CPU清洗入口、核对 --help 和对应函数，形成独立制备清单；Run冻结依赖初始adapter，需在下一阶段小规模GPU预检时初始化后完成。允许单独的“验收资源”步骤，不把它当正式NICE/训练。若平台要求所有任务镜像在GPU节点启动前就完成，需要agent增加独立CPU任务清单制备适配并验证，不能在正式计算时临时联网。

### 2. 交付验收门（正式计算前）

必须交付：源码commit、主运行镜像digest/pip freeze、全部模型/data来源锁和checksum、221题冻结清单及内部镜像catalog、官方grader来源、授权GPU UUID与路径映射。构建/计算节点绝对挂载路径一致，Docker CLI/SDK/grader连接同daemon，预装镜像含官方标签。

运行配置设deployment.preinstalled=true；完成上述任务catalog后设deployment.offline=true并填catalog及其sha256。联网制备配置与离线计算配置分离，必须在正式Run冻结前确定；不得修改已冻结snapshot切换配置。缺模型/数据/镜像即停止，禁止自动外网补齐。

**不要一开始直接运行run_server.py --execute-server**：默认入口整合下载和正式计算，会破坏本交接要求的阶段隔离。先完成上述准备及配置，官方身份、真实前反向容量和开发题官方判分验收通过后再运行。若现有入口无法消费B新制备输入，agent先在B独立配置/适配中解决并测试；不改A的已冻结证据。

### 3. 独立完整计算

使用 scripts/run_server.py 的 --base-config 指向B独立离线预装配置、--storage-root /data/nice-b/experiment、--config-dir 指向B生成配置目录，显式授权GPU UUID及 --execute-server。不传 --reuse-from A路径。具体参数先运行 --help 核对。来源锁预置到B storage，避免离线启动条件缺失。恢复保持同源码/环境/输入身份，不改snapshot绕过失败。

流程：NICE160→训练候选LoRA梯度/投影评分→高低选样→4分支训练/12节点→100 Smith验证选择4份权重→原协议测评/诊断→audit/打包。所有底座冻结，仅LoRA梯度；评分无optimizer.step。推理服务与训练占卡分工按主配置执行，不把独立分片说成DDP。

### 4. 完成训练后的Claude SDK独立重测

以B刚生成的完整run作为 --old-run（通常为storage/runs/main，以snapshot/selection实际位置确认），按 ../reevaluation/DEPLOYMENT_SDK.md 配置B自己的内部网络和派生镜像，执行 scripts/run_reevaluation.py preflight 后run。复用B自己的五份adapter及原120任务，不访问A服务器。默认8服务×4槽、250轮；必须完整并发容量验收，失败停止。此阶段只测评不重训，结果与原协议独立汇报。

## 交付与约束

本地验证不能替代B服务器真实GPU、Docker、模型或官方判分。公司base digest、CUDA/torch组合、锁、任务revision和路径由服务器agent实测确定并记录，不凭文档占位值运行。首轮启动前给出缺项清单，消除后才正式计算。已完成题/阶段不重复；基础设施最多两次重试，模型失败不挑结果重试。不commit/push，不上传权重、数据、凭据或状态。

最终交付原实验和SDK重测各自results zip/completion、简短版本/硬件/协议报告；权重和完整trace留B服务器。失败交具体日志与阶段，不宣称完成。
