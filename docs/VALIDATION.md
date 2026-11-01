# 本地验证记录

## 2026-11-01 追加：原 NICE 题复测与扩大 token 预算

当前正式协议更新为：75 轮、每次输出 32768、整题累计输出 262144、上下文 131072 token，交互墙钟 3600 秒、单工具 300 秒。NICE、SFT 验证、benchmark 与原题复测使用同一预算；SFT 教师轨迹上限仍为 32768。真实 smoke 强制核验完整上下文及 32768 输出位置的 NICE 反向容量，短输出测试不能代替。

- 冻结 4 份 checkpoint 选择后，initial+4 份权重对原 NICE 20 题各 greedy 运行一次，新增 100 episode，总计 2060。独立 nice_retest 结果空间，原随机 160 次 rollout 不覆盖，复测不参与权重选择，也不计入独立 benchmark。
- 新预算全尺寸 mock、独立串行评分、恢复及 zip 校验均通过：2000 候选、256 更新、2048 条次曝光，160+1300+500+100=2060 episode；2000 条分数和 20 份方向与串行参考一致，4040 个缓存文件恢复后 checksum 不变，audit 无 FAIL。
- 核验全部 5 个模型各 20 题、原题 ID 完全一致、repeat=0、greedy 标志、固定分母与 selection.json 不变；4 类新增 CSV 行数为 5/100/20/2，自动进入最终 zip。合成包 1,442,919 字节，CRC 与逐文件 checksum 通过；合成结果不构成研究证据。
- 本机全量回归曾在绘图时因 Anaconda NumPy/MKL 与 PyTorch 重复加载 libiomp5md.dll 异常退出。独立最小复现确认 OpenMP Error #15；设置当前测试进程 `MKL_THREADING_LAYER=SEQUENTIAL` 后，矩阵运算、PyTorch 反向与绘图联合复现通过。不使用忽略重复运行库错误的开关；不将本机设置强加到 Linux GPU 实验。
- 最终测试共 **328 项通过、0 失败**：基础 Anaconda 环境全量 pytest 为 **296 passed、3 个模块级 skip，210.57 秒**；随后在既有 `runs/real-interface-audit-20261028/venv` 中实际补跑这 3 个模块（官方模型接口、模型运行优化、Parquet），**32 passed、0 skip，22.37 秒**。该环境 Transformers=5.21.0、PEFT=0.21.2、datasets=5.1.0；真实微型随机模型不下载目标权重。新增覆盖大输出跨度容量门槛、复测必须晚于权重冻结、禁止未选中节点、原题 ID 与分母、恢复不重复、4 类复测表入包。
- 三张彩色图人工复核与程序布局审计通过；git diff --check 通过。日志为 `runs/a-nice-retest-pytest-final-20261101.log` 与 `runs/a-nice-retest-real-interfaces-20261101.log`。测试分两个环境执行，不宣称基础环境已安装全部模型后端依赖。

证据在被忽略的 `runs/a-nice-retest-full-20261101-v2/validation_checks.json`、`runs/a-nice-retest-serial-20261101-v2/` 与 `runs/verify_nice_retest_20261101.py`。GPU 容量与目标模型仍未在本机验收，服务器按 HANDOFF 的新预算真实 smoke 执行。

## 2026-11-01 师兄 A 新定义初次验收（扩大预算及追加复测之前）

当前正式配置为 configs/main.yaml，交接见 HANDOFF.md。下方 2026-10-28 等记录属于旧定义，不是本轮实验参数或目标模型成绩。

- 最终源码全量 CPU 回归：**324 passed，0 failed，0 skipped**，耗时 251.31 秒。包含新流程及历史基础回归。
- 新流程分块验证：Verified NICE/benchmark 拆分、未入候选池轨迹 ID 连接 SWE-smith 验证任务、两侧显式来源家族闭包、冻结候选被改动时阻止训练、同组跨学习率成员/顺序一致、12 节点精确曝光、每 (组, lr) 独立选择和平分规则、禁止提前/额外 checkpoint 接触 benchmark。
- 实际 Transformers/PEFT 微型随机 Qwen2 与 Qwen3_5：LoRA 梯度/有限差分/更新/重载，真实 padding 与 batch/累积，分块 CE 与有效位置输出层的损失/梯度对照，零 dropout checkpointed NICE 重放，常驻底座切换 LoRA 与新加载数值一致，greedy 不重复 teacher-forcing 前向，工具历史保留与生成额度边界。
- 进程/生命周期单元验证：8 卡评分分片、引用/顺序/缓存校验与失败回收；4 训练分支＋并发验证角色；训练卡释放后加入推理池；worker 租约；控制器服务退出和交付；Windows 瞬时文件锁下保持节点原子发布。
- 最终全尺寸显式 mock：**2000 候选、20 NICE 题 ×8、SWE-smith 100 验证、Verified 100 benchmark、高低各 512、2 LR/4 分支、12 节点、4 份选中权重**；256 更新、2048 条次曝光；160+1300+500=**1960 episode**；审计无 FAIL。
- 2000 条候选分数和 20 份验证向量与独立串行参考完全一致。恢复运行通过，训练/episode 计数不增加，4040 个投影缓存文件 SHA256 保持。
- CSV 行数、模型 ID、节点曝光、同组两学习率的数据顺序、选权重规则和任务集合核对通过。mock 的正式 success_rate/resolved_count 保持 null，合成值单列标记。
- 三类彩色/灰度 PNG 布局检查、人工图像复核及严格文件检查通过；PDF/SVG 同步导出。训练进度图连接有序曝光节点，benchmark 点图为 95% Wilson 区间，不伪装跨 seed 方差。
- 恢复后结果包 **1,448,175 字节（合成验证包）**；zip CRC、内嵌 checksum 与当前产物一致。包含 selection、各 split ID、报告/表格/图，排除原始任务记录、模型/LoRA/optimizer 权重、trace 和投影缓存。
- git diff --check 通过；默认服务器入口的只读计划输出通过。未执行 commit/push。

本机证据在被 Git 忽略的 `runs/a-verified-full-20261101-v6/`（含 validation_checks.json）、`runs/a-full-tests-final-v6-20261101.log`；串行参考在 `runs/a-verified-serial-20261101-v6/`。详细本地报告为 `runs/A_REDESIGN_VALIDATION_20261101.md`。这些不是服务器输入，不应上传 Git。

**边界：没有在本机下载或运行目标 27B 权重，没有启动真实 H200、Docker 任务或官方完整实验。**服务器必须使用新 clone、新 venv、新 storage，通过旧输入 checksum 核验、221 道任务环境预检、32k SFT 容量、131072/32768 输出位置的 NICE 反向容量与真实开发题 smoke。可选 CUDA 内核的实际启用和吞吐也要在 H200 上确认；CPU 通过不代表真实容量或网络已经验收。

## 2026-10-28 最后五块严格复核

按用户最后指定的五块重新展开：数据获取/目录、数据处理、训练/梯度、NICE、SFT/benchmark。发现并修复主任务与Pro按动态HEAD取差会漏掉Agent已commit修复，以及文本换行转换破坏CRLF补丁的真实判分问题。现固定任务起点SHA，主任务按bytes读取/保存补丁，Pro由Git写容器文件并通过Harbor download_file下载；Pro独立重判核对相同干净root/commit。Git仓库被模型破坏记任务失败，Docker故障仍记基础设施错误；pristine_regrade仅在实际执行重判时为true。另修复pad_token_id=0被or替换及非法下载limit/mode未提前拒绝。

最终同一生产源码三轮完整回归均为229passed、0失败、0跳过；每轮包括4项实际模型接口与1项实际Parquet接口。真实样本66项复验通过；新mock/resume/bundle通过，3110合成episode、360合成更新，audit为111PASS、1预期mockWARN、0FAIL；9类彩色/灰度图布局和目视检查通过。black检查59文件、flake8 F/E9、compileall与git diff --check通过。真实原生Task/JobConfig/AgentFactory及Harbor上传/下载接口在已安装库中另行确认，未启动容器。

新增真实Git回归覆盖已commit、暂存、新文件、二进制、含空格目录及LF/CRLF逐字节重放。实际Arrow/datasets/HF RepoFile测试通过项目hf_split导出与断点恢复；真实Verified原始Parquet也经过当前项目函数和实际datasets加载器导出前三条，与原始记录逐字段相等，外部传输在该项测试中替换，未伪称HF直连成功。本机Windows出现过datasets缓存锁路径过长，改用独立短缓存目录后通过；服务器handoff同步短存储根说明。

实际官方模型接口测试新增generate→NICE backward，候选与主验证反传后检查基座梯度为None、评分参数不变；SFT仅更新LoRA，后续完整模型加载不等于全量训练。NICE公式、fresh Adam顺序、零奖励/固定分母、源码对照和数据隔离继续核对，研究设置未改。完整结果及五块分项说明位于Git忽略的 `runs/strict-final-review-20261028/五块最严格复核报告.md`、`final-verification.json`；旧证据保持不变。27B/H200/官方容器仍须服务器实测。

## 2026-10-28 真实样本与官方模型接口补验

以下为新验证，历史章节中的“未下载真实数据”只描述对应旧轮次。原始样本、第三方源码、临时环境与完整本地报告均在Git忽略的 `runs/real-interface-audit-20261028/`。

取得训练轨迹、SWE-smith任务、Verified、Live各3条原始记录，以及Pro Public V2和Terminal-Bench2.0各1个原生任务；Terminal严格使用官方注册的89题与commit `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`。注册清单经Git blob SHA与GitHub元数据核对；任务文件逐个校验blob SHA。Verified完整Parquet通过源LFS SHA256检查；大分片只读取所需字节范围，Live结果又用真实HTTP范围读取复核，未用补造记录补齐实验数量。

真实记录专项 **66 passed**：字段逐项缺失/置空、真实Qwen fast tokenizer、完整多轮轨迹、工具观察不监督、超长拒绝、原生文件删除/置空及固定样本不足拒绝。3条训练轨迹中2条完整通过，输入为13397/11228 token，监督3748/2587 token；另1条超过32768而拒绝。3条主任务中1条题面为空，已修复为抽样前排除并留报告。Verified和Live分别通过官方TestSpec转换；Pro与Terminal通过实际Harbor Task/JobConfig加载，另测试实际Job.create但不启动容器。

独立环境安装真实Transformers/PEFT后，**3项官方模型CPU接口测试通过**。下载实际配置与tokenizer；实际官方类为Qwen3_5ForConditionalGeneration，使用39580参数的同架构随机小模型验证加载、4096个LoRA可训练参数、有限差分、NLL回放、一次更新、基座冻结、保存和精确重载。只替换GPU绑定，未下载27B目标权重。Linux x86_64/Python3.12的server+dev依赖完整解析通过；Windows缺Unix resource，完整SWE运行器未验收，纯转换/API函数测试明确绕过其平台初始化后执行，不能当作完整官方运行器已跑通。

实际目标权重索引也完成核对：18个分片、1199个tensor名称、64个文本LoRA目标模块；索引从Qwen官方ModelScope取得，Git blob与固定HF模型版本一致，不涉及权重下载。最终同一不变代码的三轮完整回归在三个独立进程/目录中执行，均为 **207 passed、0失败、0跳过**，包括实际官方模型接口测试；没有把重复次数当作新增用例数。另一次新mock及resume通过，主510+迁移2600=3110合成episode、360合成更新、52迁移汇总行；audit为111PASS、1预期mockWARN、0FAIL；9类彩色/灰度图布局检查和目视检查通过。black检查57文件、flake8 F/E9、compileall和git diff --check通过。报告和机器汇总为本地忽略目录中的 `真实数据与接口复核报告.md`、`final-verification.json`。

已找到NICE作者官方实现，固定commit `9516c140158299e5d5c3ba82e235ed2d69132ed0`，实际执行其中候选梯度、奖励梯度和影响分数三个核心函数，与项目CPU实现对照通过。原论文warmup/多checkpoint/Adam状态/TRAK与本项目变体差异已写入METHOD.md。服务器仍须实测目标权重、32768-token更新、8卡服务、镜像可达性和官方容器判分。

## 2026-10-28 连续三轮端到端复核

按用户要求，修复边界问题后冻结同一产品源码，连续三轮从入口到交付进行本地复核；每轮包含完整pytest、独立fresh mock、resume、全部阶段checksum、图像布局与读图、打包及跨轮不变量比较。三轮分别使用1/3/7个mock worker，均未发现新增问题。

| 轮次 | pytest | mock worker | 新增问题 |
|---|---|---:|---:|
| 1 | 166 passed，202.59秒，0失败/0跳过 | 1 | 0 |
| 2 | 166 passed，203.26秒，0失败/0跳过 | 3 | 0 |
| 3 | 166 passed，202.89秒，0失败/0跳过 | 7 | 0 |

累计498次测试执行（166个不同用例重复三轮）。每轮3110合成episode、360合成更新、52迁移汇总行/200配对，固定50分母和正式null；audit均111 PASS、1预期mock WARN、0 FAIL。恢复后分数字节不变，所有阶段checksum通过；不同worker数的分数、子集成员、预算、主任务结果和迁移抽样完全一致。各轮9类彩色/灰度图已复核，各结果zip57文件且无权重/检查点载荷。black检查54文件、flake8 F/E9、compileall及git diff --check三轮通过。

修复阶段新增29项回归：初始adapter完整staging后原子发布、复用时模型锁/目标模块检查、全部训练检查点链验证、完成episode仅汇总中断时不重跑、prepared服务身份/GPU和loopback代理检查、PID记录失败关闭进程、无效整数预算/零或非有限学习率拒绝、矛盾status/resolved不产生正式成功率。Handoff已同步恢复语义与新目录要求。

完整本地报告为Git忽略的 `runs/three-pass-review-20261028/连续三轮复核报告.md`；source.json、verification.json、每轮review-notes.md、pytest.xml、commands.json、mock和results.zip提供追溯证据。CPU线程mock不代表真实GPU负载，仍没有目标模型/实际GPU/官方容器/32768 token更新实测；不能据此承诺所有服务器条件下零故障。本轮没有commit/push，旧证据保留。

## 2026-10-28 最终全面复核

全套 `python -m pytest -q --junitxml=runs/final-audit-20261028/pytest-final.xml`：**137 passed，96.22秒，无失败、无跳过**。除原多卡、下载、梯度、训练、恢复、判分和完整CLI回归外，新增11项覆盖官方API常量/函数签名、固定兼容发布引用、Verified镜像namespace、单侧同源元数据隔离、主Agent三轮/多工具/失败观察回传，以及原生reward缺少Agent证据时拒绝接受结果。

本轮修复了真实上游接口漂移：SWE-bench固定官方v4.1.0引用解析出的完整commit，依赖约束swebench==4.1.0、ghapi<2；安装后主环境与Live独立环境执行pip check，并在模型下载前核对实际判分API。Git和Docker Compose纳入预检。Verified准备和判分统一swebench namespace；数据隔离对两侧instance_id/source_instance_id并集求交；原生Agent证据缺失记基础设施错误。

新完整mock在 `runs/final-audit-20261028/mock-final/`：300候选、30主题、12子集、360合成更新、主510+迁移2600=3110合成episode。52迁移行、固定50分母、正式成功率/delta/相关性为null。audit：111 PASS、1预期mock WARN、0 FAIL。9类PNG/PDF和灰度图布局及读图通过；真实打包脚本生成57文件结果包，包含四套各50个ID，无模型/检查点载荷。black检查53文件、flake8 F/E9、compileall及git diff --check通过。

最终本地报告与机器证据位于Git忽略的 `runs/final-audit-20261028/最终全面复核报告.md`、verification.json、blocks.csv、local-preflight.json。官方来源复核仅为只读代码/元数据核对，未下载目标权重或执行GPU/Docker。真实Python依赖安装、模型加载、32768-token容量、多推理副本、网络和官方容器仍须服务器验收；本地默认preflight按预期FAIL。HANDOFF已同步环境创建、兼容基线与新目录要求，历史证据保留。

## 2026-10-28 可配置GPU池与并行推理

新增 `--num-gpus N`（默认8、最低2）和可选授权UUID池；自动生成1个梯度/训练角色及N−1个独立推理服务，管理全部服务的端口、日志、身份和失败收尾。主任务及四套benchmark按固定任务索引取模并行，保留题目、种子和固定分母。评分与12分支训练仍单卡顺序执行，未实现八卡评分或并行分支训练。

全套 `python -m pytest -q --junitxml=runs/gpu-pool-20261028/pytest-final.xml`：**126 passed，180.07秒，无失败、无跳过**。新增13项测试覆盖2/4/8卡配置、默认8卡、可见设备顺序、非法/重复/不足设备、整个GPU池预检、7服务生命周期和部分启动失败清理；四套200题用实际Python线程验证每题恰好一次及正确服务路由，模型/容器接口仍为替身。

新完整mock在 `runs/gpu-pool-20261028/mock-final/`，3110合成episode、360合成更新、52迁移行和固定50分母；审计111 PASS、1预期mock WARN、0 FAIL。9类彩色/灰度图布局检查与读图通过。black检查51文件、flake8 F/E9和compileall通过。详细本地报告与证据在Git忽略的 `runs/gpu-pool-20261028/多卡修改验证报告.md`、verification.json。真实多GPU模型加载、容量和官方容器执行仍待服务器验收；使用新存储根与配置目录，旧证据保留。

## 2026-10-28 数据流与服务器托管复核

全套 `python -m pytest -q --junitxml=runs/flow-audit-20261028/pytest-final.xml`：**113 passed，177.98 秒，无失败、无跳过**。按配置/在线获取/托管、清洗与子集、梯度与协议、训练恢复、环境与benchmark、完整CLI六块归档，见本地 `runs/flow-audit-20261028/blocks.csv`。代码映射见 [DATAFLOW.md](DATAFLOW.md)。

本次修复首次原生 Git 获取、初始化中断恢复、Terminal 2.0 精确注册任务范围、已有模型的实际版本绑定与路径保护；新增服务器单入口及服务身份/GPU核对、代理绕过、失败清理和停止服务后的分析/audit/zip。新增13项CPU回归，其中Git使用真实合成仓库和bundle传输，不访问外网或提交项目仓库。HANDOFF已改为单入口和可复制的Claude执行指令。

新完整 mock 在 `runs/flow-audit-20261028/mock-final/`：300候选、12子集、360合成更新，主510+迁移2600=3110合成episode，52个迁移汇总行，各固定50分母；正式成功率/delta/相关性保持null。审计111 PASS、1预期mock WARN、0 FAIL。9类图的布局检查通过，已读取彩色和灰度总览复核。black检查49文件、flake8 F/E9单进程检查、compileall通过。

本地复核报告为Git忽略的 `runs/flow-audit-20261028/复核报告.md`，数值证据在同目录verification.json。没有下载目标权重、准备正式数据或执行真实GPU/Docker；实际架构加载、tokenizer/LoRA模块、32768-token容量、生成/梯度logprob对齐和官方容器链路仍须服务器验收。历史证据保留；源码变化后使用新存储根与配置目录。

## 2026-10-27 SWE-smith 主集与四组迁移评测

用户确认：主 NICE/同题评测改用 SWE-smith 30 题；Verified、Pro Public V2、Live Python full、Terminal-Bench 2.0 各随机固定 50 题。训练候选仍 300，12 个分支各 30 条一遍，训练后 LoRA 仍 12 份。配置与执行说明见 TRANSFER.md。

最终全套 `python -m pytest -q --junitxml=runs/transfer-validation-20261027/pytest-final.xml`：**100 passed，90.18 秒**，无失败、无跳过。测试覆盖任务隔离、迁移抽样/固定分母、各判分器分流、原生任务与镜像身份、Pro 全新环境补丁重判接线、增量分析、SWE-smith 冻结 SHA 与隐藏历史隔离，以及原有梯度/训练/恢复逻辑。隐藏历史检查在独立合成 Git 仓库执行，没有提交项目仓库。

新完整 mock 位于 `runs/transfer-validation-20261027/final-mock/`：510 次主集 + 2600 次迁移 = **3110 次合成 Agent episode**，360 次合成更新；52 组迁移汇总、200 行逐题配对、12 个合成 adapter 目录。审计 **111 PASS、1 WARN、0 FAIL**，WARN 仅为合成证据。正式指标仍为 null。九类 PNG/PDF/灰度图、300 DPI 和布局检查通过，已读取总览与新增迁移彩色/灰度图复核。black 检查 46 文件、flake8 F/E9、compileall 通过。

本地报告与数值核对保存在 Git 忽略的 `runs/transfer-validation-20261027/验证报告.md`、`verification.json`。未下载目标模型权重、未准备正式真实数据或执行 GPU/Docker；四套真实 benchmark 链路、实际 tokenizer、完整长度容量仍须服务器验收。此次改动要求新配置/输出目录，旧快照和报告保留。以下为历史验证记录。

## 2026-10-27 在线准备入口复验

新增配置生成、固定官方 harness 安装、`prepare-online` 与 `run-online`，操作顺序见 ONLINE.md。全套 `python -m pytest -q --junitxml=runs/online-validation-20261027/pytest.xml`：**74 passed，136.14 秒**，无失败、无跳过。

新增测试通过外部 API、模型和 Docker 的替身，检查四个来源的 commit 固定与重复配置、数值类型、split 范围、缓存路径、校验值、下载中断/发布恢复、完整模型快照恢复、下载后真实清洗分支、Run 冻结时机、独立开发题及主 pipeline 接入。未在本机下载真实数据或目标权重，未执行真实 GPU/Docker。服务预检失败阻止主实验，本地模式拒绝在线运行与 benchmark 安装。

新目录 `runs/online-validation-20261027/mock/` 完成 300 候选、12 子集、510 合成 episode 和 360 合成更新，审计 **49 PASS、1 WARN、0 FAIL**。七类 PNG/PDF/灰度文件及布局检查通过。black 检查 36 文件、flake8 F/E9 与 compileall 通过。详细本地报告位于被 Git 忽略的 `runs/online-validation-20261027/在线流程验证报告.md`，旧修复报告与运行快照保留。

## 2026-10-27 修复后复验

此前审查复现的七项问题已修复，新增协议与服务器保护回归测试。全套 `python -m pytest -q --junitxml=runs/fix-validation-20261027/pytest.xml`：**62 passed，221.82 秒**，无失败、无跳过。覆盖文本块、原生 Qwen 工具格式、多观察分组 mask、编辑器、deadline、依赖身份、adapter 切换恢复和定向容器清理；真实后端使用替身的测试不代表 GPU/Docker 已验收。

原定向数学、数据和模板检查修复后复跑 **17/17 PASS**，包含实际 Runtime 的 CPU LoRA 有限差分、奖励梯度分母、恢复逐值一致及官方 Jinja 模板往返。清洗样本仍为合成文件和 fixture tokenizer，未取得真实数据或目标 tokenizer。

新目录 `runs/fix-validation-20261027/mock-final/` 的完整 mock 完成 300 候选、12 个子集、510 次合成 episode 与 360 次合成更新。审计 **49 PASS、1 WARN、0 FAIL**；WARN 仅为合成结果不构成正式证据。七类 PNG/PDF/灰度图与布局检查通过，已读取成图检查裁切、标记、分组点、矩阵和标签。black 检查 32 文件、flake8 F/E9 与 compileall 通过。

新增 `preflight --prepared` 在本机按预期返回 FAIL，列出缺失的服务器依赖、GPU/Docker、冻结输入、实际文件和服务。真实模型加载、完整 32768 token 更新、官方判分和网络下载仍需服务器验证。主流程与 audit 已强制检查真实 smoke 的完整长度更新证据，不能用短样本或 mock 替代。修复后的协议/mask、源码及环境身份要求使用新运行目录。

本地详细修复报告与复查证据位于被 Git 忽略的 `runs/fix-validation-20261027/`；修复前审查证据原样保留。以下为上一轮历史记录。

日期：2026-10-26，用户时区 Asia/Shanghai。验证在本仓库工作区执行。没有创建 commit 或向 GitHub 推送。

## 已完成

`python -m pytest -q --junitxml=runs/verification-junit.xml`：**36 passed，74.76 秒**。包括一个完整 CLI mock 流程及 resume 检查。CPU 小模型直接随机初始化，不下载权重。

测试覆盖：

- fresh Adam 方向与真实 PyTorch AdamW 首步更新一致。
- assistant 监督掩码排除用户、工具观察、header；处理已知空 non-thinking scaffold。
- 终止 submit、工具配对、旧导出的唯一调用 ID 补全、未知工具拒绝。
- 生成 token 的 NLL 求和、固定零向量分母、逐题/高效聚合一致。
- 流式投影的分块不变性、可复现性、精确误差和 Adam/投影次序。
- 确定性仓库比例抽样、预算条件子集、随机组分数、并列/不可达预算。
- 独立分支相同起点，完整步骤恢复与连续 CPU 训练逐值一致。
- 检查点损坏拒绝、未发布检查点保留、已发布步骤日志恢复。
- 基础设施重试上限跨 resume 保留、全新 attempt、模型失败不重跑。
- 并行题目分片、episode 证据 checksum、源码/配置/来源变化使缓存失效。
- 全零奖励阻止评分，正式分析拒绝 mock，缺题和空审计不伪造成功率。
- 导入模块不加载 torch；本地 fetch-model 入口拒绝下载。

`python -m black --check nice_validation tests scripts`：27 个文件无需改动。

`python -m flake8 nice_validation scripts tests --select F,E9 --jobs 1`：通过。

`python -m compileall -q nice_validation scripts tests`：通过。源码、配置和文档的尾部空白检查通过。

本地 wheel 构建也已验证。标准 build_meta 使用的临时目录在此受管 Windows 沙箱中遭遇 ACL 限制；改用同一 setuptools/wheel 后端的工作区直接构建成功。生成文件位于 `runs/wheels-direct/`。不代表已在目标服务器安装或验证 GPU 依赖。

## 完整 mock

```powershell
python -m nice_validation --config configs/development.yaml --output runs/local-validation run --profile mock
```

流程完成：300 条候选评分，30 道合成验证任务，120 次 NICE 合成 rollout，390 次合成正式评测，12 个子集和 360 次合成训练生命周期更新。这里没有真实 LoRA 优化，也没有真实模型成功率。

审计没有 FAIL。唯一 WARN 是 `formal_evidence`：运行属于 synthetic fixture，不能作为研究证据。正式 rate/delta/相关性为 null，演示值使用独立字段。各训练分支初始化、数量、预算、分母、随机组分数和全部产物 checksum 均通过。

生成七类图，每类含 PNG、PDF 和灰度版；七个布局检查均 PASS。已人工读图检查分布、散点标签、全部组点、重叠矩阵和仓库构成。未发现文字裁切、缺字、标签遮挡或丢点。没有用平滑密度或均值柱掩盖每组三个点。

本机证据入口：

- `runs/verification-junit.xml`
- `runs/local-validation/report.md`
- `runs/local-validation/audit.json`、`audit.md`
- `runs/local-validation/tables/`、`figures/`
- `runs/local-validation/config/environment.json`
- `runs/local-validation/doctor.json`

这些运行产物均被 Git 忽略。服务器接手后应生成自己的真实证据，不能把本地合成产物改标签当作正式运行。

## 环境限制与未执行项

本机为 Windows，Python 3.12.4，16 个逻辑 CPU，RAM 约 31.7 GiB。预检未找到 nvidia-smi 或 Docker。目标模型、transformers、peft、datasets、LangChain 和 SWE-bench 真实后端依赖未安装。网络预检明确为 not_probed；外部资料查阅不等于完成真实数据下载。

没有下载 Qwen 权重，没有启动目标模型或真实训练，没有运行 GPU、多卡、容器或官方 SWE-bench 实测。真实数据 ID/revision/checksum 尚待服务器冻结。模型架构与模板、LoRA 模块、tool split 实际记录、官方 harness 安装 commit、任务镜像、生成/回放 logprob、32768 长度显存、吞吐及并行稳定性，都需要按 HANDOFF.md 先做独立真实 smoke。

初次测试还暴露了本机 NumPy/PyTorch BLAS 运行库冲突和 pytest 临时目录 ACL 问题。代码改用显式向量归约计算 norm/余弦/秩相关；CPU 测试使用继承工作区权限的临时目录。没有关闭数值检查，也没有设置 KMP_DUPLICATE_LIB_OK 来掩盖冲突。
