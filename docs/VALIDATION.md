# 本地验证记录

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
