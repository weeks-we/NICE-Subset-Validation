# 原实验环境修复与断点恢复辅助说明

本文件与 [Dockerfile](Dockerfile)、[SERVER_AGENT_PROMPT.md](SERVER_AGENT_PROMPT.md) 可单独发送给服务器agent。目标是保留已完成成果，补齐环境，用原代码、原配置、原storage继续。**不要求更新原仓库、新clone或采用新增离线入口。** 三份文件放独立部署目录；研究定义以原冻结证据和服务器原交接为准。

## 1. 先盘点，不重复已完成步骤

2026-11-08 20:11的r4中间包记录：数据划分、真实smoke、Verified 20题×8次NICE rollout已完成，43/160成功；随后prepare-environments:selection报Stale file handle。包内没有正式2000候选梯度评分、SFT和后续评测完成证据。服务器可能已有新进展，以最新状态为准。

定位原代码、解释器/镜像、启动命令、生成配置、storage、snapshot和stage日志。确认控制器/worker是否存活，避免双重启动。Stale file handle先查CPFS/共享存储挂载、失败文件及读写能力，不直接归因于缺依赖或镜像。

保留原环境，优先修复后恢复。已有模型、tokenizer、原始数据、任务镜像和完整阶段只核验复用，不重复下载、清洗、抽题或运行NICE160。必要核验不是重做正式计算。损坏证据保留并报告，不覆盖。

## 2. 完整运行环境构建

[Dockerfile](Dockerfile) 现在明确安装系统工具、Python 3.12及venv、CUDA扩展编译工具、Docker CLI、CUDA版PyTorch、数据处理/绘图、Transformers/PEFT/Accelerate、数据加载库、测试工具及固定commit的SWE-bench/SWE-smith判分工具。它不再依赖基础镜像预装这些Python包，不安装或复制更新后的实验代码。

仍需服务器agent提供不可猜测的原环境信息：原CUDA工具链和Python版本、基础镜像digest、PyTorch精确版本及其CUDA wheel源、两个官方harness的40位commit，以及原环境完整依赖锁。基础镜像使用公司可获取的NVIDIA CUDA devel、Ubuntu24.04镜像并固定digest；若原系统不是该组合，应先按原证据调整系统安装层，不随意更换原数值环境。

构建目录仅需：

```text
recovery-build/
  Dockerfile
  runtime.lock.txt
```

runtime.lock.txt必须是原验证环境的**完整精确依赖清单**，不能留空或只列缺项。删除原项目editable安装条目及swebench/swesmith条目，后两者由Dockerfile按原commit安装；其他包保留原版本和必要VCS来源，包括模型原先使用的CUDA快速内核。不得包含凭据或失效的本机文件地址。agent应检查锁的完整性、普通版本约束与原VCS来源，不能只靠pip freeze输出判断恢复身份一致。

Dockerfile先安装构建工具与CUDA版PyTorch，再安装完整锁和项目已知依赖，最后安装官方harness并执行pip check及关键模块导入检查。普通精确版本还作为constraints，冲突时应停止排查，不能删锁放任升级。原安装若用特殊wheel/构建参数或公司证书，agent在独立构建环境补相应层和受控包源，保留来源证据。

```bash
docker build -f Dockerfile \
  --build-arg BASE_IMAGE=公司CUDA_devel_Ubuntu24.04镜像@sha256:实际digest \
  --build-arg TORCH_VERSION=原PyTorch精确版本 \
  --build-arg TORCH_INDEX_URL=匹配原CUDA的实际wheel源 \
  --build-arg SWEBENCH_COMMIT=原40位commit \
  --build-arg SWESMITH_COMMIT=原40位commit \
  --build-arg PYTHON_ENV=原venv绝对路径 \
  -t 内部仓库/恢复环境:版本 .
```

CUDA devel提供编译工具链，宿主机驱动由平台提供。TORCH_CUDA_ARCH_LIST默认9.0用于H200；MAX_JOBS默认4限制构建并发，不改变实验GPU分工。原快速内核包及版本必须放进锁，必要的专用编译参数由agent依据原环境补充；构建时不要求GPU，真实快速路径、显存及梯度一致性在服务器验收。

基础运行依赖尽量在构建阶段装齐，任务特定镜像仍按第3节独立准备。镜像不包含模型、数据或正式结果，不启动GPU实验。运行时挂载原实验代码及原目录，不安装新仓库。构建日志、environment-freeze.txt和system-packages.txt留存用于比较。公司网络、镜像源、registry权限由agent填写，凭据不写镜像层。

**完整重建仍不保证原断点身份一致。** 包来源、解释器、路径、Python补丁版本及底层库等都可能影响校验。原环境仍可恢复时可以继续使用，不要求为使用本文件而重建；确需重建时先验证并保留原环境，不覆盖，不修改snapshot绕过检查。

## 3. 剩余任务环境只补缺项

读取原run/manifests下的冻结清单，实际路径以原配置为准：

| 用途 | 清单 | 任务数 |
|---|---|---:|
| NICE及最后原题复测 | validation.json | 20 |
| checkpoint选择 | sft_validation.json | 100 |
| 最终Verified benchmark | benchmark.json | 100 |
| 开发验收 | development_task.json | 1 |

先检查原image map、镜像锁和Docker daemon，身份正确的镜像直接复用，仅补缺失任务环境。不重抽题、不重制全部221题。

任务镜像是独立容器，不塞进运行Dockerfile。外网拉取、SWE-smith task ref获取、公共工作区制备或官方配方构建，在公司允许联网的制备阶段完成。若经内部仓库转运，记录任务ID、Agent工作镜像、grader镜像和digest/内容ID，计算节点提前装载。

遵循**原程序**的映射格式、官方标签、锁文件及stage规则，不要求使用新仓库脚本/catalog。不改写已冻结映射/checksum、不手动把stage标为完成；原格式不支持所需变动时报告兼容问题。SWE-smith checkout和隐藏历史隔离沿用原逻辑，不把参考补丁或答案放入Agent镜像，grader与工作镜像不可混用。

daemon或节点改变可能使原内容ID引用不可见，需提前恢复相同镜像内容和官方标签。Docker SDK、CLI和grader连接同一个daemon。运行容器与daemon看到的工作区绝对路径一致，模型、数据和结果路径保持原配置。公司Docker连接、内网registry、证书、挂载及GPU UUID由agent确认，不自动启用privileged或改变权限。

## 4. 继续原实验

1. 保留原源码、冻结配置、初始adapter、模型和数据；新容器仍挂载原代码，不安装更新仓库。
2. 用原程序校验源码、环境、harness来源、模型/adapter、任务及完成阶段checksum，不改snapshot消除报错。
3. 复用证据完整的已有smoke。环境改变时按原规则补必要工程验收，不为验收重跑整个NICE160，也不把验收当正式结果。
4. 确认原控制器/worker退出后，使用**原启动命令、原storage、原生成配置和原恢复选项**继续；按原机制确认残锁。本说明不给新storage/reuse-from启动示例，避免误开新实验。
5. 检查程序确实跳过完整阶段，从失败/未完成阶段继续。按中间包状态，先完成selection环境准备和initial基线，再进入正式NICE验证/候选梯度评分；以服务器原源码顺序及最新stage为准。

公司镜像地址和连接方式可按平台配置，但不改任务、模型版本、学习率、batch、checkpoint、采样、评测预算或GPU分工。若新环境不能满足身份检查，**保留成果，报告具体差异及修复选项，不清空、不从头启动、不绕过校验**。本说明不提供跨身份迁移。

原入口可能仍有联网行为；检查实际调用及官方harness，在制备阶段补齐资源。无改码不能消除外网需求时，列出域名、资源、阶段及平台解决方式，不强行采用新入口。未经实际判分验收不能声称完全离线。

## 5. 汇报与交付

先简报最新进度、故障原因、复用/补齐资源、环境身份是否兼容及确切恢复阶段。之后持续推进并记录命令、镜像digest及恢复日志。完成后按原交接返回results zip和completion.json，不要求Git提交、推送或替换原实验目录。
