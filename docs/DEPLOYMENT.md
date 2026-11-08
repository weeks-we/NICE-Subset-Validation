# 公司平台：构建与正式运行分离

本说明补充 [HANDOFF](HANDOFF.md)，不改变 configs/main.yaml 的实验定义。服务器 agent 全程负责构建、资源核验、安装、任务镜像制备、计算、恢复与汇报；公司地址、权限、挂载和调度参数由该 agent 根据平台实际情况填写。不要把已验证资源全部重新下载。

## 先检查现有成果，不盲目重做

已有模型、数据、任务镜像、已发布制备记录应核验复用。当前返回的 r4 证据证明 NICE160 完成；卡在 SWE-smith selection 环境准备，报 Stale file handle。先定位并修复存储故障，不能把它当成缺依赖。

**旧作业恢复与新镜像部署是两条路径。** 原代码、依赖、配置及挂载身份保持一致时，使用原恢复机制跳过已完成阶段。不要为了部署新镜像先破坏能恢复的旧环境。应用本次源码变化或改变数值依赖后，旧 snapshot 的身份校验可能拒绝恢复；不得改 snapshot/checksum 或复制旧正式结果伪装续跑。应保留旧证据，报告需要新运行的范围，再按 HANDOFF 创建新 storage。本功能不实现跨身份迁移已完成 rollout。

## 构建阶段

1. 从已通过真实 smoke 的服务器收集 Python/CUDA/PyTorch、数值库、扩展版本，以及 source_revisions 中 SWE-bench/SWE-smith 的固定 commit。导出 runtime.lock.txt，移除旧 editable 项目地址和两个 harness 条目；检查无凭据、无旧机器绝对路径，确认其余包使用确定版本。把它放在隔离构建上下文的 deployment/ 下，不提交 Git。若已有合格运行镜像，核验后直接复用。
2. 使用 [Dockerfile](Dockerfile) 模板。BASE_IMAGE 必须固定 digest，并已包含 Python3.12+、Git、Docker CLI、匹配的 CUDA PyTorch 和必要系统依赖/编译器。公司已有基础镜像若不满足条件，agent 在构建阶段补系统安装层；不要猜测私有镜像地址。CUDA 扩展按 H200 架构与已验证栈构建，不能只凭 import 成功认定快速路径可用。

```bash
docker build -f docs/Dockerfile \
  --build-arg BASE_IMAGE=内部基础镜像@sha256:实际digest \
  --build-arg SWEBENCH_COMMIT=实际40位commit \
  --build-arg SWESMITH_COMMIT=实际40位commit \
  -t 内部仓库/实验运行镜像:版本 .
```

只用干净构建上下文：排除 .git、runs、数据、模型、缓存、结果、凭据和本地报告；保留代码、配置、docs 及单独准备的 deployment/runtime.lock.txt。可将 [dockerignore.template](dockerignore.template) 复制为隔离构建目录的 .dockerignore，并按公司目录补充排除项，提交构建前人工检查上下文；模板不是凭据检测器。构建可联网，计算阶段不安装依赖。构建后保留 pip freeze、镜像 digest 和构建日志，不把基础镜像里有 CUDA 误认为宿主机驱动已就绪。

3. 使用原运行的四份冻结 manifest：validation.json（20）、sft_validation.json（100）、benchmark.json（100）、development_task.json（1）。它们位于原 run/manifests 下；不重新抽样，不按结果换题，不把包含参考补丁的完整 manifest 上传公共仓库。制备机器需读取同版本任务文件和官方 harness。
4. 在有 Docker 和外网的制备节点执行：

```bash
python scripts/prepare_task_images.py prepare \
  --config /绝对路径/已有server.local.yaml \
  --run /绝对路径/原storage/runs/main \
  --catalog /绝对路径/deployment/task-catalog.json \
  --registry-prefix 内部仓库/项目名 --execute-server
```

脚本只读取任务文件，不运行模型、不改原 run。优先使用本地基础镜像；缺失则拉取；SWE-smith 公共工作镜像沿用项目既有 checkout/隐藏历史隔离逻辑。Agent 镜像和 grader 基础镜像分别推送并固定 digest。每完成一道即写 catalog，重复执行核验并复用已发布项。若官方镜像拉取失败，agent 依据固定官方 harness 在制备阶段补构建后重试；脚本不会猜测官方构建配方或换题。已有旧 task_images.json 不会未经核验直接当制备成果；尤其 Smith 成品需证明 checkout 与隔离正确。

目录写入正常、Docker daemon 可用和 registry 已登录由平台准备；凭据放平台 secret，不写 Dockerfile/镜像层。脚本不自动配置公司权限或偷偷启用 privileged。平台若不能创建任务容器，先明确阻塞，不改成无容器模拟评测。

## 内网装载阶段：提交正式 GPU 作业前

计算节点/持久 Docker daemon 必须先从**内部**仓库装载所有 catalog 镜像。无需外网：

```bash
python scripts/prepare_task_images.py load --config /配置/server.local.yaml \
  --run /原storage/runs/main --catalog /deployment/task-catalog.json --execute-server
python scripts/prepare_task_images.py check --config /配置/server.local.yaml \
  --run /原storage/runs/main --catalog /deployment/task-catalog.json --execute-server
sha256sum /deployment/task-catalog.json
```

load 校验内容 ID，并恢复官方 grader 预期的本地镜像标签；check 只检查不下载。容器内 Docker SDK、Docker CLI 和官方 grader 必须连接**同一个 daemon**，镜像可见性一致。挂载宿主机 Docker socket 需要公司授权，且相当于高权限访问；不自动启用。平台若使用远程 Docker/TLS，agent 配置对应 DOCKER_HOST/证书。运行容器与 daemon 必须对任务工作目录具有一致的绝对路径；结果目录可写，模型/原始数据只读。镜像中不安装宿主机驱动。

## 正式运行阶段

使用基于 main.yaml 的新 overlay；公司 agent 填实际绝对路径及 SHA，不改研究参数：

```yaml
extends: /opt/nice/configs/main.yaml
deployment:
  preinstalled: true
  offline: true
  task_catalog: /deployment/task-catalog.json
  task_catalog_sha256: 填sha256sum输出的64位值
```

用原入口传 --base-config /部署配置/compute.yaml，其他参数依 HANDOFF。新部署使用新 storage/config-dir，并传 --reuse-from 原storage。此模式跳过 pip/harness 安装、启用 Hugging Face 离线模式，缺模型/数据立即停止；环境准备只接受已装载且 digest/任务/harness 匹配的镜像，不 git fetch、不外网 pull。catalog 文件校验值进入配置和外部身份，冻结后不能替换。

在平台用运行镜像启动的容器内执行（路径、内部镜像 digest、GPU UUID 和挂载由 agent 预先确认）：

```bash
python scripts/run_server.py \
  --base-config /部署配置/compute.yaml \
  --storage-root /data/nice-prebuilt-new \
  --config-dir /部署配置/generated-new \
  --reuse-from /data/原storage \
  --model-path /models/已核验基座 \
  --num-gpus 8 --execute-server
```

此命令是新部署入口，不是给已冻结 r4 强制切环境的命令。若原 r4 可按原环境恢复，继续使用原命令及其原配置，无需先迁移镜像。

新运行仍需必要的 GPU/任务环境验收；只有同身份运行内已完成且校验通过的阶段才自动跳过。启动时检查全部任务资源，然后使用官方判分；官方 harness 若仍需要额外环境或尝试联网，应在预检/smoke 阶段修复制备，不把缺环境记成模型失败。代码禁止了自身已知的下载入口，但不替代平台网络策略；若要严格禁止外网，平台应限制计算侧出口并允许内网服务，最终实测官方判分链路。Agent 任务容器原有 network=none 保持不变。

构建、加载、check、真实 smoke 的日志和镜像清单落盘；不要只回复“镜像已构建”。最终交付仍按 HANDOFF 的 audit、results zip 和 completion.json。
