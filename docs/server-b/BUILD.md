# B联网构建与离线运行分离

B已确认使用8张H20。构建前核验8张授权GPU及单卡显存/driver/device capability，并显式传TORCH_CUDA_ARCH_LIST；Dockerfile不再默认H200架构。不要继承A的单卡144GB或每卡4并发配置。B短测从单槽开始，模型/真实长度容量不足时如实报告，必要的TP2只适用于推理且须两张实际可用卡；训练并行不能据此推定可用。

**最新范围以HANDOFF.md顶部为准：本次制备全部221题环境、完整数据/模型及SDK/vLLM层，再做有界短测，不再仅制备一题。以下早期“单题/可选SDK”表述不适用。** 资产与任务镜像可全部准备，但正式NICE/训练/600次评测不自动执行。

两次环境的明确覆盖关系见 [IMAGE_COVERAGE.md](IMAGE_COVERAGE.md)。本目录主Dockerfile不含SDK/vLLM；若要完整检查之前两次环境，需另构建该清单中的SDK工具箱、vLLM服务及一题任务SDK派生层，连同原grader做短测，不运行完整实验。

当前只做主镜像构建与短程环境诊断，以HANDOFF.md顶部覆盖条款为准。下面的数据/任务catalog及SDK镜像是完整部署背景，不要求本次全部制备；主镜像、少量样本、一次短GPU测试和单任务判分完成后报告并停止，不启动完整实验。SDK/vLLM是独立镜像层，未构建/未测试就记NOT_TESTED。

主Dockerfile是完整训练/数据/判分运行环境模板，不含数据权重或具体任务镜像。构建前先新解析并冻结B自己的完整runtime.lock.txt，不能使用要求A旧环境存在的恢复锁。

在单独构建目录放本目录Dockerfile及runtime.lock.txt，执行：

```bash
docker build -f Dockerfile \
 --build-arg BASE_IMAGE="$INTERNAL_CUDA_DEVEL_UBUNTU2404_DIGEST" \
 --build-arg TORCH_VERSION="$PINNED_TORCH_VERSION" \
 --build-arg TORCH_INDEX_URL="$APPROVED_CUDA_WHEEL_INDEX" \
 --build-arg SWEBENCH_COMMIT="$PINNED_SWEBENCH_COMMIT" \
 --build-arg SWESMITH_COMMIT="$PINNED_SWESMITH_COMMIT" \
 --build-arg TORCH_CUDA_ARCH_LIST="$ACTUAL_GPU_ARCH" \
 -t "$INTERNAL_B_RUNTIME_IMAGE" .
```

以上变量全部由B服务器agent从实际平台与来源核验填写，不包含默认假定的CUDA/torch组合。H200架构9.0；非H200不能沿用。构建后pip check、核心import、pip freeze、系统包列表、镜像digest与来源日志必须留存；CUDA快速内核要在锁中有固定版本/commit，编译成功不等于实际前反向可用。

GPU驱动、Docker daemon、nvidia-container-toolkit、公司CA/registry凭据及可写挂载属于平台，不在Dockerfile里伪安装。控制器容器访问Docker socket意味着宿主级控制权限，须用平台批准方式；不自动privileged或全局prune。内层任务路径必须在daemon宿主上同绝对路径可见。

后续SDK/vLLM三种构建使用 ../reevaluation/ 的独立Dockerfile，按其说明从仓库根作为context，不能把所有Python包装进同venv。任务SDK层来自已核验的每个工作镜像，不用于grader；镜像上传内部仓库后固定digest，计算阶段pull=never。

数据/模型放外部资产卷，不COPY进镜像或Git。为B独立实验生成完整revision/来源锁、清洗划分证据及221题镜像catalog；资产/任务清单与运行配置在正式计算前冻结。制备步骤与GPU正式运行不是同一条无条件命令，详见HANDOFF。
