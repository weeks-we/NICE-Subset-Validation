# 两次环境的覆盖核对

最新用户范围：本次全部221题工作镜像、官方grader及SDK派生层都提前制备，完整模型/数据准备齐全；只对固定小样本做训练和测评短测。下文早期“一题即可”的最低验收范围由HANDOFF.md顶部替代，不能作为少准备资源的理由。

结论：本目录主Dockerfile覆盖第一次的主运行环境类别；**它本身不包含第二次Claude SDK/vLLM环境**。完整环境是下列多镜像组合，不能只构建主镜像就声称两次环境全部完成。

| 镜像/资产 | 构建文件 | 包含内容 | 本次诊断要求 |
|---|---|---|---|
| 主运行环境 | 本目录Dockerfile + B新生成runtime.lock.txt | Python3.12、CUDA版Torch、Transformers/PEFT、数据处理、NICE/SFT依赖、绘图测试、Docker CLI/SDK、固定commit的SWE-bench/SWE-smith | 必做，真实前反向短测 |
| vLLM推理服务 | ../reevaluation/Dockerfile.vllm | 官方vLLM0.22.0依赖栈、固定采样转换策略、离线模型加载设置 | 要覆盖第二次环境必须构建并短测 |
| SDK工具箱 | ../reevaluation/Dockerfile.sdk | SDK0.2.87、固定Claude2.1.153及系统工具，独立venv | 要覆盖第二次环境必须构建并短测 |
| 任务SDK工作环境 | ../reevaluation/Dockerfile.task-sdk | 一道干净任务父镜像+独立SDK/Claude层，保留testbed | 单题SDK接入验收必备 |
| 官方任务grader | 官方固定harness准备的任务镜像 | 该题测试环境与官方判分 | 单题正/负例必备，不加SDK层 |
| 模型/tokenizer、数据 | 外部资产目录及来源锁 | 固定revision基座与少量诊断数据 | 不COPY进镜像，不上传Git |

主镜像与vLLM环境分离，因为CUDA/PyTorch/Transformers等依赖组合可能冲突；单一venv同时pip安装全部不是本项目支持的部署方式。SDK工具箱也不能代替真实任务工作区。Dockerfile只能装运行程序，不能把其他Docker镜像变成自己内部已加载的任务镜像。

服务器agent若本次要检查“之前两次环境”，应构建上述主环境、vLLM、SDK工具箱及一题SDK工作镜像，并准备该题原grader。**不需要准备全部221题或启动完整实验**。资源/权限不足而只完成主环境时，第二次环境必须标NOT_TESTED，不得报告完整通过。

构建文件引用仓库内辅助脚本和requirements，给B应提供同一commit的完整仓库；只发本目录Dockerfile不能构建SDK/vLLM镜像。主Dockerfile可单独配runtime.lock.txt构建；其余三个从仓库根作为build context，具体命令见../reevaluation/DEPLOYMENT_SDK.md。官方Claude checksum需联网制备时查固定版本官方manifest，不接受自行算哈希当可信来源。

主依赖锁由B首次构建解析，快速内核和系统库仍需真实GPU验收；基础镜像digest、torch/CUDA、harness commit及SDK任务所用独立Python路径未由本机实测确定。宿主驱动、Docker daemon、nvidia-container-toolkit、内网和挂载由平台提供。故本清单表示构建覆盖，不能代表B已构建成功。
