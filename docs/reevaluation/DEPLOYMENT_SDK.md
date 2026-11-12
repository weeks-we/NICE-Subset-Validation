# 本次 SDK 重测：先复用，再补运行层

本文件是 HANDOFF.md 顶部重测补充的最新 SDK 部署入口，取代此前“Python直接拼claude命令”的调用方式。实验任务、已选权重、250 turn和其他预算不变。SDK底层仍启动Claude Code程序，不调用Claude云端模型。

## 必须复用的旧资源

先读取原 storage 的 snapshot、selection.json、模型锁及任务镜像清单。原固定revision基座/tokenizer、initial adapter、四份已选LoRA、100+20题及已准备的任务/grader镜像全部核验后复用。不得重新下载最新模型/数据替换；不得重训、重算NICE、重选checkpoint。原Python/conda测试环境、仓库base_commit与grader不变。旧目录只读，新环境与结果另存。

原任务镜像可用不等于已包含SDK。仅为agent侧补独立 `/opt/nice-sdk`；不向testbed环境pip install，不替换其Python，不把旧训练依赖锁灌进vLLM环境。grader继续使用原镜像，不装SDK。

## 构建阶段需要下载/制备什么

| 资源 | 来源/文件 | 复用规则 |
|---|---|---|
| vLLM GPU镜像 | 官方 `vllm/vllm-openai:v0.22.0`，镜像到内部仓库并固定digest；`Dockerfile.vllm`核验版本 | 已有同版本同内容可直接复用，不更新旧训练环境 |
| 独立SDK工具箱镜像 | 公司Python3.12 Debian镜像固定digest；`Dockerfile.sdk` | 用于SDK smoke，不替代SWE任务镜像 |
| SDK Python包 | `claude-agent-sdk==0.2.87`及依赖，构建阶段下载；保存pip freeze及构建日志 | 不把SDK依赖安装进旧testbed |
| Claude原生运行程序 | 官方2.1.153 linux-x64，要求官方SHA256；安装脚本在本目录 | SDK0.2.87自带CLI2.1.150，本项目显式cli_path选择2.1.153，禁止误用自带版本 |
| 实际任务SDK派生层 | `Dockerfile.task-sdk`，FROM原任务digest，只补 `/opt/nice-sdk` | 已有同一层则复用；保留父镜像及派生digest对应关系，不重建任务原始依赖 |
| 原模型/LoRA/数据/任务镜像 | 从旧锁定位并校验 | 不重复下载，不COPY进新镜像 |

SDK0.2.87及其自带CLI版本依据官方tag源码：[pyproject](https://github.com/anthropics/claude-agent-sdk-python/blob/v0.2.87/pyproject.toml)、[自带版本](https://github.com/anthropics/claude-agent-sdk-python/blob/v0.2.87/src/claude_agent_sdk/_cli_version.py)。自定义cli_path、tools、system_prompt等按[固定版本类型定义](https://github.com/anthropics/claude-agent-sdk-python/blob/v0.2.87/src/claude_agent_sdk/types.py)使用。该组合仍须服务器真实验收；不声称发布者验证了本项目的LoRA/vLLM组合。

主要运行依赖集中在 requirements-sdk.txt：SDK0.2.87、MCP1.23.0、AnyIO4.14.2、Sniffio1.3.1；两份SDK Dockerfile使用同一文件。完整传递依赖仍以构建产物的pip freeze和内部镜像digest冻结，不把这四项误称完整哈希锁。不得向旧训练环境安装此依赖组。

在仓库根目录作为Docker build context执行，例如：

```bash
docker build -f docs/reevaluation/Dockerfile.vllm \
  --build-arg VLLM_BASE_IMAGE="$INTERNAL_VLLM_IMAGE_DIGEST" \
  -t "$NEW_INTERNAL_VLLM_IMAGE" .

docker build -f docs/reevaluation/Dockerfile.sdk \
  --build-arg SDK_BASE_IMAGE="$INTERNAL_PYTHON_IMAGE_DIGEST" \
  --build-arg CLAUDE_SHA256="$OFFICIAL_CLAUDE_LINUX_X64_SHA256" \
  -t "$NEW_INTERNAL_SDK_IMAGE" .

docker build -f docs/reevaluation/Dockerfile.task-sdk \
  --build-arg TASK_IMAGE="$ORIGINAL_TASK_IMAGE_DIGEST" \
  --build-arg SDK_PYTHON="$EXISTING_NON_TESTBED_PYTHON" \
  --build-arg CLAUDE_SHA256="$OFFICIAL_CLAUDE_LINUX_X64_SHA256" \
  -t "$NEW_INTERNAL_TASK_SDK_IMAGE" .
```

变量由服务器agent按原证据及公司平台填写，不能随便改实验参数。CLAUDE_SHA256必须来自官方对应平台发布清单；不能仅下载后二次计算同一文件充当可信校验。需要公司CA/代理时在构建平台配置，不能关闭TLS验证。构建产物同步内部仓库，记录digest、父镜像、pip freeze、Claude checksum与来源。已有镜像先inspect再决定是否构建。

`Dockerfile.task-sdk`要求存在独立于testbed的Python>=3.10及venv支持，默认检查/opt/conda/bin/python。若原镜像没有，停止并在构建阶段制备公司批准的独立Python运行层，保留原testbed解释器；不要升级testbed或在正式任务里临时安装。还须测试运行库/glibc、Git/ripgrep、非root UID及工作区权限。不保证任意旧镜像可直接复制工具箱venv使用。

## 运行阶段

1. `scripts/prepare_reevaluation.py`只读核验旧120题、模型、adapter和原任务镜像；建立新agent派生镜像映射，原grader映射保持不变。这个脚本不会自动构建/下载或启动vLLM。
2. 用独立vLLM镜像部署原基座+指定adapter，只读挂载资源；使用授权GPU UUID，按既定八卡TP1/单槽验收容量。任务容器只访问内部模型端点；禁公网答案检索。正式运行只用已制备镜像，pull=never。
3. `scripts/smoke_reevaluation.py`现在通过Python Agent SDK调用query，显式指定 `/opt/nice-sdk/bin/claude-pinned`，不再由项目手工拼CLI命令。先用工具箱测试，再在实际SDK任务派生镜像/所有服务与adapter路由上测试。

```bash
python scripts/smoke_reevaluation.py \
  --image "$PREPARED_SDK_IMAGE_DIGEST" --network "$RESTRICTED_NETWORK" \
  --endpoint "$PRIVATE_VLLM_ROOT_URL" --model "$SERVED_ADAPTER_ALIAS" \
  --claude-version 2.1.153 --output "$NEW_STORAGE/deployment/sdk-smoke-unique" \
  --execute-server
```

由平台secret注入ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN；不得写入Git。SDK使用原生claude_code系统提示preset、明确工具集合、空外部settings来源、禁止额外MCP和子agent，所有模型别名同adapter。SDK不是采样转换器；Qwen thinking/xhigh、历史、采样及输出预算仍须后端映射核验。

smoke使用短预算（12 turn、8192单次输出、600秒），要求Read/Write/Bash及新的随机工具反馈；逐条保存SDK事件，失败留日志并退出。它不代表官方SWE判分、LoRA数值或262K容量已经通过。必须再做独立开发题的SDK→实际仓库→补丁→原官方grader测试，不能要求模型一定解对，但必须区分模型失败与基础设施错误。

## 独立重测执行入口

新增完整入口 `scripts/run_reevaluation.py`。服务器 agent 按 `deployment.example.json` 在 Git 外填写机器配置：授权 GPU UUID、现有 reference/grading Python、内部镜像 digest、Docker internal 网络及网桥地址。`gateway_url` 不带端口，控制器按服务编号分配端口；上游 vLLM 只绑定宿主机127.0.0.1。任务网络必须 `Internal=true`，保存平台网络策略证据，不依赖公司 HTTP 代理。

任务SDK派生镜像构建另传 `--build-arg TASK_IMAGE_ID=sha256:原agent镜像内容ID`，写入 `nice.parent_image_id` 标签。`task_images` 覆盖原120题和原开发验收题，父镜像须与原映射一致。开发题通常是SWE-smith，继续原SWE-smith判分；Verified继续原SWE-bench判分。原始grader不加SDK层，不重建依赖；`grading_python`须已有原source-locked swebench/swesmith。

```bash
python scripts/run_reevaluation.py preflight \
  --old-run /existing/completed/run --deployment /private/reevaluation-deployment.json \
  --output /new/sdk-reevaluation --execute-server

# 相同参数/目录继续执行，也用于中断恢复。
python scripts/run_reevaluation.py run \
  --old-run /existing/completed/run --deployment /private/reevaluation-deployment.json \
  --output /new/sdk-reevaluation --execute-server
```

验收顺序：核验旧资源 → 在授权GPU生成短输入HF/PEFT参考 → 每卡一个vLLM服务 → 每服务×五adapter比较top-20 logprob（15条参考，绝对误差≤0.15）→ 每服务真实完成229376输入+32768输出容量测试 → 全服务×全alias SDK文件工具测试 → 开发题参考补丁正例、空补丁负例、真实SDK补丁官方判分 → 实际观测原生compact_boundary。任一失败不进入正式任务，保留日志；不要求模型一定解对开发题。TP2只能在正式执行前另行冻结，不能静默缩窗口或量化。

`vllm_policy.py`补齐原生Anthropic转换遗漏的采样字段并强制thinking/xhigh；启动时校验镜像内策略源码。网关记录实际input/output usage，累计输出包含thinking及摘要请求。冻结250轮、单次32768、累计262144、上下文262144、墙钟7200秒、单工具1200秒。原生压缩显式传262144窗口和80%提前阈值；固定CLI是否支持以真实compact_boundary验收为准，失败禁止正式运行。网关同时根据实际剩余上下文保留输出空间。

八服务各一个推理任务槽，静态注册五份LoRA，只读挂载；生成补丁后释放推理槽，最多八个官方判分并行，与后续推理重叠。600题分benchmark500和NICE诊断100独立汇报，不重新选权重。工具hook限制超时和后台模式；外层监督工具时间/墙钟，取补丁前终止容器UID1000全部进程。预算停止有非空补丁仍官方判分，空补丁按官方失败约定记录。强制中止的流若未收到最终usage会标记incomplete_usage，不能把部分token计数当成完整开销。正式阶段不pull/build/pip install或下载数据模型。

恢复按题：完成结果核验checksum后跳过；已发布补丁但判分中断只继续判分。未发布结果的中断尝试按基础设施错误处理，最多两次重试，保存全部尝试，不选最好结果；不声称恢复到中断前精确token。正常模型失败不重试。源码、资源、配置、GPU/驱动身份变化须另建输出目录，禁止改frozen.json绕过。

启动恢复先清理带本次绝对存储根标签 `nice.run_root` 的遗留容器，避免硬中断后旧vLLM继续占显存；不执行全局prune，不删除共享资源。新任务容器只给root监督进程工作区权限调整及终止子进程所需的CHOWN/FOWNER/DAC_OVERRIDE/KILL能力，SDK始终以UID1000运行且禁止提权。压缩/thinking验收不只检查开关：须观测实际历史thinking块和压缩后输入token减少。

完成返回 `controller/results-*.zip` 和 `controller/completion.json`；摘要/逐题CSV/PNG/audit在 `reports/`，完整事件、预算、补丁、官方判分证据在 `episodes/`，不打包巨大trace或权重。未完成禁止生成完成包。本机只运行显式synthetic的600题mock：

```bash
python scripts/run_reevaluation.py run --mock --output runs/sdk-600-mock
```

SDK会话统一所有辅助模型别名，并移除嵌套CLAUDECODE标记。query()一次调用包含多轮原生工具交互；本项目任务恢复由独立journal负责，不把SDK已有resume选项冒充精确进程恢复。旧run_server.py仍是原实验入口，不用于新重测。

旧 docs/Dockerfile / DEPLOYMENT.md / SERVER_AGENT_PROMPT.md 是旧实验恢复资料。本次构建与启动优先使用本目录，不照旧指令重跑已完成计算。
