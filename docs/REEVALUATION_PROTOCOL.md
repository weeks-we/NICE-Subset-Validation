# 固定权重重测协议（2026-11-12）

最新调用方式：Python Claude Agent SDK0.2.87，显式指定Claude程序2.1.153；SDK底层仍调用该程序。以 [SDK部署说明](reevaluation/DEPLOYMENT_SDK.md) 为当前实现和环境准备依据；下文直接CLI命令是历史方式，250 turn及其余研究预算不变。

本文件确定重测研究参数，不修改 main.yaml、旧 snapshot 或旧结果。configs/reevaluation.yaml 为参数清单，scripts/run_reevaluation.py 为独立执行器；须按SDK部署说明通过服务器真实验收后运行。

## 已确定的参数

| 项目 | 固定值与依据 |
|---|---|
| 软件 | vLLM 0.22.0、Claude Code 2.1.153；禁自动更新。后者依据公开降级恢复记录选择，不宣称完整组合已实测 |
| 基座 | 复用旧锁定 revision 的 Qwen/Qwen3.8-27B、tokenizer/chat template，BF16；不取当前 main 替换旧版本 |
| 权重 | initial 使用原初始 adapter；其余为旧 selection.json 中 low_lr1_p100、low_lr2_p025、high_lr1_p050、high_lr2_p100；不重选 |
| 任务 | 原 Verified benchmark 100 题与原 NICE 20 题；五份权重各运行一次，共 600 episode；两类分开报告 |
| Thinking | enable_thinking=true、preserve_thinking=true、Qwen reasoning_effort=xhigh；对应 Qwen 官方默认推荐 |
| 采样 | temperature=1、top_p=.95、top_k=20、min_p=0、presence_penalty=0、repetition_penalty=1，采用 Qwen thinking 推荐；不是 greedy |
| 长度 | 输入加本次输出共不超过 262144；单次输出至多 32768，含 thinking；整题所有推理请求累计输出至多 262144，含辅助请求与 thinking |
| 交互预算 | 250 个主 agent turn、墙钟 7200 秒、单工具 1200 秒；官方 grader 另计 3600 秒。均为本项目工程预算，非 Verified 统一标准 |
| 工具 | Bash、Read、Edit、Write、Glob、Grep；禁子 agent、WebSearch/WebFetch、交互提问及额外 MCP。禁止通过 shell 联网下载答案；靠容器出口策略执行隔离 |
| 上下文 | 使用 Claude Code 原生上下文管理/压缩，不叠加原最近五观察裁剪。后端硬上限仍为 262144；必须记录实际压缩触发位置，不能默认 CLI 已识别自定义模型窗口 |
| GPU | 8×H200-144GB，默认每卡一个 TP1 服务、每服务一个任务；容量不足在正式运行前统一冻结 TP2/四服务，仍不足则停止反馈。BF16、不量化、不缩窗口 |
| 判分 | 保留原官方 harness revision、任务及 grader 镜像。预算结束提取非空补丁送官方判分，同时保留 stop_reason；不把预算结束直接等同于判分失败 |
| 重试 | 同类基础设施错误最多两次重试并保留证据；模型失败不重试。不得按 benchmark 成绩调参、换题或换权重 |

32768 单次输出参考 Qwen 模型卡的 QwenSWEBench 记录；其 8 小时及 avg@3 不适用于本项目，不一并复制。模型卡另有面向 1M 上下文的更大输出建议，不适用于此处固定的原生 256K 窗口。250 turn 与两小时墙钟结合旧实验大量 ACTION_LIMIT 的经验给出上限，并不要求模型用满。

## 参数如何实际生效

Claude Code 在每题容器内运行，使用 `claude -p`、明确 `--model`、`--max-turns 250`、工具白名单、stream-json 日志及隔离的会话目录。仅在无敏感宿主挂载的任务容器中自动批准工具。每题全新工作区，不暴露测试答案、其他任务或模型目录。

vLLM 固定 qwen3_coder 工具解析、qwen3 reasoning 解析。主请求及 Opus/Sonnet/Haiku 别名全部指向同一当前 adapter；不使用外部 Claude 模型。初始模型也不得省略原初始 adapter。正式启动须核验实际服务版本、adapter 文件与加载日志。

采样参数、Qwen thinking/template 参数及输出上限必须在最终进入 vLLM 的请求或后端转换层强制一致，不能只依赖 CLI 环境变量或 vLLM 默认 generation config。特别是 Claude 的 effort/thinking budget 不等于 Qwen reasoning_effort；不能假定会自动正确映射。若需要转换层，只做字段映射/预算控制，保持 Messages 流式与工具语义，并记录实现与请求证据。不得静默忽略不支持的字段。

源码复核发现：vLLM 0.22.0 的 Anthropic `_build_base_request` 显式传递 temperature/top_p/top_k/chat_template_kwargs，但没有在该函数里显式传递 Qwen reasoning_effort 和三项 penalty/min_p。因此接通 HTTP 不能证明配置全生效；后续执行器必须补齐确定的后端参数映射或强制默认值并核验，不把缺省参数偶然相同当作实现。参考 [对应版本转换源码](https://github.com/vllm-project/vllm/blob/v0.22.0/vllm/entrypoints/anthropic/serving.py)。

累计输出、墙钟与单工具限制需要实际计数/监督；CLI max-turns 不是所有 HTTP 请求数。上下文压缩等辅助请求也计入 token/墙钟。终止后先停止推理和任务进程，再提取工作区补丁并送判，不能提前销毁容器。验证限额真实触发及中断恢复后，才可开始正式任务。

## 服务器 agent 只需决定部署信息

原 storage、五份 adapter、基座路径、内部镜像地址/digest、GPU UUID、私有端点、专用网络和企业 CA/secret 由服务器 agent 查找并核验，写入新输出目录的 deployment 配置，不写死到 Git。这些是机器事实，不是尚未选择的实验超参。找到多份资源时按原锁与 checksum 决定；缺失或不一致先报告，不能擅自换版本、重新下载最新任务或重新选择 checkpoint。

目前未完成的是完整 600 次执行器及真实服务器验收，不是让服务器自由决定研究方案。准备/CLI smoke 的 PASS 不代替参数映射、LoRA 数值、长上下文容量、开发题官方判分、预算触发及恢复验收。未通过不得开始正式计算。

## 来源

- [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)：thinking、采样、原生窗口及各评测说明；不宣称照搬了完整 Verified 官方 agent。
- [vLLM 0.22.0 Claude Code 接入](https://docs.vllm.ai/en/v0.22.0/serving/integrations/claude_code/)：Messages 接口与模型别名；未指定 CLI 精确版本。
- [vLLM #44000](https://github.com/vllm-project/vllm/issues/44000)：2.1.153 降级恢复的公开反馈；原报告中 ctx/msg 角色猜测后来被作者澄清，实际确认的是 system 角色错误。
- [Claude Code 安装与版本管理](https://code.claude.com/docs/en/setup)：指定版本、禁自动更新及二进制完整性核验；构建阶段记录平台、版本与官方校验信息，正式运行不在线安装。
