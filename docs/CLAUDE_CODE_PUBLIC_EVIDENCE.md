# Claude Code 接入与评测公开证据扩展核对

核对日期：2026-11-12。本文是来源核对，不是已完成的服务器验收，不修改实验配置。优先论文原文、作者仓库、官方模型卡及代码；榜单聚合、新闻、搜索摘要和个人论坛仅用于找线索，不作为版本组合可用的证明。

## 直接相关的实验

| 来源 | 实际组合 | 已明确的设置 | 对本项目的边界 |
|---|---|---|---|
| [What Does a Harness Buy?](https://arxiv.org/html/2610.04433v1)，第3节、附录A；[代码](https://github.com/YangzeLiu/what-does-a-harness-buy) | Qwen3.8-27B / Qwen3.6-35B-A3B + 本地 vLLM + Claude Code + Verified | 300 step；vLLM 0.27.1 BF16；CLI 2.1.259–2.1.273；Qwen窗口131K，并实测压缩 | 同名基座和同benchmark的最直接例子；没有验证本项目 LoRA 或旧版本组合，论文还披露墙钟不对称 |
| [Lego-RL](https://arxiv.org/html/2608.17393v1)，表2；[代码](https://github.com/LegoX/Lego-RL) | Qwen3.5-35B-A3B、Qwen3.6-35B-A3B、KAT-Coder-V2.5-Dev及训练后模型 + Claude Code + Verified | 同一评测协议 temp=.7、200 turn、200K context；原生harness配合推理代理 | 是开源模型训练后仍可用真正CLI评测的直接例子，但不是本项目 LoRA 的数值正确性证明 |
| [MiniMax M2.1 官方卡](https://huggingface.co/MiniMaxAI/MiniMax-M2.1)，Evaluation Methodology Notes | MiniMax-M2.1 + Claude Code + Verified | 主表默认Claude Code；覆盖默认系统提示；4次运行取平均 | 有官方运行记录；CLI精确版本与完整预算未公开在该说明中，不能当完全复现配方 |
| [MiniMax M2 官方卡](https://huggingface.co/MiniMaxAI/MiniMax-M2)，评测说明 | M2（230B总/10B激活）+ Claude Code + Multi-SWE-Bench / SWE-bench Multilingual | 300 max step、8次平均 | 这是多语言SWE，不是Verified；M2的Verified采用OpenHands、100 step |
| [KAT-Coder-V2.5 报告](https://arxiv.org/html/2607.05471v1)，评测及gateway部分 | KAT模型 + Claude Code + 多项代码/agent benchmark | 强调统一harness/环境；gateway处理模型与工具协议及token一致性 | 支持协议适配的重要性；没有据此补造未公开的CLI版本或轮数 |

35B-A3B表示约35B总参数、约3B激活，不能当作35B稠密模型的计算成本。

## 可检查的实现与接入文档

1. [腾讯 workbuddy-bench 配置](https://github.com/Tencent/workbuddy-bench/blob/main/configs/harnesses/claude-code/_defaults.yaml)及[执行代码](https://github.com/Tencent/workbuddy-bench/blob/main/src/workbuddy_bench/agents/cc_agent.py)：默认256 turn，确实加入 `--max-turns` 并调用真实CLI。支持vLLM Anthropic透传或Anthropic→OpenAI代理；采样和template参数需要后端/代理处理。它是WorkBuddy任务集，不冒充Verified实验。
2. [vLLM 0.22.0官方接入](https://docs.vllm.ai/en/v0.22.0/serving/integrations/claude_code/)：通过ANTHROPIC_BASE_URL及模型别名使用本地模型，不指定精确CLI版本或Verified预算。
3. [QwenCloud官方接入](https://docs.qwencloud.com/developer-guides/clients-and-developer-tools/claude-code)：也是原生CLI配置Anthropic端点；云端有多档模型路由。本项目必须将辅助别名统一到当前adapter，不能原样复制云端混合模型设置。当前网页已出现更新模型，不能把它当旧模型静态协议。
4. [Unsloth本地接入](https://unsloth.ai/docs/basics/claude-code)：给出本地Qwen/Gemma经llama.cpp等连接CLI的教程；不是vLLM+Verified的完整验收证据。
5. [Ornith作者仓库](https://github.com/ornith-ai/Ornith-1)：Terminal-Bench 2.1采用Claude Code 2.1.126、单次131072输出，并修改Qwen模板；其Verified采用OpenHands。不能拼接两处设置作为Verified+Claude Code标准。
6. [Thunderdome作者仓库](https://github.com/signalnine/thunderdome)：公开Qwen3.6-27B NVFP4、本地vLLM、Claude Code经协议代理的记录；19道自有编程任务，不是Verified，也不是BF16 LoRA组合。

## 已找到的具体复现陷阱

- [MiniMax M2 #38](https://github.com/MiniMax-AI/MiniMax-M2/issues/38)：直接把problem_statement交给CLI未必复现厂商设置，讨论给出prompt模板和 `--max-turns 300` 的命令；不能把“有CLI”当prompt协议相同。
- [vLLM #44000](https://github.com/vllm-project/vllm/issues/44000)：CLI升级后的system消息角色兼容问题，存在退回2.1.153恢复的用户记录；不是所有新版本都不可用，更不是Qwen+LoRA完整组合证明。
- [GPUStack #5934](https://github.com/gpustack/gpustack/issues/5934)：Qwen3.6-27B原始vLLM端点可用，经路由失败；端点可用与代理正确是两项不同验收。
- Lego-RL[当前安装脚本](https://github.com/LegoX/Lego-RL/blob/main/scripts/setup_env.sh)默认vLLM v0.19.0并应用项目补丁。不能只抄版本号、忽略补丁，也不能把当前main默认当作论文全部实验冻结环境。其[兼容性表](https://lego-rl.pages.dev/docs/reference/compatibility)明确区分Tested和Experimental。
- Claude Code[当前官方CLI说明](https://code.claude.com/docs/en/cli-reference)默认没有max-turns上限；公开设置200/256/300反映不同预算，不是最佳值定理。固定旧CLI不能盲用较新官方网页全部变量。

## 对本项目的明确决定与未完成项

沿用已有任务镜像及原官方grader，不引入Harbor，不重训、不重选权重。当前250 turn仍是项目设置，不声称来自某份Claude Code的同值默认；公开200/256/300说明该量级存在先例，不证明250足够或300更优。

context=262144、单次32768、累计262144必须分开：原生窗口不推导累计限额。取消额外累计硬限制的建议尚未实施。thinking及模板、采样、辅助请求路由、压缩位置、超时与后台进程、停止后补丁判分，需要最终代码实现及实际端到端证据。

本次未找到“Qwen3.8-27B固定revision + 本项目r16 LoRA + vLLM0.22.0 + CLI2.1.153 + Verified”完全相同组合的公开验收。版本组合仍属待服务器验收候选，不能用公开接入教程或CPU测试声称已证实。准备阶段必须完成独立开发题真实CLI→指定adapter→多轮工具→补丁→官方grader及压缩/截断验收，失败停止；不以正式benchmark调参。完整600次执行器当前尚未接入。
