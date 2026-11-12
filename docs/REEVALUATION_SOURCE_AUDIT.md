# 重测文档与预算复核（2026-11-12）

本次只记录来源复核和修订建议，不改变已配置的预算。此前 REEVALUATION_PROTOCOL.md 中“已确定”表示项目选择，不代表预算已被证明充足，也不代表运行器已实现。

## 核心结论

1. 262144 原生上下文限制一次请求的输入加输出，不限制一题累计生成；工具观察、系统提示和历史 thinking 都占输入。压缩后可以继续运行，整题累计生成可超过窗口。
2. 当前额外设置的 262144 累计输出没有来自 Verified 或 Qwen 的官方依据，尚无真实 token 分布证明够用。250 次主回复平均约 1049 token 即会耗尽，还没计算辅助压缩生成。建议正式冻结前取消这项额外硬限制（仍完整记录用量），以 turn/墙钟及单次限额约束；若必须限制成本，应另外确定预算并报告预算截断，不能将窗口值直接当累计值。
3. Qwen 模型卡的 32768 输出是 QwenSWEBench 具体评测设置，不是 Verified 强制要求；1M 上下文下的 262144 reasoning/131072 final 建议也不能直接套入本次 256K 窗口。
4. 单次 32768 可保留为有官方编码评测实例支撑的起点，但必须统计 length/max_tokens 截断；缺少真实轨迹时不能保证充足。没有必要让每个请求预占全部窗口作为输出。
5. Claude Code 原生压缩需要显式匹配后端窗口、保留输出空间并验收实际触发点；配置环境变量的数值不一定就是触发时的真实 token 数。当前官方网页描述包含晚于固定 CLI 2.1.153 的功能，不能不查版本就复制所有新变量。

## 最贴近的公开实验

[What Does a Harness Buy? Tokens, Mostly.](https://arxiv.org/html/2610.04433v1)，附录 A，及其[开源仓库](https://github.com/YangzeLiu/what-does-a-harness-buy)，直接包含本地 Qwen3.8-27B、vLLM、Claude Code、SWE-bench Verified。作者使用 300 step；Qwen 服务窗口 131K，Claude Code 压缩变量设 110K，实测中位触发位置约 78K。其预算表未列整题 262144 累计输出上限，不能据此断言所有代码都不设该限制。它还指出不同框架墙钟不对称及外网检索答案问题，不能把其分数当成本项目预期。

该实验 CLI 为 2.1.259–2.1.273，不是本项目 2.1.153；其 Harbor 0.22.0 是评测编排框架，不是 vLLM 0.22.0。不能把它当作我们固定软件组合已通过验证的证据。具体 job 文件可用于下一步实现审查，不能只抄论文参数。

[腾讯 workbuddy-bench 实现说明](https://github.com/Tencent/workbuddy-bench/blob/main/configs/harnesses/claude-code/CONFIG.md)也直接使用真正 CLI，支持 Anthropic/vLLM 透传或到 OpenAI 的协议转换；明确处理 effort 映射和压缩，默认 256 turn。它证实可行的实现方式，不证明精确模型/LoRA/版本组合。

## 官方文件给出的边界

- [Qwen 模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)：thinking 推荐采样、保留历史、xhigh 默认、原生 262144；Pro 的 Claude Code 评测说明与自家 QwenSWEBench 设置须分开引用。
- [Claude Code 模型与压缩](https://code.claude.com/docs/en/model-config)、[环境变量](https://code.claude.com/docs/en/env-vars)：自定义模型不一定自动识别真实窗口；增加单次输出会减少压缩前可用输入。最新页并非固定旧版本的兼容保证。
- [SWE-bench 官方判分](https://www.swebench.com/SWE-bench/guides/evaluation/)：接收 instance_id/model_patch 并在容器测试。此判分协议不规定统一的 Claude Code turn、thinking 或整题 token 预算。

## 正式开跑前必须完成

目前仓库完整 600 次 CLI 执行器仍未接入。须实现并实测：真实请求有效参数与思考回传；固定 CLI 窗口识别/压缩；跨至少两次工具调用的 thinking 连续性；原 LoRA 与服务路由；工具命令/后台进程监督；非空补丁预算末送判；独立开发题官方判分；断点恢复。仅以“文件工具 smoke 成功”或 CPU 单测通过不能验收这些项目。

没有证据可将旧低分全部归因于 harness。已知的高比例空补丁/预算耗尽足以要求排查，但 thinking、预算、工具链及模型能力的贡献仍需固定协议后的逐题日志辨别。
