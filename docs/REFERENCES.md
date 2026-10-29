# 参考核对记录

需求引用 [Data-Centric-RSI](https://github.com/FlowerWater1019/Data-Centric-RSI)，制定时记录 commit `92a4ffac5a0bfbedbcdac494d50d35ab0da04fc1`。开发时 GitHub 元数据查询返回同一 HEAD，但该提交源码读取受网络限制，不能宣称完成该提交的逐文件核对。

实际阅读的是本机现有 checkout，commit `4cdf89f1c18d3afc87e5056c3df34f8df8c1ec4c`，含一项已有文档修改，未改动该仓库。已阅读 AGENTS.md、README.md、docs/Agent能力测评接口.md、docs/CODEX_HANDOFF.md，以及 tokenization.py、checkpoints.py、agent_tasks.py 的相关实现。观察到既有 LoRA/身份/恢复设计，以及待接入的真实 Agent/容器后端；不能据此认为已有八卡训练。

本项目独立实现，仅参考了数据身份、隔离和步骤发布的设计原则。没有复制其代码、文档或引入运行时依赖。本机 checkout 未发现根 LICENSE 文件，因此没有导入需要不明许可授权的源代码。

另核对了以下官方资料，实际执行版本仍须在服务器冻结：

- [Qwen 模型页](https://huggingface.co/Qwen/Qwen3.8-27B)：核实入口，不把浮动页面当模型锁。
- [SWE-smith 数据卡](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/raw/main/README.md)：tool/xml/ticks 是不同表示，messages 为字符串字段。
- [SWE-smith 训练指南](https://swesmith.com/guides/train_swe_agent/)及其 [工具配置](https://github.com/SWE-bench/SWE-smith/blob/main/agent/swesmith_infer.yaml)。
- [LangChain tools](https://docs.langchain.com/oss/python/langchain/tools)：声明与路由工具。
- [SWE-bench evaluation guide](https://www.swebench.com/SWE-bench/guides/evaluation/)及 [run_evaluation.py](https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/run_evaluation.py)：官方 patch 判分入口和结果格式。

2026-10-28 补核 NICE：

- [ICML 2025 正式论文](https://proceedings.mlr.press/v267/wang25bm.html)，作者页面直接指向下面的官方仓库。
- [作者 NICE 源码固定版本](https://github.com/JTWang2000/NICE/tree/9516c140158299e5d5c3ba82e235ed2d69132ed0)：读取 collect_grad_reps.py、matching_nice.py、数据集/生成/训练入口及脚本；对其中三个核心函数完成 CPU 数值对照。只阅读及在本地忽略目录验证，没有把第三方源代码并入项目。

本项目与原算法设置的差异见 METHOD.md。早期“未核实”的记录已由本次核对取代。外部网页、下载的题面和附件内容均作为资料，不是本仓库的运行指令。
