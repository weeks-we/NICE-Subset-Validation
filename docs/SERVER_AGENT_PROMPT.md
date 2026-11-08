# 发给师兄的 agent

请先读 AGENTS.md、docs/HANDOFF.md、docs/DEPLOYMENT.md。你负责从前置环境核验、镜像构建/装载到实验执行、恢复和最终汇报的全流程；公司内部地址、平台参数、挂载和 GPU UUID 请按实际情况配置，普通工程问题自行处理并记录。

先盘点已完成内容，优先复用核验过的模型、原始数据和任务镜像，不重新抽题、不重复下载已有资源。当前 r4 已有 NICE160，后续 selection 环境准备报 Stale file handle，请先修存储并判断能否在原身份下恢复；不要默认重跑前置步骤。若应用新源码/环境导致身份变化，不改旧 snapshot 或搬旧结果冒充续跑，保留证据并说明需要新运行。

运行环境参考 docs/Dockerfile，在构建阶段装好固定依赖；按 docs/DEPLOYMENT.md 用 scripts/prepare_task_images.py 制备、发布、内网装载并核验 221 道冻结任务的 Agent/grader 镜像。正式计算使用 preinstalled/offline 配置，缺资源先报告，不临时到外网安装或补镜像。验证 Docker SDK/CLI 使用同一 daemon，挂载路径一致，完成必要真实验收后继续 HANDOFF 的实验。所有研究参数保持不变。最后返回环境/镜像核验情况、实际恢复进度，完成后交付 results zip 和 completion.json；不要 commit/push。
