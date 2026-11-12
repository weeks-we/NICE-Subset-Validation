先读 docs/HANDOFF.md 顶部重测补充及 docs/reevaluation/DEPLOYMENT_SDK.md。用户已将调用方式改为Python Claude Agent SDK，保持250 turn和其他实验参数；SDK0.2.87、显式Claude运行程序2.1.153，vLLM0.22.0均须验收。

先核验并复用原基座/tokenizer、五份adapter、120题及已有任务与grader镜像。不重训、不重算NICE、不重选checkpoint、不更新旧testbed环境或改写旧结果。只有缺少的新SDK/vLLM运行层在构建阶段准备，三份Dockerfile见同目录；正式任务不联网下载安装。路径、镜像digest、官方二进制SHA256、GPU UUID和私有网络按平台证据填写。

按同目录 deployment.example.json 在Git外写机器配置，覆盖120题及原开发验收题的SDK派生镜像（记录原父镜像ID标签），复用原grader解释器及镜像。准备Internal=true专用任务网络和可达的私有网关地址；按DEPLOYMENT_SDK.md先运行 scripts/run_reevaluation.py preflight，全部真实验收通过后，用相同配置/存储目录执行run。该入口管理八卡服务、600题调度、预算监督、逐题恢复、官方判分和结果打包；不要调用旧训练入口。失败保留证据并报告，不绕过验收或缩预算。最终返回controller/results-*.zip及completion.json。不commit/push，不暴露凭据。
