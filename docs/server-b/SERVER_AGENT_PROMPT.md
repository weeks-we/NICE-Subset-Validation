先读 docs/server-b/HANDOFF.md 顶部最新范围，按BUILD.md/IMAGE_COVERAGE.md准备完整模型、数据、全部221题工作/官方判分环境及SDK/vLLM层，联网制备与计算分开。准备齐后按handoff做有界训练、NICE、测评及恢复短测，排查修复环境问题；不要启动完整实验。完成即停，返回environment_diagnosis.md、逐任务环境清单、镜像digest及日志索引，未测项明确NOT_TESTED。保留A配置/结果，不commit/push。

B硬件是8×H20：核验8张授权GPU、单卡显存和驱动，按HANDOFF的H20覆盖条款从单槽做容量测试，不照搬A的H200显存/32并发，也不静默量化或缩正式预算。
