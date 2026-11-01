# 当前代码数据流与检查点

1. scripts/run_server.py → experiment_server.py：新 clone / 环境 / 授权 GPU 与控制器生命周期。
2. experiment_online.py：旧来源锁与 model_lock 核验、三个原始 split 复用/下载、新配置、新初始 adapter、环境预检。
3. experiment_data.py：Verified 20/100 拆分，SWE-smith 轨迹清洗、2000 候选、未入池轨迹对应 100 验证任务、独立开发题、泄漏检查。
4. harness.py + model.py + service.py：多轮模型/工具/Docker、实际 token IDs、原始工具证据、固定预算与官方判分；常驻底座切 LoRA。
5. scoring.py + parallel_scoring.py + score_features.py：8 次奖励方向、逐样本 LoRA 梯度、共享参考与投影、缓存恢复、NICE 排名。
6. subsets.py：分数升序二等分、高低各 512、共同 token 预算、单 seed。
7. experiment_design.py + experiment_training.py：4 jobs/12 points、两个 LR、真实有效 batch、原子发布节点、可信断点恢复。
8. batch_tuning.py + batching.py：全卡容量/吞吐测量、统一 micro-batch、按轨迹平均的累计梯度；不按成绩调 batch。
9. experiment_parallel.py：4 卡训练、其余验证，训练进程退出后全卡加入推理池；训练 worker 只写分支，主进程发布全局阶段。
10. experiment.py：独立 selection/benchmark/nice_retest 结果空间、三节点内选择、平分更早；冻结选择后仅 initial 和四个选中权重运行 100 题 benchmark 及原 NICE 20 题单次 greedy 诊断复测。
11. experiment_report.py + bundle_results.py：严格计数、模式与 checksum 审计、配对 CSV、图布局与合成标识、小型 zip。

新流程测试见 test_experiment.py、test_experiment_parallel.py、test_new_model_runtime.py、test_parallel_scoring.py、test_batched_training.py。原有数学、mask、数据格式、补丁、官方接口、异常与恢复测试继续作为基础回归；它们使用 legacy/development 配置，不定义本轮实验。
