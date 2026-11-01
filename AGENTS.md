# 项目协作规则：师兄 A 新实验

1. 当前唯一正式配置为 `configs/main.yaml`，唯一交接为 `docs/HANDOFF.md`。先读 README、METHOD、DATA、HANDOFF。B 独立项目已移除；`configs/legacy.yaml` 和 development.yaml 只用于历史回归，不是新实验入口。
2. 用户于 2026-11-01 授权新定义：SWE-smith 原冻结 tool 轨迹清洗后抽 2000 候选；按 NICE 升序二等分，高低各随机选 512，只有 sampling seed=11。两个学习率 1e-6/5e-6 × 高低两组，4 个独立 LoRA 分支，各训练一遍，有效 batch=8，128/256/512 条曝光保存阶段权重。
3. Verified test 固定抽 20 题作 NICE，8 次 rollout/题，最后 initial+四份选中权重在原题各 greedy 复测一次，单独作为诊断，不参与权重选择也不是独立测试；从剩余来源独立抽 100 题只作最终 benchmark。SWE-smith 另持出 100 个未入候选池的成功轨迹 ID 对应任务，用于 SFT 阶段选择；另 1 题仅开发验收。所有集合按 ID、显式来源家族及近似题面隔离。
4. 每个 (学习率, 高低组) 内比较三个节点，在同一 SWE-smith 验证集按官方 resolved 数选一个；平分选更早节点。最终 initial+四个选中权重共享 100 个 Verified benchmark 任务。不能按 benchmark 换题、调参、选权重或重试模型失败。旧四套迁移 benchmark 关闭。
5. 模型仍为固定 revision 的 Qwen/Qwen3.8-27B，正式加载类 AutoModelForImageTextToText；NICE 和 SFT 都只对 LoRA 求梯度，底座冻结。评分不做 optimizer.step。所有训练分支从共同初始 adapter 开始；同组的两个学习率使用相同成员和顺序。
6. 默认 8×H200-144GB：NICE 全卡独立分片；4 卡独立训练、其余卡验证已发布节点；训练结束后全卡推理。每卡一个服务、两个任务槽。物理 micro-batch 1/2/4 实测后统一冻结，保持有效 batch=8；不把多卡误称 DDP，不自动改学习率或训练轮数。
7. NICE、SFT 验证、最终 benchmark 与原题复测统一 75 轮、单次 32768/整题累计 262144 输出 token、上下文 131072、墙钟 3600 秒、单工具 300 秒。保留最近 5 个工具观察参与模型输入，完整观察仍落盘。预算耗尽且有非空补丁自动送官方判分，保留 stop_reason。这套 token/时间值是工程预算，不冒称 Verified 的统一官方标准。正式执行前必须通过 131072 上下文/32768 输出位置的真实 NICE 反向容量验收，不静默缩预算或在冻结后变更协议。
8. 本机只做 CPU 单测、微型随机模型与显式 mock；不能下载目标权重或启动真实 GPU/Docker 实验。真实执行使用明确 execution: server 和授权 GPU UUID。mock 指标不进入正式研究结论。
9. 新代码 clone 到新目录、新建 venv 和存储根。只复用核验过的基座/tokenizer、原始数据、来源锁与缓存，不复用旧梯度、选样、训练权重、运行快照或结果。不得改写旧 snapshot 来恢复。保留模型/数据/源码/环境/adapter/RNG/checksum 证据；同类基础设施错误最多两次重试。
10. 研究代码变化后运行全量 pytest；流程变化后另做全尺寸新 mock、串行评分对照、恢复、audit、CSV/PNG 与 zip 检验。真实权重、GPU 容量和容器判分必须由服务器 preflight/smoke 验收。
11. 不提交数据、权重、缓存、凭据、运行状态或本地报告；用户自己 commit/push，不替用户执行 Git 提交、推送或合并。B 未提交修改及旧证据已保留在被忽略的 runs/before-a-redesign-*。
