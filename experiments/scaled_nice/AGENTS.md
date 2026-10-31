# 第二轮独立实验的运行规则

本目录是用户授权新增的独立实验。先读本目录 README 和仓库根目录的
`docs/HANDOFF_SCALED.md`。父目录原实验的 300/30/30、8×H200、四套 benchmark
及旧 handoff 不适用于本目录；其他证据完整性和安全规则继续适用。

- 默认设计由 `configs/scaled.yaml` 决定：3000 候选、100 主验证任务、每题 4 次
  NICE rollout、12 个 150 条子集、seeds 11/22/33；外部 benchmark 关闭。
- 默认 Qwen2.5-Coder-7B-Instruct，4 张 GPU 按阶段复用。训练是每卡一个独立
  分支；禁止把它解释为同一分支的 DDP，或悄悄改变 effective batch。
- 从本目录执行，使用独立 Python 环境、配置目录、存储根目录和空闲端口。
  不改父项目的源码、原 handoff、原结果或正在运行的第一轮实验。
- 评分和 SFT 只对 LoRA 参数求梯度；评分不执行 optimizer.step。
  所有分支从同一初始 adapter 开始。不按结果筛题、更换子集或删除失败。
- 新协议最多 75 次模型交互，预算耗尽时非空补丁自动送官方测试判分，
  保留原 stop_reason。NICE 和主评测必须使用同一协议。
- 本机仅 CPU 单元测试和显式 mock，不下载目标权重，不启动真实 GPU 任务。
  真实运行使用授权 Linux 服务器和完整 GPU UUID。
- 保留 source/config/model/data/environment 身份、checkpoint、RNG、日志及校验和。
  不篡改已有 snapshot 以强制恢复；需要改变研究定义时新建存储根目录。
- 改代码后运行 `python -m pytest -q`，流程变化后运行新的完整 scaled mock，
  检查 audit、CSV 和 PNG。明确区分 mock、微型随机模型接口测试和真实实验。
- 不提交数据、权重、缓存、运行状态或凭据；用户自行 commit/push。
