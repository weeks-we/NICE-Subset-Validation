# 在线准备与复用

正式入口和完整交接见 [HANDOFF](HANDOFF.md)，当前 default 是根目录 configs/main.yaml。使用新 clone、新 venv、新 storage 和 config-dir，保留旧实验。

```bash
python scripts/run_server.py --storage-root /data/nice-a-verified-v2 --reuse-from /data/old-experiment --config-dir configs/server-a-v2 --execute-server
```

默认 8 卡；需要时加 --model-path、--num-gpus、--gpu-uuids、--port。不带 --execute-server 只看计划，不下载或占 GPU。--reuse-from 指旧存储根，利用旧 source_revisions、raw source sidecar/checksum 和 model_lock 映射真实版本；绝不把 SWE-smith 数据集 revision 与同名 Git harness commit 混为一谈。

旧目录移动后，模型会在旧 storage/models 的有限范围按可信锁定位，数据尝试旧 storage/data 下原文件名；不会任意选同名模型，多个匹配需明确 --model-path。旧 prefix-limit 导出仅用于定位版本，完整 split 下载至新 storage，不覆盖旧文件。实际路径、版本、复用/下载原因见 data/reuse.json，最终条数和 checksum 见 data/source_lock.json。三个来源均需 prepared preflight 通过。

首次准备时旧 raw 文件确实不存在，也会在新 storage 按旧 ID/revision 自动下载完整 split；文件损坏或来源冲突则保留证据并停止，不能当作缺失悄悄覆盖。中断下载使用缓存和带锁 partial 恢复。模型缺失、镜像拉取失败、来源锁丢失等由服务器 Claude 按 HANDOFF 的恢复预案排查，无法核实版本不切 latest。冻结运行的路径/checksum 不能改写以绕过恢复检查。

新代码检查模型 ID、固定版本、文件 checksum、官方上下文容量，已有基座/tokenizer 不复制不重下。旧 JSONL 和 cache 可直接引用；没有的来源才按已固定版本获取。原始 full tool、Smith train、Verified test 准备后创建新初始 LoRA、划分清单和任务环境；旧 gradients、子集、训练 LoRA、结果、state 不复用。

SWE-bench/SWE-smith harness 固定 commit 安装在新环境；Docker 已有镜像层复用，新增任务仍可能下载或构建。预检与完整 smoke 通过才执行正式阶段。任何代码/环境/研究定义变更后新建 storage，禁止改旧 snapshot 强行恢复。

大模型仍要由磁盘加载进 GPU；模型目录校验读取大文件不是重新下载。后续 checkpoint 验证复用已驻留底座，只换 LoRA。
