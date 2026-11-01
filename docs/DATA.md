# 当前数据来源与划分

唯一正式运行参数见 configs/main.yaml 与 [HANDOFF](HANDOFF.md)。

- 原始轨迹：SWE-bench/SWE-smith-trajectories/tool，沿用服务器旧固定 revision；不是下载 xml/ticks 或根据最新数据卡强制截成 5k。
- 配套任务：SWE-bench/SWE-smith/train，与轨迹 instance_id 严格连接。
- NICE 与最终 benchmark 原始任务：SWE-bench/SWE-bench_Verified/test。先冻结 20 题用于 NICE，再从剩余合格任务固定抽 100 题用于测试；不是同一套题换名字。
- 最终另以 initial+四份选中权重复测原 NICE 20 题，各一次 greedy；直接复用已冻结题目，不新增划分、不覆盖随机 rollout、不参与选权重。此项是诊断，不是独立测试。
- 清洗成功、完整、工具消息合法、≤32768 token 的轨迹，去重到独立任务后固定选 2000 候选。SFT 验证 100 题来自候选池以外的合格轨迹 ID 对应任务；另留 1 题开发验收。
- 在选样/模型执行前做 ID、source_instance_id 显式来源家族、精确/近似题面隔离。候选不能包含任何 NICE/benchmark/SFT 验证/开发任务。缺少对应任务、样本不足或预算不可行明确停止，不复制、截断或按模型结果换题。

显式家族信息同时合并轨迹与任务两侧并计算关联闭包，防止仅轨迹侧的来源信息在 ID 连接时丢失。规范化家族 ID 单独记录，不改写原始记录的来源 SHA256。

来源归一化继续支持 JSON 编码 messages、官方文本块列表与原生 function/parameter 工具格式。消息保留原始证据；assistant-only mask 由真实 tokenizer/chat template 定位，完整工具观察不监督。工具调用和返回 ID 关系要验证；不能静默丢弃不支持的多模态内容。

模型冻结前后的数据文件带 `.source.json`，保存源 ID/revision/split、分片列表、无前缀 limit、条数、SHA256。复用旧数据时核对后直接引用原绝对路径，不改写旧文件；新的 source_lock、reuse.json 与清洗/划分清单写在新 storage。

各 split 的原始完整下载与正式使用量不同：完整 tool / Smith train / Verified test 只准备一次；正式使用 2000 候选、20 NICE、100 SFT 验证、100 benchmark、1 开发。任务镜像只准备实际使用的任务，不是全部数据集镜像。

不能从这些检查推出基座预训练绝无污染；来源记录和未知项必须如实保留。
