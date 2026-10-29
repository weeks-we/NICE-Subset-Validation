# 数据来源与冻结

2026-10-27 更新：规范化器支持官方文本块列表，按原顺序直接拼接文本，保留原始消息；非文本块明确拒绝。内部工具参数统一为对象。连续工具观察按模板中的完整分组定位，工具返回仍不监督。生成端支持 Qwen 原生 function/parameter 格式以及兼容 JSON 格式，参数中的换行和缩进不被 strip。

官方来源：

- [SWE-smith trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)：仅 tool split。
- [SWE-smith 任务集](https://huggingface.co/datasets/SWE-bench/SWE-smith)：train split 为主 validation 来源，与训练候选按任务划分。
- 外部迁移评测为 Verified、Pro Public V2、Live Python full 快照、Terminal-Bench 2.0，各固定 50 题，不参与 NICE 或训练。来源、执行器和隔离规则见 [TRANSFER.md](TRANSFER.md)。
- [官方训练说明](https://swesmith.com/guides/train_swe_agent/)。
- [官方工具配置](https://github.com/SWE-bench/SWE-smith/blob/main/agent/swesmith_infer.yaml)和 [SWE-agent editor](https://github.com/SWE-agent/SWE-agent/blob/main/tools/edit_anthropic/bin/str_replace_editor)。

本地没有下载或解析真实数据，没有真实候选 ID、manifest 或 SHA256 可交付。`configs/sources.pending.json` 明确记录此状态。合成 fixture 在 `nice_validation.data.fixture()` 中生成，不能冒充官方记录。

## 获取

推荐使用 [ONLINE.md](ONLINE.md) 的 `configure_server.py` 与 `prepare-online`：自动固定训练/主任务/模型、四个迁移集和所有 harness 版本；下载所需 HF split 与固定原生任务 checkout，发布来源锁，再接入 prepare-data。默认使用指定存储根目录下的绝对路径。中断重跑保留来源和 checksum；恢复时逐行核对已存前缀，不完整尾行单独留证。独立获取命令如下。

先运行 `python scripts/resolve_sources.py --output data/source_revisions.json`。把实际 commit 填入本机 server 配置的 `data.training_revision`、`data.validation_revision`、`model.revision`。脚本只查询元数据，不下载权重。`main` 只能用于发现 commit，不能作为正式冻结身份。

`fetch-data` 先列出固定 revision 的目标 parquet 文件和预计完整分片字节数。stream 模式只遍历所需 split；download 模式只下载匹配该 split 的 parquet。两种模式都不读取 xml/ticks。`--limit` 只限原始训练记录，validation 始终完整获取。stream 的实际传输量受 parquet 分块影响，不等于最终 JSONL 大小。

来源文件为 JSONL：训练保留原始 messages、instance_id、traj_id、resolved、model、patch 等列；validation 保留官方完整元数据供私有判分。`*.source.json` 和 `source_lock.json` 保存 commit、分片列表、读取上限、记录数及原始文件 SHA256。中断保存 `.partial` 和对应来源身份；恢复跳过已保存的前缀，不更换 revision。已有来源必须匹配 checksum 和 sidecar，不能覆盖异源数据。

运行 `python scripts/inspect_data.py --source data/training.jsonl --rows 3` 检查实际消息字段、工具名和参数。当前协议支持 `bash(command)`、`str_replace_editor` 的 view/create/str_replace/insert/undo_edit，以及无参数 submit。未知工具或参数直接过滤并汇报，不通过改名假装可转换。

旧导出若只有一个未返回的工具调用，允许为缺失 ID 的观察补上该唯一 ID。多个未返回调用且观察缺 ID 时拒绝。终止 submit 允许没有工具观察。规范格式把 JSON 字符串 arguments 解析为对象，保留原始 messages 和转换说明。工具调用/观察的全部内容可回溯到原始来源记录。

## 清洗与选择

抽取 SWE-smith 主集之前先校验原生字段，排除缺失、空白的 `problem_statement`，将排除数量和任务 ID 写入 `manifests/main_task_filter_report.json`。2026-10-28 下载的前三条真实记录中存在一条空题面，因此不能从未经筛选的整个 split 直接抽题。这个过滤在任何模型成绩产生前执行，不补写题面，不据结果换题；过滤后还须有至少31题供主集和独立开发题使用。

`schemas.py` 检查任务身份、题面、镜像和测试列表；Verified 检查 base_commit/version，Live 检查 base_commit/test_cmds/log_parser。Live 没有 version 是合法原生格式。JSON 字符串形式的测试名列表可解析为列表，保留其他原始字段。迁移集遇到必需字段缺失时直接失败，不悄悄过滤后替换抽样总体。

只保留源 `resolved is true`，不声称本机重新验证了训练成功性。检查消息结构、工具配对和完整终止，按 instance_id、traj_id 稳定排序去重，每题取首条合格轨迹。使用目标 fast tokenizer 和 chat template 统计完整输入与监督 token，最大 32768，不截断。

先冻结迁移 200 题，再从 SWE-smith 任务源按仓库比例选择主 30 题与独立开发题；主任务也排除迁移题相同 ID/重复问题。训练清洗检查这些持出题的相同 ID、显式 source_instance_id、规范化用户问题和长文本相似度（阈值 0.9）。不把整个 SWE-smith 同源任务集都排除，否则会误删可用于训练的其余任务。记录缺失 family 元数据的数量；未知同源变体和模型预训练污染无法完全识别。额外 patch 列只用于来源指纹，不拼入模型输入。

ID/显式同源检查按两侧instance_id与source_instance_id的并集求交，即使只有一侧声明同源也排除；主任务相对迁移集、训练相对全部持出集以及不同迁移套件之间均检查该边界。排除迁移重叠后仍须至少31个合格主任务，才能选择30主题和1独立开发题。

对合格不同训练任务按 repo 数量使用最大余数法分配 300 个名额，组内按稳定 seed=42 抽样。SWE-smith 主 validation 同法选 30 题并冻结同 repo 候补顺序；独立开发题与训练候选不重合。官方数据表中的不同动作表示数不能相加当作独立任务数。

产物包括 `source_lock.json`、`filter_report.json`、`candidates.json`、`validation.json`、`validation_reserve.json`、`mapping.json`、`protocol.json` 和 `checksums.json`。

## 环境预检与最终冻结

`prepare-environments` 在不调用模型的情况下获取环境镜像并固定 digest。SWE-smith 先在可信准备容器拉取任务分支、记录真实 SHA，再隔离隐藏测试历史并发布 Agent 镜像；正式 SWE Agent 容器没有宿主挂载、凭据、Docker socket 或网络。grader 在独立环境按相同 SHA 恢复测试，模型消息只含公开问题和工具观察。Pro/Terminal 遵循原生 Harbor 任务的网络策略，并核对实际服务镜像身份。

若最终冻结前环境预检失败，可运行：

```bash
python -m nice_validation --config configs/server.local.yaml replace-validation --task <failed_instance_id>
python -m nice_validation --config configs/server.local.yaml prepare-environments
```

替换只使用已有预检失败证据和第一个合格同 repo 候补；候补排除训练候选、开发题及相同/近似训练问题，保存旧题、新题、原因与原预检报告。没有合格同 repo 候补则停止。不接受“模型做不出”作为理由。最终冻结或任一主 rollout/evaluation 启动后禁止换题。四个迁移集不自动替换任务，环境故障保留分母并修复/重试。

`agent.smoke_task_path` 指定一个不在主 30 题中的独立开发任务。环境准备时一并获取其镜像，但不会计入主 episode 数或用于改主选题。

在线准备按 instance_id 从 SWE-smith 剩余任务中选最后一个作为开发题，并在训练清洗时排除。它写入 manifests/development_task.json 并同步到配置指定路径，保存独立身份记录，不读取模型成绩。

真实数据、镜像和 tokenizer/模型保留在服务器。Git 只保存获取代码、来源入口、合成 fixture 和待冻结说明。结果包通过 `scripts/bundle_results.py` 导出已冻结 ID 和 SHA256，不包含大文件。
