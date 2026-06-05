# CAM_sim 项目系统化与工程化路线图

这份文档回答的不是“某个具体实验结果是多少”，而是：

- 这个项目现在为什么会让人读起来费劲；
- 应该怎样把它变成一个可复现、可维护、可扩展的工程；
- 接下来按什么顺序整理最划算。

## 1. 目前的主要问题

从工程化角度看，当前项目已经有不错的研究内容，但还带着比较明显的“研究草稿阶段”特征。

### 1.1 文档角色混在一起

现在的文档里同时混着几类内容：

- 项目入口说明
- 硬件模型定义
- 机制解释
- SPICE / DC 校准
- 实验结果总结
- 临时口径修正

这会导致一个问题：
同一个概念，比如 baseline、容量、search cycle、verify cycle，会在多个文档里重复出现，而且不一定同时更新。

### 1.2 配置、代码、报告之间缺少强约束

现在项目里已经有：

- 多种 `config`
- 多份 `reports/*.json`
- 多份 `reports/*.md`
- 若干手工整理后的 `docs/*.md`

但三者之间的关系还不够制度化。  
一旦 baseline 定义变了，就容易出现：

- 配置已经更新
- 代码已经更新
- 某些报告还是旧的
- 某些文档还沿用旧解释

### 1.3 baseline 治理不够严格

这次 `per_hash_fifo` 被误当成“无限大 baseline”，就是一个典型例子。  
问题不在于研究中做了近似，而在于：

- 这个近似没有被明确标注成 `legacy heuristic`
- 它后来又被拿去做容量结论

这说明项目已经到了需要“baseline 治理”的阶段。

### 1.4 结果产出流程还偏手工

现在很多关键结论已经有了，但生成链路还不够工程化：

- 哪个配置是正式 baseline
- 哪个报告是当前有效版本
- 哪个对比是用来回答哪个问题

这些仍然主要靠人脑记忆，而不是靠目录结构、脚本命名和文档入口来约束。

## 2. 推荐的工程分层

我建议把 `CAM_sim` 以后长期固定成下面五层：

### 第 1 层：项目入口层

目标：让第一次进项目的人能在 5 分钟内知道项目做什么、怎么跑、先看什么。

应由这些文件承担：

- `/CAM_sim/README.md`
- `/CAM_sim/docs/README.md`

### 第 2 层：规范与模型层

目标：定义“项目当前口径是什么”。

这一层应该只放“规范”，不放一次性草稿。

建议以这些文件为主：

- `/CAM_sim/docs/HARDWARE_MODEL.md`
- `/CAM_sim/docs/CAM_SIMULATOR_NOTES.md`

这一层应该负责定义：

- 算法接口
- 数字 / 模拟模型边界
- baseline 定义
- 容量模型定义
- search / verify / update 的口径

### 第 3 层：设计解释层

目标：解释“为什么这样设计”，帮助人理解机制。

建议以这些文件为主：

- `/CAM_sim/docs/analog_cam_8x16bit_hash_match_zh.md`
- `/CAM_sim/docs/design/README.md`

这一层不负责维护最终结果表，而是负责把机制讲明白。

### 第 4 层：实验结果层

目标：回答“跑出来是什么结果”和“怎么解释这些结果”。

建议分成两部分：

- `docs/` 里的稳定总结文档
- `reports/` 里的原始实验产物

对应文件可以是：

- `/CAM_sim/docs/8x16bit_latency_comparison_zh.md`
- `/CAM_sim/docs/28nm_cam_scaling_report_zh.md`
- `/CAM_sim/reports/*.json`
- `/CAM_sim/reports/*.md`

关键原则是：

- `reports/` 保存实验事实
- `docs/` 保存稳定结论

### 第 5 层：论文与外部参考层

目标：让项目的外部背景、参考论文、survey 有固定位置，不干扰主工程说明。

对应目录：

- `/CAM_sim/docs/paper/`

## 3. 推荐的“单一事实源”制度

后面如果想把这个项目做得越来越稳，最重要的是明确每个问题到底以哪份文件为准。

建议固定成下面这样：

- “当前硬件模型定义”  
  以 `docs/HARDWARE_MODEL.md` 为准

- “项目怎么构建、怎么运行”  
  以 `README.md` 为准

- “某次实验到底跑出了什么”  
  以 `reports/*.json` 为准

- “某条结果怎么解释”  
  以 `docs/*_comparison*.md` 或 `docs/*_report*.md` 为准

- “相关论文怎么分类”  
  以 `docs/paper/survey/CAM_SURVEY.md` 为准

这样做的好处是，一旦后面结论冲突，就知道该改哪一层，而不是每份都重写。

## 4. 推荐的命名和目录规则

### 4.1 配置命名

建议以后配置统一遵循：

`<impl>_<verify_mode>_<storage_policy>_<capacity>_<freq>.json`

例如：

- `digital_per_head_global_unbounded_500mhz.json`
- `digital_per_head_global_lru_512kb_500mhz.json`
- `analog_rc_global_unbounded_500mhz.json`

这样一看名字就知道：

- 是数字还是模拟
- verify 方式是什么
- 存储策略是什么
- 容量是多少
- 时钟是多少

### 4.2 报告命名

建议报告名也跟配置名对齐，避免现在这种“有些文件名像结论，有些像配置快照”的混搭。

### 4.3 `legacy` 明确入名

所有已经不建议继续作为 baseline 的旧口径，都应该显式带上：

- `legacy`
- `heuristic`
- `deprecated`

至少三者之一。

比如旧 `per_hash_fifo` 的基线结果，如果还需要保留，就不应该再叫“默认 500mhz baseline”，而应该叫：

- `*_legacy_per_hash_fifo_*`

## 5. 推荐的实验工程化动作

### 5.1 建立实验矩阵脚本

现在项目已经很适合加一个统一 runner：

- 输入：数据集列表、配置列表
- 输出：报告 JSON、对比 Markdown、汇总表

这能把“人手点命令”变成“可重复实验”。

### 5.2 把对比脚本升级成正式工具链

现在已经有：

- `compare_reports.py`
- `compare_capacity_reports.py`

下一步可以把它们变成：

- 统一输入 schema
- 统一输出列名
- 自动生成 summary 表

### 5.3 区分 generated 与 curated

建议制度化区分：

- `reports/`
  默认认为是 generated 或半 generated
- `docs/`
  默认认为是 curated

这样就不会把“临时实验草稿”误当成“长期主文档”。

### 5.4 加最小 CI

推荐至少加三类自动检查：

- `ctest`
- 一条小 trace 的 digital smoke run
- 一条小 trace 的 analog smoke run

如果再进一步，可以加：

- 对关键报告字段做 schema 校验
- 对 baseline 名称做 lint

## 6. 推荐的文档维护规则

### 规则 1

新实验先落到 `reports/`，结论稳定后再上升到 `docs/`。

### 规则 2

一份文档只承担一种主要职责，不要同时兼顾“项目入口 + 模型定义 + 实验结论 + 未来计划”。

### 规则 3

凡是会影响实验解释的口径修正，必须同时更新三处：

- config
- report summary
- `HARDWARE_MODEL.md`

### 规则 4

任何 baseline 修改，都要在文档里明确写出：

- 为什么旧 baseline 不再合适
- 新 baseline 的定义是什么
- 哪些历史结果仍然有效，哪些只可作 legacy 参考

## 7. 建议的下一阶段优先级

如果按投入产出比排序，我建议这样做：

### P0

- 统一 baseline 命名和定义
- 统一容量实验口径
- 固定 docs 入口页

### P1

- 做一个实验矩阵 runner
- 自动生成多数据集汇总表
- 把 `reports/` 的关键结论自动抽成 `summary.md`

### P2

- 拆分“主 CAM 容量”和“candidate CAM 容量”两个瓶颈
- 给 entry 大小建一个更显式的位级模型，而不是只用 `node_entry_bytes`
- 把数字 / 模拟的配置 schema 文档化

## 8. 一句话结论

这个项目现在最需要的，不是再多写几份解释文档，而是把下面三件事固定下来：

- 什么是当前正式 baseline
- 哪份文档是哪个问题的唯一解释来源
- 哪些结果是生成产物，哪些结果是稳定结论

只要这三件事固定住，`CAM_sim` 就会从“研究过程中不断长出来的代码和文档集合”，变成一个真正可维护的硬件建模工程。
