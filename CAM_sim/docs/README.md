# CAM_sim Docs Guide

这份索引页的目标不是重复所有细节，而是把 `CAM_sim` 的文档体系变成一个可导航、可维护、可扩展的入口。

## 1. 先看什么

如果你是第一次进入这个项目，建议按下面顺序阅读：

1. 项目入口：
   [README.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/README.md)
2. 核心硬件模型：
   [HARDWARE_MODEL.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/HARDWARE_MODEL.md)
3. 模拟 CAM 工作过程：
   [analog_cam_8x16bit_hash_match_zh.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/analog_cam_8x16bit_hash_match_zh.md)
4. 文档与工程化路线：
   [PROJECT_ENGINEERING_PLAN_zh.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/PROJECT_ENGINEERING_PLAN_zh.md)

## 2. 文档分层

当前 `docs` 推荐按“用途”而不是按“文件历史”来理解：

- 项目总览与规范：
  [README.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/README.md),
  [PROJECT_ENGINEERING_PLAN_zh.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/PROJECT_ENGINEERING_PLAN_zh.md)
- 设计与模型：
  [HARDWARE_MODEL.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/HARDWARE_MODEL.md),
  [analog_cam_8x16bit_hash_match_zh.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/analog_cam_8x16bit_hash_match_zh.md),
  [CAM_SIMULATOR_NOTES.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/CAM_SIMULATOR_NOTES.md)
- 结果与实验解释：
  [8x16bit_latency_comparison_zh.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/8x16bit_latency_comparison_zh.md),
  [28nm_cam_scaling_report_zh.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/28nm_cam_scaling_report_zh.md),
  [results/README.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/results/README.md)
- 论文与调研：
  [paper/README.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/paper/README.md),
  [paper/survey/CAM_SURVEY.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/paper/survey/CAM_SURVEY.md)

## 3. 各文档职责

为了避免同一个结论在多个文件里互相覆盖，建议把职责固定下来：

- `README.md`
  只负责项目入口、构建、运行、结果文件格式，不承载长篇设计推导。
- `HARDWARE_MODEL.md`
  作为当前实现口径的唯一模型说明，尤其是 baseline、cycle model、capacity policy、search/verify 定义。
- `analog_cam_8x16bit_hash_match_zh.md`
  负责解释模拟 CAM 是怎么工作的，偏机制讲解，不负责维护实验结果表。
- `8x16bit_latency_comparison_zh.md`
  负责多数据集对比结论，偏“读结果”。
- `28nm_cam_scaling_report_zh.md`
  负责 SPICE / DC / 缩放推导，偏“来源与校准”。
- `reports/*.md` 与 `reports/*.json`
  负责具体实验产出，不作为长期叙述性文档的主承载位置。

## 4. 推荐维护规则

- 规则 1：一个问题只保留一个主文档。
  比如“容量 baseline 的定义”应该以 `HARDWARE_MODEL.md` 为准，其它文档只引用。
- 规则 2：`docs/` 放解释，`reports/` 放产物。
  不要把一次性实验草稿长期堆在 `docs/` 根目录。
- 规则 3：新结果先产出到 `reports/`，再把稳定结论抽象进 `docs/`。
- 规则 4：baseline 一旦修正，要同时更新配置、报告、总文档，不允许只改一层。
- 规则 5：文件名尽量表达“主题 + 口径”，避免只看名字不知道它是设计、校准还是结果。

## 5. 现在最值得优先做的工程化动作

- 把“实验配置命名规范”和“baseline 命名规范”固化下来。
- 把 `reports/` 里的关键结果生成流程脚本化。
- 把 `docs` 的入口和角色分层稳定下来，减少“同一个问题要翻 4 份文档”的情况。
- 把容量实验、模拟前端校准、数字验证路径分成清晰的独立章节，而不是混在一个故事里。

## 6. 目录入口

- 设计类入口：
  [design/README.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/design/README.md)
- 结果类入口：
  [results/README.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/results/README.md)
- 论文类入口：
  [paper/README.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/paper/README.md)
