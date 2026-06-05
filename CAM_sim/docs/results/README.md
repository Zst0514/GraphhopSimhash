# Results Docs

这个目录对应的是“结果解释层”。

这里的核心原则是：

- `docs/` 负责写稳定结论
- `reports/` 负责保存实验产物

## 当前主文档

- 多数据集耗时对比：
  [8x16bit_latency_comparison_zh.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash-main/CAM_sim/docs/8x16bit_latency_comparison_zh.md)
- 28nm SPICE / DC / 缩放报告：
  [28nm_cam_scaling_report_zh.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash-main/CAM_sim/docs/28nm_cam_scaling_report_zh.md)
- 512KB 容量实验总结：
  [capacity_lru_512k_summary.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash-main/CAM_sim/reports/capacity_lru_512k_summary.md)

## 对应实验产物目录

- 汇总报告：
  [/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash-main/CAM_sim/reports](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash-main/CAM_sim/reports)

## 建议维护方式

- 新实验先写到 `reports/*.json` 和 `reports/*.md`
- 只有在结论稳定后，才把结果抽象进 `docs/*.md`
- 如果 baseline 或配置口径更新，优先更新：
  1. `reports/` 中对应实验
  2. `HARDWARE_MODEL.md`
  3. 这里列出的稳定结果文档
