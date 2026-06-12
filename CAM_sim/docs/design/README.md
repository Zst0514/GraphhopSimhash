# Design Docs

这个目录对应的是“设计与模型解释层”。

它的职责不是保存每一次实验结果，而是回答：

- 这个系统想模拟什么硬件对象；
- 数字路径和模拟路径分别怎么工作；
- 当前行为级模型的边界在哪里。

## 当前主文档

- `CAM` 选型总结：
  [CAM选型总结.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/design/CAM选型总结.md)
- 硬件模型总说明：
  [HARDWARE_MODEL.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/HARDWARE_MODEL.md)
- 模拟 CAM 的 8x16bit 匹配机制解释：
  [analog_cam_8x16bit_hash_match_zh.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/analog_cam_8x16bit_hash_match_zh.md)
- 外部 CAM 模拟器与参考说明：
  [CAM_SIMULATOR_NOTES.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/CAM_SIMULATOR_NOTES.md)

## 这一层建议放什么

- 行为模型定义
- RC / threshold / verify 机制解释
- baseline 与容量模型定义
- 配置字段的语义说明

## 这一层不建议放什么

- 一次性实验草稿
- 某次跑数的临时结论
- 没有稳定下来的对比表
