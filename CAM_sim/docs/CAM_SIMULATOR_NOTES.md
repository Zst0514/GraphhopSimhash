# CAM 模拟器说明

这个子项目的第一版没有直接内置第三方 CAM 工具。  
它当前只是预留了配置字段，后续可以用外部 CAM 工具给出的能耗、面积、时延结果来替换这些代理参数。

## CAMASim

CAMASim 是一个开源的 CAM 加速器模拟器。它的 README 主要覆盖这些能力：

- `cam.write()` 与 `cam.query()` 的功能级仿真；
- 性能评估；
- 可配置的 mapping / search 策略；
- 通过 EvaCAM 或用户自定义数据进行硬件代价估计。

项目地址：<https://github.com/menggg22/CAMASim>

在当前项目里的推荐用法：

1. 保留现有 C++ 模拟 CAM 模型，作为功能参考模型。
2. 用 CAMASim 为可比的 `16-bit / 8-bank / threshold-search` 场景估算搜索、写入延迟和能耗。
3. 将校准后的参数写回 `analog_cam_cpp/configs/camasim_cost_stub.json`，或者新建一份配置文件。

## EvaCAM

EvaCAM 是一个 C++ 的 CAM 电路 / 架构评估工具。公开 README 表示它支持 TCAM、analog CAM 和 multi-bit CAM；但当前公开的 v1 发布信息里，首先放出的主要还是 TCAM exact-match 版本，ACAM / MCAM 的更新是后续计划。

项目地址：<https://github.com/eva-cam/EvaCAM>

在这里更合适的用法是：

- 把 EvaCAM 视为一个“校准后端”，而不是硬依赖；
- 如果拿到了可用的 ACAM / threshold-search 校准模型，就用它输出的时延、能耗、面积去替换当前的 proxy 数值。

## NVSim-CAM

NVSim-CAM 是文献中提出的电路级模拟器，面向新型非易失 CAM 设计。  
它对 CAM 的能耗 / 面积建模很有参考价值，但当前仓库第一版并没有把它直接接进来。

参考页面：  
<https://researchportal.hkust.edu.hk/en/publications/nvsim-cam-a-circuit-level-simulator-for-emerging-nonvolatile-memo/>

## 当前默认解释

默认的模拟 CAM 配置里写的是：

```json
{
  "calibration": "spice_28nm_16b_timing_proxy"
}
```

这意味着，当前生成的报告应该被理解为：

- `28nm / 16-bit` 前端 timing 已经对齐到一组 SPICE 校准代理值；
- 架构趋势对比；
- 行为级近似结果；
- 不是已经完成电路校准的测量级结论。

换句话说，当前默认结果适合做：

- 数字路线与模拟 CAM 路线的结构性比较；
- 参数扫面；
- 敏感性分析；
- 系统层面的早期取舍。

如果你想复现旧的保守口径，仓库里还保留了一份：

- `analog_cam_cpp/configs/analog_cam_legacy_proxy.json`

它对应之前的 `cam_search_cycles = 3`、`40/100/20 ps` 旧 proxy。

如果要把结果上升到更强的硬件结论，需要把这里的 proxy 参数替换成经过外部工具或电路模型校准后的数值。
