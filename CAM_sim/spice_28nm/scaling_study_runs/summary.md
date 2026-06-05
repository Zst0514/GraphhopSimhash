# 28nm CAM 前端缩放实验

- 模型甲板：`TSMC28 CRN28HPC+ top_tt`
- 供电/偏置点：`VDD=0.90 V, VREF=0.60 V, VEVAL=0.60 V`
- 偏置缩放说明：`0.9 V / 0.6 V` 是从论文 `1.2 V / 0.8 V` 工作点归一化得到的 28nm 偏置代理。
- 完整搜索时间估算中的感测时间假设：`20.0 ps`

## 实测 crossing time

| 字长 (bit) | ML 电容 (fF) | 预充到 0.9VDD (ps) | 普通 CAM d=1 到 Vref (ps) | HD-CAM d=3 到 Vref (ps) | HD-CAM d=2 到 Vref (ps) | HD d2-d3 窗口 (ps) |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 3.800 | 30.781 | 60.539 | 63.589 | 69.047 | 5.458 |
| 64 | 13.400 | 105.907 | 81.214 | 87.698 | 105.209 | 17.511 |

## 完整搜索时间估算

下面的估算采用：

- `search_time_exact = t_precharge90 + t_exact_d1 + t_sense`
- `search_time_hdcam = t_precharge90 + t_hdcam_d3 + t_sense`
- `500 MHz 预算 = 2000 ps`

| 字长 (bit) | 普通 CAM 完整搜索 (ps) | 普通 CAM 推导 Fmax (GHz) | HD-CAM 完整搜索 (ps) | HD-CAM 推导 Fmax (GHz) | 普通 CAM 满足 500MHz? | HD-CAM 满足 500MHz? |
|---:|---:|---:|---:|---:|---|---|
| 16 | 111.320 | 8.983 | 114.370 | 8.744 | 是 | 是 |
| 64 | 207.121 | 4.828 | 213.606 | 4.682 | 是 | 是 |

## 告警统计

| Case | 字长 (bit) | Spectre 告警数 |
|---|---:|---:|
| exact_d1 | 16 | 5 |
| exact_d1 | 64 | 5 |
| hdcam_d2 | 16 | 5 |
| hdcam_d2 | 64 | 5 |
| hdcam_d3 | 16 | 5 |
| hdcam_d3 | 64 | 5 |
| precharge90 | 16 | 5 |
| precharge90 | 64 | 5 |
