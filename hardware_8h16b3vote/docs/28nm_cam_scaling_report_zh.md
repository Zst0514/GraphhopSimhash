# 28nm CAM/HD-CAM 缩放到 500MHz 的验证报告

## 1. 问题定义

目标是回答三个问题：

1. 以 Garzon 等人在 2025 JSSC HD-CAM 论文给出的 `65nm / 64bit / 125MHz / 1.2V` 为参考点，缩放到 `28nm` 后，`500MHz` 是否可达。
2. `64bit -> 16bit` 的缩放，是否是达到 `500MHz` 的必要条件。
3. 在同一时钟频率下，`数字 CAM` 和 `模拟 HD-CAM` 谁更合适。

本报告使用两条证据链：

- `SPICE`：测 28nm 行级前端的 match-line 预充和放电 crossing time。
- `DC`：测 28nm 标准单元库下 `XOR + popcount + threshold` 的数字 Hamming compare 时序。

## 2. 论文基线

从论文 `A 128-kbit Approximate Search-Capable CAM (CAM) With Tunable Hamming Distance` 抽取到的关键基线如下：

- 工艺：`65nm CMOS`
- 宏规模：`4 x (512 x 64)`，总计 `128 kbit`
- 工作点：`1.2V`, `125MHz`
- 机制：通过 `Veval` 控制 ML 放电速率，通过 `Vref` 设定 HD 判决边界
- 论文中用于目标 `HD=2`/`HD=5` 的代表性偏置点集中在 `Vref=0.8V`，`Veval=0.8V / 0.6V`

需要特别说明的是：当前环境没有论文使用的 `65nm PDK`，所以“`65nm -> 28nm` 的纯工艺缩放倍率”不能直接实测；我们能实测的是 `28nm` 端点，并据此判断 `500MHz` 是否还有充足余量。

## 3. 28nm SPICE 设置

SPICE 使用本机可用的 `TSMC28 CRN28HPC+ top_tt` 模型甲板：

- 模型甲板：`/opt/synopsys/TSMC28/.../toplevel_1d8.scs`
- 供电/偏置：`VDD=0.9V`, `VREF=0.6V`, `VEVAL=0.6V`

这里的 `0.9V / 0.6V` 不是论文原值照搬，而是把论文 `1.2V / 0.8V` 做了归一化缩放后得到的 `28nm core-safe proxy`。这样做的原因是仓库旧的 `1.2V` 28nm 试验会触发 LVT core device 的 over-voltage 警告，不能作为可信结论。

Match-line 电容沿用仓库现有 16-bit proxy 的线性缩放：

- `C_ML = 0.6 fF + 0.2 fF/bit * word_bits`
- 因此：
  - `16bit -> 3.8 fF`
  - `64bit -> 13.4 fF`

本次 SPICE 只验证最核心的行级前端，不包含完整 bank、replica row、MLSA 布局寄生和全宏布线。

输出文件：

- [SPICE 汇总表](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/hardware_8h16b3vote/spice_28nm/scaling_study_runs/summary.md)
- [SPICE 原始 JSON](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/hardware_8h16b3vote/spice_28nm/scaling_study_runs/summary.json)

## 4. SPICE 结果

### 4.1 行级 crossing time

| Word bits | ML cap (fF) | Precharge to 0.9VDD (ps) | Exact d=1 to Vref (ps) | HD d=3 to Vref (ps) | HD d=2 to Vref (ps) | HD window d2-d3 (ps) |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 3.800 | 30.781 | 60.539 | 63.589 | 69.047 | 5.458 |
| 64 | 13.400 | 105.907 | 81.214 | 87.698 | 105.209 | 17.511 |

解释：

- `Exact d=1 to Vref`：普通 CAM 从 `d=0/1` 分界上需要的最短放电时间。
- `HD d=3 to Vref`：以 `HD<=2` 为目标时，`d=3` 首次掉到 `Vref` 的时间。
- `HD window d2-d3`：`d=2` 和 `d=3` crossing time 的间隔，代表近似判决窗口。

### 4.2 估算的完整前端搜索时间

为了把 SPICE crossing time 转成可对比频率，这里采用保守估算：

- `search_time_exact = t_precharge90 + t_exact_d1 + t_sense`
- `search_time_hdcam = t_precharge90 + t_hdcam_d3 + t_sense`
- `t_sense = 20 ps`

| Word bits | Exact full search (ps) | Exact implied Fmax (GHz) | HD full search (ps) | HD implied Fmax (GHz) | 500MHz exact? | 500MHz HD? |
|---:|---:|---:|---:|---:|---|---|
| 16 | 111.320 | 8.983 | 114.370 | 8.744 | yes | yes |
| 64 | 207.121 | 4.828 | 213.606 | 4.682 | yes | yes |

### 4.3 SPICE 结论

1. `500MHz` 对 28nm 行级前端来说不是瓶颈，`64bit` 和 `16bit` 都远小于 `2ns` 周期预算。
2. 如果只从“能不能到 `500MHz`”这个问题看，`64bit -> 16bit` 不是必要条件，因为 `64bit` 行级前端也已经明显满足要求。
3. `16bit` 的主要收益是更快，但它在当前 `VREF/VEVAL` 缩放点下的 `d=2/3` 判决窗口只有 `5.46 ps`，明显小于 `64bit` 的 `17.51 ps`。

这意味着：

- `16bit` 的模拟 HD-CAM 前端虽然更快，但判决裕量更紧，对 `Vref`、`Veval`、sense timing、PVT 更敏感。
- `64bit` 前端更慢，但仍然很快，而且 `d=2/3` 的时间间隔更大，边界更宽松。

本次 SPICE 日志中的 warning 只有模型 include 带来的 `Duplicate scope option scalefactor`，没有再出现之前 `1.2V` 试验中的 `Vgs/Vds` over-stress 告警。

## 5. 28nm DC 设置

DC 使用 Docker 中的 Synopsys DC O-2018.06-SP1 和本机 28nm 标准单元库：

- Docker 镜像：`dc_final_env:v1`
- 工具路径：`/opt/synopsys/dc_2018/syn/O-2018.06-SP1/bin/dc_shell`
- 库：`/pdk/tsmc28/logic/db/tcbn28hpcplusbwp40p140lvtssg0p9v125c_ccs.db`
- 约束：`500MHz`, `2.0ns` 时钟，慢角 `0.9V / 125C`

综合对象不是自定义 CAM bitcell，而是数字近似搜索中最关键的标准单元数据通路：

- `query XOR stored`
- `popcount`
- `dist <= threshold`

对应文件：

- [RTL](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/hardware_8h16b3vote/dc_28nm/rtl/hamming_threshold_compare.v)
- [DC TCL](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/hardware_8h16b3vote/dc_28nm/run_dc.tcl)
- [16b timing](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/hardware_8h16b3vote/dc_28nm/reports/hamming_threshold_compare16.timing.rpt)
- [64b timing](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/hardware_8h16b3vote/dc_28nm/reports/hamming_threshold_compare64.timing.rpt)

## 6. DC 结果

| Design | Library corner | Clock period (ns) | Critical path length (ns) | Slack (ns) | Logic levels | Cell area | Dynamic power |
|---|---|---:|---:|---:|---:|---:|---:|
| `hamming_threshold_compare16` | `ssg0p9v125c` | 2.00 | 0.30 | 1.58 | 6 | 38.43 | 14.56 uW |
| `hamming_threshold_compare64` | `ssg0p9v125c` | 2.00 | 0.41 | 1.47 | 8 | 155.11 | 56.61 uW |

从具体 timing path 看：

- `16bit` 输入到 `match_reg/D` 的 arrival time 为 `0.35ns`
- `64bit` 输入到 `match_reg/D` 的 arrival time 为 `0.46ns`

所以数字路径在 `0.9V / 125C` 慢角下也明显满足 `500MHz`。

### 6.1 DC 结论

1. `64bit` 数字 Hamming compare 在 28nm 慢角下已经可以闭合 `500MHz`，`16bit` 当然也可以。
2. 因此，从数字逻辑角度看，`64bit -> 16bit` 也不是“为了到 500MHz 必须做”的缩放。
3. `16bit` 的主要收益体现在面积和功耗更低；`64bit` 大约是 `16bit` 的 `4.0x` 面积和 `3.9x` 动态功耗。

## 7. 数字 CAM vs 模拟 HD-CAM：同频率下谁更好

### 7.1 如果评价标准是“500MHz 下更容易做对、做稳”

结论：`数字 CAM / 数字 Hamming compare 更好`。

原因：

1. DC 在 `0.9V / 125C` 慢角下，`64bit` 仍有 `1.47ns` slack，时序闭合非常宽松。
2. 数字路径是确定性的 `XOR + popcount + threshold`，不存在 `Vref/Veval` 校准和微小 crossing window 漂移的问题。
3. 相比之下，模拟 HD-CAM 在 `16bit` 缩放后虽然更快，但当前 paper-like 偏置下 `d=2/3` 只有 `5.46ps` 的判决窗口，这对 comparator offset、PVT、jitter 都更敏感。

如果项目目标是：

- 尽快落地
- 在 `500MHz` 下稳定工作
- 最小化 calibration 复杂度

那么优先建议数字实现。

### 7.2 如果评价标准是“前端行级搜索延迟和 in-memory pruning 能力”

结论：`模拟 HD-CAM 更有吸引力`，但前提是你能接受校准复杂度。

原因：

1. 本次 SPICE 的 28nm 行级 HD-CAM 前端估计搜索时间只有：
   - `16bit`: `114 ps`
   - `64bit`: `214 ps`
2. 这比 DC 的数字 compare 数据路径更短，说明模拟前端在阵列内部做近似筛选的潜力很强。
3. 把仓库里的模拟前端默认时间口径从旧的 `3-cycle` proxy 更新为
   `spice_28nm_16b_timing_proxy` 之后，`500MHz` 架构级结果更明显地支持这一点：当候选 survivor 很多、数字 `XOR+popcount` 成为瓶颈时，模拟 HD-CAM 往往能赢。

仓库现有 `500MHz` 行为级结果显示：

- `cora`：模拟约 `4.884 ns`，数字 `per-head verifier` 约 `8.521 ns`
- `pubmed`：模拟约 `4.372 ns`，数字 `per-head verifier` 约 `13.621 ns`
- `arxiv`：模拟约 `4.415 ns`，数字 `per-head verifier` 约 `57.833 ns`

这说明在同频率下，“谁更好”并不是一个单一答案，而是取决于瓶颈在哪里：

- 如果瓶颈在 `verifier`，模拟更有优势。
- 如果 survivor 很少、精确性优先，而且你不想承担模拟校准复杂度，数字仍然更合适。

同时要避免一个常见误读：

- 这里说的“模拟更快”，是系统级总查询延迟更低
- 不是“把数字 verify 去掉以后，模拟前端 search 还比数字前端更快”

按当前 `500 MHz` 的 C++ 周期模型，数字和模拟前端搜索都记作 `1 cycle`。
如果看更细的 SPICE `ps` 级口径，模拟前端其实仍比普通精确 CAM 略慢，
只是差异只有约 `1.03x`，不足以在 `2 ns` 周期预算下拉开整数周期差距。

### 7.3 最终判断

如果必须给一个工程上的默认推荐：

- `默认推荐数字 CAM 路线`

原因不是它更快，而是：

- 它已经在慢角下轻松闭合 `500MHz`
- `64bit` 和 `16bit` 都不需要为时序而降维
- 它对 PVT 和 calibration 更不敏感

模拟 HD-CAM 更适合在下面这种目标下作为优化方向：

- 你明确要把大量候选验证前移到阵列内部
- 你愿意做 `Vref/Veval` 标定
- 你愿意接受 `16bit` 缩放后更紧的判决窗口，并继续做更细的 SA / Monte Carlo / PVT 验证

## 8. 直接回答原问题

### 8.1 `125MHz` 能否通过 `64bit -> 16bit` 和 `28nm` 缩放达到 `500MHz`？

能。

但更准确地说：

- 以当前 `28nm` SPICE 和 DC 结果看，`64bit` 本身就已经足够达到 `500MHz`
- `16bit` 缩放不是为了“勉强够到 500MHz”，而更像是为了进一步减面积/功耗

### 8.2 `64bit -> 16bit` 是不是必要条件？

不是必要条件。

- 模拟前端：`64bit HD-CAM` 估计完整搜索时间 `213.6ps`
- 数字路径：`64bit` 在 DC 慢角下仍有 `1.47ns` slack

所以 `64bit` 已经满足 `500MHz`。

### 8.3 同频率下数字 CAM 和模拟 CAM 谁更好？

如果以工程可实现性、鲁棒性、无需校准为主，`数字 CAM 更好`。  
如果以阵列内近似筛选和系统级减少 verifier 压力为主，`模拟 HD-CAM 更好`，但要付出 calibration 和 PVT 风险管理的代价。

## 9. 当前结论的边界

本报告的结论可信，但还不是最终签核级结论，原因如下：

1. 没有论文 `65nm` PDK，所以无法把“纯工艺缩放倍率”单独拆出来实测。
2. SPICE 还是行级前端 proxy，不是完整 `128-kbit` 宏。
3. DC 验证的是标准单元数字 compare 路径，不是 custom CAM array macro。
4. 模拟 HD-CAM 如果要真走向流片实现，下一步必须补：
   - `SS/TT/FF`
   - `Monte Carlo`
   - `MLSA offset / noise`
   - `Vref/Veval` 扫描下的 `d=2/3` 窗口鲁棒性

## 10. 建议的下一步

1. 如果你准备走 `数字 CAM`，可以直接把 `16bit` 或 `64bit` 的 `XOR + popcount + threshold` 路线继续扩成 bank 级 RTL，并做更完整的 DC/ICC 估计。
2. 如果你准备走 `模拟 HD-CAM`，建议先固定 `16bit` 目标，补 `Vref/Veval` sweep 和 Monte Carlo，重点看 `5.46ps` 这个窗口能不能被稳定保持。
3. 如果你要做论文式对照图，建议把本次 `28nm` 结果整理成：
   - `论文基线：65nm / 64b / 125MHz`
   - `我们的 28nm 端点：64b 和 16b 都有明显高于 500MHz 的裕量`
   - `关键权衡：16b 更快，但模拟判决窗口更小`
