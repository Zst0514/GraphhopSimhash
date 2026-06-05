# 28nm SPICE Experiment Skeleton

这个目录用于做一个最小可执行的 `28nm` 前端放电速度实验，目标不是完整复现 CAM 宏，而是先回答两个问题：

1. `Exact CAM` 的 `ML` 放电速度大致是多少。
2. `HD-CAM` 在加入 `Meval` 后，`ML` 放电相对普通 CAM 慢多少。

## 已确认可用的 28nm 资源

这台机器上已经存在可读的 `TSMC28` 设计库与模型文件：

- 标准单元综合库：
  - `/opt/synopsys/TSMC28/logic/db`
- 标准单元晶体管级网表：
  - `/opt/synopsys/TSMC28/logic/tcbn28hpcplusbwp40p140lvt_180b/TSMCHOME/digital/Back_End/spice/tcbn28hpcplusbwp40p140lvt_110a/tcbn28hpcplusbwp40p140lvt_110a.spi`
- 标准单元 LPE 网表：
  - `/opt/synopsys/TSMC28/logic/tcbn28hpcplusbwp40p140lvt_180b/TSMCHOME/digital/Back_End/lpe_spice/tcbn28hpcplusbwp40p140lvt_110a/tcbn28hpcplusbwp40p140lvt_110a_lpe_typical.spi`
- Spectre model deck：
  - `/opt/synopsys/TSMC28/iPDK_CRN28HPC+_v1.0_2p2a_20160226_all/CRN28HPCp/models/spectre/toplevel_1d8.scs`

其中 `toplevel_1d8.scs` 会进一步 include：

- `crn28hpcp_lct_1d8_elk_v1d0_2p2_shrink0d9_embedded_usage.scs`
- `cln28hpcp_hv_1d8_elk_v0d1_2p1_shrink0d9_embedded_usage.scs`

并在 `top_tt` section 下提供 `nch_lvt_mac` / `pch_lvt_mac` 等器件定义。

## 重要边界

这些文件说明：

- 这台机器上确实有 `28nm` 的可用模型资源。
- 但它们主要是标准单元和 iPDK 模型，不是现成的 CAM bitcell PDK。
- 因此，`HD-CAM` 自定义 bitcell 需要我们自己写晶体管级网表。

换句话说：

- 做 `XOR/popcount` 这类数字逻辑综合，现有库已经够用。
- 做 `HD-CAM` / `Exact CAM` 的放电速度比较，需要自写最小 bitcell / row netlist。

## 目录内容

- [exact_cam_frontend_tt.scs](./exact_cam_frontend_tt.scs)
  - 普通 CAM 的最小放电路径实验模板
- [hdcam_frontend_tt.scs](./hdcam_frontend_tt.scs)
  - 带 `Meval` 的 HD-CAM 最小放电路径实验模板

这两份模板都只抓最核心的部分：

- `ML` 预充
- mismatch 导致放电
- `Exact CAM`：直接放电
- `HD-CAM`：通过 `Meval` 限流放电

它们不是完整 CAM word，也没有真正的 `SL/XNOR/MLSA` 外围，只是为了先量化：

```text
ML crossing time to Vref
```

## 建议实验步骤

1. 先跑 `Exact CAM`
   - 固定 `mismatch_bits = 1`
   - 记录 `ML` 从 `VDD` 掉到 `Vref` 的 crossing time

2. 再跑 `HD-CAM`
   - 固定 `mismatch_bits = 3`
   - 扫 `VEVAL`
   - 记录 crossing time

3. 取：

```text
HD-CAM crossing time / Exact-CAM crossing time
```

作为前端纯放电判决窗口的速度比

4. 再把 precharge / sensing 的公共开销加回去，得到完整搜索路径速度比

## 如果你要继续严肃做下去

下一步应该补三样东西：

1. `16-bit` row 级 `ML` 电容提取或等效估计
2. `SL` 驱动和 `MLSA` 的最小 testbench
3. `SS/TT/FF + Monte Carlo` sweep

只有做到这一步，才能把现在的架构级 proxy 进一步压成更可信的 transistor-level 结论。
