# CAM Simulator Notes

This subproject does not vendor third-party CAM tools in the first version.
It exposes config fields that can be replaced with external CAM cost numbers.

## CAMASim

CAMASim is an open-source CAM accelerator simulator. Its README describes:

- functional simulation for `cam.write()` and `cam.query()`,
- performance evaluation,
- configurable mapping/search schemes,
- optional hardware cost estimation through EvaCAM or user-provided cost data.

Project: <https://github.com/menggg22/CAMASim>

Recommended use here:

1. Keep the current C++ analog CAM model as the functional reference.
2. Use CAMASim to estimate search/write latency and energy for a comparable
   16-bit, 8-bank, threshold-search setting.
3. Copy calibrated values into `analog_cam_cpp/configs/camasim_cost_stub.json`
   or a new config file.

## EvaCAM

EvaCAM is a C++ CAM circuit/architecture evaluation tool. The public README
states support for TCAM, analog CAM, and multi-bit CAM; however, the current
public v1 release notes indicate that only the TCAM exact-match version is
released initially, with ACAM/MCAM updates planned separately.

Project: <https://github.com/eva-cam/EvaCAM>

Recommended use here:

- Treat EvaCAM as a calibration backend rather than as a hard dependency.
- If a calibrated ACAM/threshold-search model is available, use its latency,
  energy, and area outputs to replace the proxy values.

## NVSim-CAM

NVSim-CAM is a circuit-level simulator described in the literature for emerging
non-volatile CAM designs. It is useful as a reference point for CAM energy/area
modeling, but it is not wired into this first repository version.

Reference page:
<https://researchportal.hkust.edu.hk/en/publications/nvsim-cam-a-circuit-level-simulator-for-emerging-nonvolatile-memo/>

## Current Default

The default analog model is marked:

```json
{
  "calibration": "proxy"
}
```

Reports generated with this default should be interpreted as architectural
trend comparisons, not measured circuit results.
