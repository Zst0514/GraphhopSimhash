#!/usr/bin/env python3
"""使用 Spectre 运行一个小型 28nm CAM 前端缩放实验。

这个脚本刻意保持简洁：

1. 直接复用当前机器上已经可用的 28nm 晶体管模型。
2. 使用行级 match-line 放电代理模型，而不是完整 CAM 宏。
3. 复用本地 16-bit proxy 的设定，并按字长线性缩放 match-line 电容。
4. 实测：
   - 预充到 90% VDD 的时间
   - 普通 CAM 在 d=1 时 crossing 到 Vref 的时间
   - HD-CAM 在 d=2 和 d=3 时 crossing 到 Vref 的时间

输出为一个 JSON 汇总文件和一个 Markdown 表格，便于在报告中直接引用。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


MODEL_DECK = (
    "/opt/synopsys/TSMC28/"
    "iPDK_CRN28HPC+_v1.0_2p2a_20160226_all/"
    "CRN28HPCp/models/spectre/toplevel_1d8.scs"
)
ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "scaling_study_runs"
SUMMARY_JSON = RUN_DIR / "summary.json"
SUMMARY_MD = RUN_DIR / "summary.md"


@dataclass(frozen=True)
class StudyConfig:
    vdd: float = 0.9
    vref: float = 0.6
    veval: float = 0.6
    sense_time_ps: float = 20.0
    l_core_nm: float = 40.0
    w_pre_nm: float = 170.0
    w_dis_nm: float = 140.0
    w_meval_nm: float = 140.0
    w_mismatch_nm: float = 140.0
    matchline_base_cap_ff: float = 0.6
    matchline_cap_per_bit_ff: float = 0.2
    precharge_target_ratio: float = 0.9
    eval_start_ps: float = 50.0
    rise_fall_ps: float = 2.0
    tstop_ps: float = 4000.0
    maxstep_ps: float = 0.5
    word_bits: Tuple[int, ...] = (16, 64)

    def ml_cap_ff(self, word_bits: int) -> float:
        return self.matchline_base_cap_ff + self.matchline_cap_per_bit_ff * word_bits


@dataclass
class CrossingResult:
    case: str
    word_bits: int
    ml_cap_ff: float
    crossing_ps: float | None
    waveform_min_v: float
    waveform_max_v: float
    warnings: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spectre",
        default="spectre",
        help="要使用的 Spectre 可执行文件，默认从 PATH 中查找 `spectre`。",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="不重新仿真，只重新解析 scaling_study_runs 下已有的 raw/log 文件。",
    )
    return parser.parse_args()


def ensure_model_deck() -> None:
    if not Path(MODEL_DECK).exists():
        raise FileNotFoundError(f"缺少 Spectre 模型甲板：{MODEL_DECK}")


def float_to_spectre(value: float, unit: str = "") -> str:
    return f"{value:g}{unit}"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_common_header(cfg: StudyConfig) -> str:
    return f"""simulator lang=spectre
global 0

include "{MODEL_DECK}" section=top_tt

parameters VDD={cfg.vdd:g}
parameters VREF={cfg.vref:g}
parameters VEVAL={cfg.veval:g}
parameters TSTOP={float_to_spectre(cfg.tstop_ps, "p")}
parameters MAXSTEP={float_to_spectre(cfg.maxstep_ps, "p")}
parameters TSTART={float_to_spectre(cfg.eval_start_ps, "p")}
parameters TEDGE={float_to_spectre(cfg.rise_fall_ps, "p")}
parameters LCORE={float_to_spectre(cfg.l_core_nm, "n")}
"""


def build_precharge_netlist(cfg: StudyConfig, word_bits: int) -> str:
    ml_cap_ff = cfg.ml_cap_ff(word_bits)
    return (
        build_common_header(cfg)
        + f"""
parameters ML_CAP={float_to_spectre(ml_cap_ff, "f")}
parameters WPRE={float_to_spectre(cfg.w_pre_nm, "n")}

VDD_SRC (vdd 0) vsource dc=VDD
VPC (pc 0) vsource dc=0

CML (ml 0) capacitor c=ML_CAP ic=0
MPRE (ml pc vdd vdd) pch_lvt_mac w=WPRE l=LCORE

tran tran stop=TSTOP maxstep=MAXSTEP
save ml pc
"""
    )


def build_exact_netlist(cfg: StudyConfig, word_bits: int, mismatch_on: bool) -> str:
    ml_cap_ff = cfg.ml_cap_ff(word_bits)
    mis_val1 = "VDD" if mismatch_on else "0"
    return (
        build_common_header(cfg)
        + f"""
parameters ML_CAP={float_to_spectre(ml_cap_ff, "f")}
parameters WPRE={float_to_spectre(cfg.w_pre_nm, "n")}
parameters WDIS={float_to_spectre(cfg.w_dis_nm, "n")}

VDD_SRC (vdd 0) vsource dc=VDD
VPC (pc 0) vsource type=pulse val0=0 val1=VDD delay=TSTART rise=TEDGE fall=TEDGE width=3.5n period=8n
VMIS (mis 0) vsource type=pulse val0=0 val1={mis_val1} delay=TSTART rise=TEDGE fall=TEDGE width=3.5n period=8n

CML (ml 0) capacitor c=ML_CAP ic=VDD
MPRE (ml pc vdd vdd) pch_lvt_mac w=WPRE l=LCORE
MDIS (ml mis 0 0) nch_lvt_mac w=WDIS l=LCORE

tran tran stop=TSTOP maxstep=MAXSTEP
save ml mis pc
"""
    )


def build_hdcam_netlist(cfg: StudyConfig, word_bits: int, mismatch_count: int) -> str:
    if mismatch_count not in (2, 3):
        raise ValueError(f"unsupported mismatch_count={mismatch_count}, expected 2 or 3")
    ml_cap_ff = cfg.ml_cap_ff(word_bits)
    branch_controls = []
    for idx in range(1, 4):
        source_kind = "VDD" if idx <= mismatch_count else "0"
        branch_controls.append(
            f"VMIS{idx} (mis{idx} 0) vsource type=pulse val0=0 val1={source_kind} "
            "delay=TSTART rise=TEDGE fall=TEDGE width=3.5n period=8n"
        )
    branches = []
    for idx in range(1, 4):
        branches.append(
            f"MME{idx} (ml veval n{idx} 0) nch_lvt_mac w=WMEVAL l=LCORE\n"
            f"MMI{idx} (n{idx} mis{idx} 0 0) nch_lvt_mac w=WMISMATCH l=LCORE"
        )
    return (
        build_common_header(cfg)
        + f"""
parameters ML_CAP={float_to_spectre(ml_cap_ff, "f")}
parameters WPRE={float_to_spectre(cfg.w_pre_nm, "n")}
parameters WMEVAL={float_to_spectre(cfg.w_meval_nm, "n")}
parameters WMISMATCH={float_to_spectre(cfg.w_mismatch_nm, "n")}

VDD_SRC (vdd 0) vsource dc=VDD
VEEVAL (veval 0) vsource dc=VEVAL
VPC (pc 0) vsource type=pulse val0=0 val1=VDD delay=TSTART rise=TEDGE fall=TEDGE width=3.5n period=8n
{chr(10).join(branch_controls)}

CML (ml 0) capacitor c=ML_CAP ic=VDD
MPRE (ml pc vdd vdd) pch_lvt_mac w=WPRE l=LCORE

{chr(10).join(branches)}

tran tran stop=TSTOP maxstep=MAXSTEP
save ml veval mis1 mis2 mis3 pc
"""
    )


def run_spectre(spectre: str, netlist_path: Path, raw_path: Path, log_path: Path) -> None:
    cmd = [
        spectre,
        "-64",
        str(netlist_path),
        "-format",
        "nutascii",
        "-raw",
        str(raw_path),
        "=log",
        str(log_path),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_nutascii(raw_path: Path) -> Dict[str, List[float]]:
    text = raw_path.read_text(encoding="utf-8")
    num_vars_match = re.search(r"No\. Variables:\s+(\d+)", text)
    if not num_vars_match:
        raise ValueError(f"failed to parse variable count from {raw_path}")
    num_vars = int(num_vars_match.group(1))

    lines = text.splitlines()
    try:
        variables_idx = next(idx for idx, line in enumerate(lines) if line.startswith("Variables:"))
        values_idx = next(idx for idx, line in enumerate(lines) if line.startswith("Values:"))
    except StopIteration as exc:
        raise ValueError(f"failed to locate Variables/Values sections in {raw_path}") from exc

    var_lines: List[str] = []
    first_var_line = lines[variables_idx][len("Variables:") :].strip()
    if first_var_line:
        var_lines.append(first_var_line)
    for line in lines[variables_idx + 1 : values_idx]:
        stripped = line.strip()
        if stripped:
            var_lines.append(stripped)

    var_names = []
    for line in var_lines:
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"unexpected variable line in {raw_path}: {line}")
        var_names.append(parts[1])

    values_text = "\n".join(lines[values_idx + 1 :])
    tokens = values_text.split()
    stride = num_vars + 1
    if len(tokens) % stride != 0:
        raise ValueError(
            f"token count {len(tokens)} not divisible by stride {stride} in {raw_path}"
        )

    data = {name: [] for name in var_names}
    for idx in range(0, len(tokens), stride):
        _point_index = int(tokens[idx])
        values = [float(token) for token in tokens[idx + 1 : idx + 1 + num_vars]]
        for name, value in zip(var_names, values):
            data[name].append(value)
    return data


def find_first_crossing_ps(
    times_s: Sequence[float],
    voltages_v: Sequence[float],
    threshold_v: float,
    start_ps: float,
    direction: str,
) -> float | None:
    start_s = start_ps * 1.0e-12
    if direction not in ("fall", "rise"):
        raise ValueError(f"unsupported direction={direction}")

    for idx in range(1, len(times_s)):
        t0 = times_s[idx - 1]
        t1 = times_s[idx]
        if t1 < start_s:
            continue
        v0 = voltages_v[idx - 1]
        v1 = voltages_v[idx]
        crossed = (
            v0 > threshold_v >= v1 if direction == "fall" else v0 < threshold_v <= v1
        )
        if not crossed:
            continue
        if math.isclose(v1, v0):
            return t1 * 1.0e12
        frac = (threshold_v - v0) / (v1 - v0)
        t_cross = t0 + frac * (t1 - t0)
        return t_cross * 1.0e12
    return None


def count_warnings(log_path: Path) -> int:
    warning_re = re.compile(r"\bWARNING\b", re.I)
    count = 0
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if warning_re.search(line):
            count += 1
    return count


def collect_result(
    case: str,
    word_bits: int,
    ml_cap_ff: float,
    raw_path: Path,
    log_path: Path,
    threshold_v: float,
    start_ps: float,
    direction: str,
) -> CrossingResult:
    data = parse_nutascii(raw_path)
    times = data["time"]
    ml = data["ml"]
    return CrossingResult(
        case=case,
        word_bits=word_bits,
        ml_cap_ff=ml_cap_ff,
        crossing_ps=find_first_crossing_ps(times, ml, threshold_v, start_ps, direction),
        waveform_min_v=min(ml),
        waveform_max_v=max(ml),
        warnings=count_warnings(log_path),
    )


def result_key(case: str, word_bits: int) -> str:
    return f"{case}_{word_bits}b"


def format_ps(value: float | None) -> str:
    if value is None:
        return "无"
    return f"{value:.3f}"


def format_ratio(value: float | None) -> str:
    if value is None:
        return "无"
    return f"{value:.3f}x"


def format_ghz(value: float | None) -> str:
    if value is None:
        return "无"
    return f"{value:.3f}"


def build_markdown_summary(cfg: StudyConfig, results: Dict[str, CrossingResult]) -> str:
    lines = [
        "# 28nm CAM 前端缩放实验",
        "",
        "- 模型甲板：`TSMC28 CRN28HPC+ top_tt`",
        "- 供电/偏置点："
        f"`VDD={cfg.vdd:.2f} V, VREF={cfg.vref:.2f} V, VEVAL={cfg.veval:.2f} V`",
        "- 偏置缩放说明：`0.9 V / 0.6 V` 是从论文 `1.2 V / 0.8 V` 工作点归一化得到的 28nm 偏置代理。",
        "- 完整搜索时间估算中的感测时间假设："
        f"`{cfg.sense_time_ps:.1f} ps`",
        "",
        "## 实测 crossing time",
        "",
        "| 字长 (bit) | ML 电容 (fF) | 预充到 0.9VDD (ps) | 普通 CAM d=1 到 Vref (ps) | HD-CAM d=3 到 Vref (ps) | HD-CAM d=2 到 Vref (ps) | HD d2-d3 窗口 (ps) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for word_bits in cfg.word_bits:
        pre = results[result_key("precharge90", word_bits)]
        exact = results[result_key("exact_d1", word_bits)]
        hd2 = results[result_key("hdcam_d2", word_bits)]
        hd3 = results[result_key("hdcam_d3", word_bits)]
        window_ps = (
            None
            if hd2.crossing_ps is None or hd3.crossing_ps is None
            else hd2.crossing_ps - hd3.crossing_ps
        )
        lines.append(
            "| "
            f"{word_bits} | {pre.ml_cap_ff:.3f} | {format_ps(pre.crossing_ps)} | "
            f"{format_ps(exact.crossing_ps)} | {format_ps(hd3.crossing_ps)} | "
            f"{format_ps(hd2.crossing_ps)} | {format_ps(window_ps)} |"
        )

    lines.extend(
        [
            "",
            "## 完整搜索时间估算",
            "",
            "下面的估算采用：",
            "",
            "- `search_time_exact = t_precharge90 + t_exact_d1 + t_sense`",
            "- `search_time_hdcam = t_precharge90 + t_hdcam_d3 + t_sense`",
            "- `500 MHz 预算 = 2000 ps`",
            "",
            "| 字长 (bit) | 普通 CAM 完整搜索 (ps) | 普通 CAM 推导 Fmax (GHz) | HD-CAM 完整搜索 (ps) | HD-CAM 推导 Fmax (GHz) | 普通 CAM 满足 500MHz? | HD-CAM 满足 500MHz? |",
            "|---:|---:|---:|---:|---:|---|---|",
        ]
    )

    for word_bits in cfg.word_bits:
        pre = results[result_key("precharge90", word_bits)]
        exact = results[result_key("exact_d1", word_bits)]
        hd3 = results[result_key("hdcam_d3", word_bits)]

        exact_full_ps = None
        if pre.crossing_ps is not None and exact.crossing_ps is not None:
            exact_full_ps = pre.crossing_ps + exact.crossing_ps + cfg.sense_time_ps
        hd_full_ps = None
        if pre.crossing_ps is not None and hd3.crossing_ps is not None:
            hd_full_ps = pre.crossing_ps + hd3.crossing_ps + cfg.sense_time_ps

        exact_fmax = None if exact_full_ps is None else 1000.0 / exact_full_ps
        hd_fmax = None if hd_full_ps is None else 1000.0 / hd_full_ps

        lines.append(
            "| "
            f"{word_bits} | {format_ps(exact_full_ps)} | {format_ghz(exact_fmax)} | "
            f"{format_ps(hd_full_ps)} | {format_ghz(hd_fmax)} | "
            f"{'是' if exact_full_ps is not None and exact_full_ps <= 2000.0 else '否'} | "
            f"{'是' if hd_full_ps is not None and hd_full_ps <= 2000.0 else '否'} |"
        )

    lines.extend(
        [
            "",
            "## 告警统计",
            "",
            "| Case | 字长 (bit) | Spectre 告警数 |",
            "|---|---:|---:|",
        ]
    )

    for key in sorted(results):
        result = results[key]
        lines.append(f"| {result.case} | {result.word_bits} | {result.warnings} |")

    return "\n".join(lines) + "\n"


def format_fmax(value_ps: float | None) -> float | None:
    if value_ps is None or value_ps <= 0.0:
        return None
    return 1000.0 / value_ps


def run_or_parse_case(
    *,
    spectre: str,
    skip_run: bool,
    case: str,
    word_bits: int,
    netlist: str,
    threshold_v: float,
    start_ps: float,
    direction: str,
    ml_cap_ff: float,
) -> CrossingResult:
    netlist_path = RUN_DIR / f"{case}_{word_bits}b.scs"
    raw_path = RUN_DIR / f"{case}_{word_bits}b.raw"
    log_path = RUN_DIR / f"{case}_{word_bits}b.log"
    write_text(netlist_path, netlist)
    if not skip_run:
        run_spectre(spectre, netlist_path, raw_path, log_path)
    return collect_result(
        case=case,
        word_bits=word_bits,
        ml_cap_ff=ml_cap_ff,
        raw_path=raw_path,
        log_path=log_path,
        threshold_v=threshold_v,
        start_ps=start_ps,
        direction=direction,
    )


def result_to_json_dict(result: CrossingResult) -> Dict[str, float | int | str | None]:
    return asdict(result)


def main() -> None:
    args = parse_args()
    ensure_model_deck()
    cfg = StudyConfig()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    results: Dict[str, CrossingResult] = {}
    for word_bits in cfg.word_bits:
        ml_cap_ff = cfg.ml_cap_ff(word_bits)
        results[result_key("precharge90", word_bits)] = run_or_parse_case(
            spectre=args.spectre,
            skip_run=args.skip_run,
            case="precharge90",
            word_bits=word_bits,
            netlist=build_precharge_netlist(cfg, word_bits),
            threshold_v=cfg.precharge_target_ratio * cfg.vdd,
            start_ps=0.0,
            direction="rise",
            ml_cap_ff=ml_cap_ff,
        )
        results[result_key("exact_d1", word_bits)] = run_or_parse_case(
            spectre=args.spectre,
            skip_run=args.skip_run,
            case="exact_d1",
            word_bits=word_bits,
            netlist=build_exact_netlist(cfg, word_bits, mismatch_on=True),
            threshold_v=cfg.vref,
            start_ps=cfg.eval_start_ps,
            direction="fall",
            ml_cap_ff=ml_cap_ff,
        )
        results[result_key("hdcam_d2", word_bits)] = run_or_parse_case(
            spectre=args.spectre,
            skip_run=args.skip_run,
            case="hdcam_d2",
            word_bits=word_bits,
            netlist=build_hdcam_netlist(cfg, word_bits, mismatch_count=2),
            threshold_v=cfg.vref,
            start_ps=cfg.eval_start_ps,
            direction="fall",
            ml_cap_ff=ml_cap_ff,
        )
        results[result_key("hdcam_d3", word_bits)] = run_or_parse_case(
            spectre=args.spectre,
            skip_run=args.skip_run,
            case="hdcam_d3",
            word_bits=word_bits,
            netlist=build_hdcam_netlist(cfg, word_bits, mismatch_count=3),
            threshold_v=cfg.vref,
            start_ps=cfg.eval_start_ps,
            direction="fall",
            ml_cap_ff=ml_cap_ff,
        )

    summary = {
        "config": asdict(cfg),
        "results": {
            key: result_to_json_dict(value) for key, value in sorted(results.items())
        },
    }
    write_text(SUMMARY_JSON, json.dumps(summary, indent=2, sort_keys=True))
    write_text(SUMMARY_MD, build_markdown_summary(cfg, results))

    print(f"[scaling-study] 已写出 {SUMMARY_JSON}")
    print(f"[scaling-study] 已写出 {SUMMARY_MD}")
    for word_bits in cfg.word_bits:
        pre = results[result_key("precharge90", word_bits)].crossing_ps
        exact = results[result_key("exact_d1", word_bits)].crossing_ps
        hd3 = results[result_key("hdcam_d3", word_bits)].crossing_ps
        exact_full = None if pre is None or exact is None else pre + exact + cfg.sense_time_ps
        hd_full = None if pre is None or hd3 is None else pre + hd3 + cfg.sense_time_ps
        print(
            "[scaling-study] "
            f"{word_bits}b exact_full={format_ps(exact_full)}ps "
            f"(fmax={format_ghz(format_fmax(exact_full))}GHz) "
            f"hd_full={format_ps(hd_full)}ps "
            f"(fmax={format_ghz(format_fmax(hd_full))}GHz)"
        )


if __name__ == "__main__":
    main()
