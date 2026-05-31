# Scripts

本目录只放可执行实验脚本和轻量 summary 工具。常用入口按用途分组如下。

## Graph-Bit 主线

```text
run_graphbit_trace_replay.sh
    当前 full-stack 主入口：
    residual/Graph-Bit trace export -> ONNXim component lookup -> trace replay。

run_graphbit_predictor_free_flow.sh
    只跑 predictor-free Graph-Bit accuracy / stop-depth trace。

run_t31_graphbit_nodewise_bound_sweep.sh
    固定 T31 residual/reuse 前端，扫描逐节点 predictor-free bound 参数。
    该入口不使用固定 high/mid/low 节点比例。

replay_graphbit_trace_scheduler.py
    用已有 node trace 和 ONNXim component lookup 重放 scheduler，输出 cycles/traffic/energy 表。

run_onnxim_graphbit_risk_bucket_components.sh
    生成 b32/b64 等 risk-bucket component lookup。
```

## Graph-Bit 调试 / 旧实验

```text
run_llama_precision_depth_sweep.sh
    P8/P6/P5/P4 embedding-pool proxy sweep。

run_onnxim_graphbit_datapath_suite.sh
    datapath component sanity check。

run_graphbit_closure_suite.sh
    demand-fetch / bound / scheduler closure 检查。
```

## Residual Reuse

```text
run_cora_residual_hamming_support_sweep.sh
run_pubmed_residual_hamming_support_sweep.sh
    residual reuse support split / threshold sweep。

summarize_residual_graphbit.py
summarize_residual_graphbit_head_threshold_sweep.py
    residual + Graph-Bit 结果汇总。
```

## ONNXim

```text
build_onnxim.sh
    编译 ONNXim。

onnxim_graphbit_microbench.py
    生成 LLaMA GEMM microbenchmark 并调用 ONNXim。
```

## Git 同步

```text
push_to_github.sh
    日常提交并推送当前分支到远端。
```

用法：

```bash
bash GraphhopSimhash/scripts/push_to_github.sh "docs: update graphbit notes"
```

脚本会执行：

```text
git add -A
git commit -m "<message>"   # 如果有本地改动
git push origin <current-branch>
```

更多复现实验命令见：

```text
docs/npu/GRAPH_BIT_FULLSTACK_REPRODUCTION_GUIDE.md
```
