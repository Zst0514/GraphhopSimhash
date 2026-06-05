# 用模拟 CAM 实现 8 个 16bit 哈希向量匹配的过程

## 1. 这份文档讲什么

这份文档只讲一件事：

```text
我们如何用模拟 CAM（analog CAM）的思想，
对 8 个 16bit 哈希头做并行近似匹配，
并判断一个旧节点能不能被复用。
```

这里的目标不是“找完全一模一样的哈希”，而是：

```text
找出汉明距离足够小的候选，
默认希望逼近“HD <= 2”。
```

文档对应当前实现：

- [analog_cam_engine.cpp](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash-main/CAM_sim/analog_cam_cpp/analog_cam_engine.cpp)
- [analog_cam_default.json](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash-main/CAM_sim/analog_cam_cpp/configs/analog_cam_default.json)
- [analog_cam_legacy_proxy.json](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash-main/CAM_sim/analog_cam_cpp/configs/analog_cam_legacy_proxy.json)
- [8x16bit_latency_comparison_zh.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash-main/CAM_sim/docs/8x16bit_latency_comparison_zh.md)

---

## 2. 核心思路

普通 CAM 只擅长做：

```text
查询值 == 存储值 ?
```

而我们要做的是：

```text
查询值 和 存储值 是否“足够接近” ?
```

这里的“接近”用汉明距离表示：

```text
两串 16bit 哈希有几位不同
```

模拟 CAM 的做法不是先把 16 位全比完、再数字地数错几位，而是利用一个物理现象：

```text
不匹配位越多，match line 放电越快；
不匹配位越少，match line 放电越慢。
```

所以可以把：

```text
汉明距离大小
```

变成：

```text
某个固定时刻下，match line 还剩多少电压
```

然后再用一个比较器阈值把“足够近”和“不够近”分开。

---

## 3. 我们的对象：8 个 16bit 哈希头

当前复用前端不是一个长哈希，而是：

```text
8 个独立 head
每个 head 16bit
```

可以把它理解成：

```text
同一个节点，会从 8 个不同视角得到 8 个短哈希。
```

硬件上对应为：

```text
head0 -> 一个 16bit CAM bank
head1 -> 一个 16bit CAM bank
...
head7 -> 一个 16bit CAM bank
```

这 8 个 bank 在一次查询时并行工作。

---

## 4. 单个 16bit head 的匹配过程

下面只看一个 head。其他 7 个 head 完全一样，只是并行执行。

### 第 1 步：输入查询哈希

比如当前查询节点在某个 head 上的哈希是：

```text
q = 1011001110001111
```

这个 `q` 会同时广播到该 head 的所有活动行。

### 第 2 步：预充 match line

每一行都有一根 `match line`，记作 `ML`。

查询前，先把每条 `ML` 充到高电平：

```text
ML = VDD
```

可以理解成：

```text
每一行先都充满电
```

### 第 3 步：逐位比较，形成放电路径

当前行里存着一个 16bit 哈希。

查询值和这一行逐位比较：

- 这一位相同：只产生很小的漏电
- 这一位不同：打开更强的放电通路

于是：

```text
不匹配位越多，整条 ML 的总放电导通越大
```

### 第 4 步：等待固定评估时间

不是一直等到电全部放完，而是只等一个固定评估时间 `t_eval`。

当前模型使用：

```text
precharge_time_ps = 30.78 ps
eval_time_ps = 64 ps
sense_time_ps = 20 ps
```

也就是一次查询的模拟搜索时间主要由：

```text
预充 + 放电评估 + 感测
```

组成。

### 第 5 步：根据 RC 模型计算 ML 电压

当前代码使用的核心公式是：

```text
V_ML = VDD * exp(-G_total * t_eval / C_ML)
```

其中：

- `VDD`：供电电压
- `G_total`：总放电导通，和不匹配位数有关
- `t_eval`：评估时间
- `C_ML`：match line 总电容

直观上看：

- `HD = 0`：放电最慢，`V_ML` 最高
- `HD = 1`：稍低
- `HD = 2`：更低
- `HD = 3`：再低
- `HD` 越大：`V_ML` 越低

### 第 6 步：比较器判断是否命中

接下来用参考阈值 `Vref` 做判决：

```text
如果 V_ML >= Vref，则这一行命中
如果 V_ML <  Vref，则这一行不命中
```

当前默认配置里：

```text
comparator_vref = -1.0
```

这表示不手工指定阈值，而是自动把 `Vref` 放在：

```text
nominal d=2 电压
和
nominal d=3 电压
之间
```

所以默认零噪声情况下，它近似实现的是：

```text
HD <= 2 -> 命中
HD >= 3 -> 不命中
```

---

## 5. 8 个 head 如何并行工作

上面说的是一个 head。完整查询时，8 个 head 同时做这件事：

```text
query node
  -> 8 个 16bit head hash
  -> 8 个 CAM bank 并行搜索
  -> 每个 bank 返回一批“足够近”的行
```

所以模拟 CAM 前端做的事情，本质上是：

```text
在 8 个独立 16bit 空间里，
同时查找哪些旧 entry 和当前 query 足够接近。
```

---

## 6. 命中行如何变成“可复用候选”

每个 head 命中的，不是最终答案，而是一批候选行。

每一行至少对应：

- `node_id`
- `timestamp`
- `该 head 上的最小距离`

然后我们在数字后端按 `node_id` 聚合：

```text
同一个 node_id 如果在多个 head 都命中，
就累计 support_count。
```

例如：

```text
head0 命中 node 100
head2 命中 node 100
head5 命中 node 100
```

聚合后得到：

```text
node 100 的 support = 3
```

---

## 7. 我们当前的复用决策规则

当前模型不是“只要有一个 head 命中就复用”，而是：

```text
support_count >= 3 才复用
否则就重算
```

也就是常说的：

```text
8 heads
3-vote
```

在多个候选都满足 `support >= 3` 时，当前 tie-break 顺序是：

```text
1. support 更高
2. min_dist 更小
3. timestamp 更新
4. node_id 更小
```

所以整个链条是：

```text
8 个 head 并行近似匹配
-> 聚合同一个 node_id 的命中票数
-> support >= 3 则接受复用
-> 否则走 compute
```

---

## 8. miss 之后怎么更新 CAM

如果当前节点没有找到可复用候选，就会重算该节点，然后把新节点写回 8 个 head 的存储。

当前实现里：

```text
每个 head 都会插入一条新记录
```

同时保留一个软件级的 bucket 结构来模拟有限保留：

```text
memo_k = 3
```

意思是：

```text
同一个精确 hash bucket 最多保留 3 个 entry
超出的旧 entry 会被标记为 inactive
```

这一步主要是为了让 cache 行为和现有 reuse 控制器保持一致。

---

## 9. 当前 RC 模型里最关键的物理参数

当前默认配置如下：

```text
clock_mhz = 500
vdd = 0.9
veval = 0.6
fixed_vref = 0.6
matchline_base_cap_f = 6.0e-16
matchline_cap_per_bit_f = 2.0e-16
mismatch_conductance_s = 1.5862e-5
exact_mismatch_conductance_s = 2.2451e-5
match_leak_conductance_s = 2.0e-7
precharge_time_ps = 30.7809
eval_time_ps = 64
sense_time_ps = 20
cam_search_cycles = 1
comparator_vref = -1.0
device_sigma_rel = 0.0
sense_noise_sigma_v = 0.0
comparator_noise_sigma_v = 0.0
```

旧的保守 `3-cycle` 版本仍保留在：

```text
analog_cam_cpp/configs/analog_cam_legacy_proxy.json
```

这些参数分别决定：

- `C_ML` 有多大
- mismatch 放电有多强
- match 漏电有多大
- 等多久再采样
- 比较器门槛设在哪里
- 是否考虑器件波动和感测噪声

当前默认值是：

```text
零噪声、零随机波动
```

所以它更像：

```text
一个物理启发的、可解释的 RC 行为模型
```

还不是 SPICE 级签核模型。

---

## 10. 它和“普通 CAM 精确匹配”的差别

普通 CAM：

```text
只有完全相同才命中
```

我们现在这个模拟 CAM：

```text
允许少量 bit 不同，只要放电后仍高于阈值，就算命中
```

所以它不是在回答：

```text
有没有完全一样的哈希？
```

而是在回答：

```text
有没有一个旧哈希，和当前这个哈希足够接近，
可以作为复用候选？
```

这正是 reuse 前端真正需要的能力。

---

## 11. 为什么 16bit x 8-head 适合这样做

这套方法对我们这个结构是比较合适的，原因有三点：

### 1. 单个 head 很短

每个 head 只有 `16bit`，所以：

- 汉明距离等级少
- `d=2` 和 `d=3` 的电压边界更容易拉开
- 比 64bit 或 128bit 更容易做阈值判决

### 2. 多头能给出置信度

我们不是只看某一个 head 是否命中，而是看：

```text
同一个 node_id 被多少个 head 同时支持
```

这让硬件前端输出的不只是“hit/miss”，还有：

```text
support 票数
```

这对后续 reuse 决策很关键。

### 3. 前端和后端职责清晰

模拟 CAM 只做：

```text
快速近似找候选
```

数字逻辑再做：

```text
聚合、投票、选最优候选、决定是否复用
```

这个分工比较清楚，也更容易扩展。

---

## 12. 这份模型现在能回答什么，不能回答什么

### 它现在能回答

- 用 RC/放电阈值思想做 `HD<=2` 类搜索，流程是否合理
- 8 个 16bit head 并行近似匹配时，系统级 reuse/cycle/energy proxy 大概怎样
- `support >= 3` 的 3-vote 复用前端是否可工作

### 它现在还不能直接回答

- 真实电路在 PVT 条件下的最终误码率
- 真实版图面积
- 真实芯片频率上限
- comparator offset、版图 parasitic、跨阵列布线后的最终签核结果

所以当前模型的定位应该是：

```text
比纯行为级更物理，
但还不是电路签核级。
```

---

## 13. 一句话总结

一句话概括我们现在这套模拟 CAM 匹配流程：

```text
把每个 16bit 哈希 head 的“差几位”，
变成 match line 在固定时刻“还剩多少电压”，
再用阈值把近似命中筛出来，
最后对 8 个 head 的命中结果做 3-vote 聚合，决定是否复用。
```

这就是我们当前用模拟 CAM 实现 `8 x 16bit` 哈希匹配的完整过程。
