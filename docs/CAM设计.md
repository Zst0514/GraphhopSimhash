# CAM 设计

## 1. 目标

本文说明我们在 GraphhopSimhash 中如何把普通 CAM 改造成 HD-CAM，并用于：

```text
8 个 16bit 哈希头
汉明距离阈值匹配
多头 support 聚合
direct / residual / compute 三段式复用
```

我们的目标不是只做精确检索，而是找“足够接近”的复用候选。

---

## 2. 普通 CAM 做什么

普通 CAM 的能力很直接：

```text
把查询值同时和所有存储行比较，
找出完全相同的条目。
```

对应架构如下：

![普通 CAM 架构](figure/CAM.png)

图里可以看成三部分：

- `CAM Array`：每行一个存储 word，每列一个 bit 位
- `Search Data Registers / Drivers`：把查询值广播到所有 bitcell
- `ML sense amplifiers`：最终输出 `hit / miss`

普通 CAM 的本质是：

```text
逐行并行，逐位精确匹配
```

它适合 exact match，不适合直接回答“差 1 位、差 2 位是否也算命中”。

---

## 3. 为什么要上 HD-CAM

在哈希复用里，真正有价值的候选往往不是完全相同，而是：

```text
汉明距离很小
```

我们现在是：

```text
8 个 head
每个 head 16bit
```

如果继续用普通 CAM，只能做“16bit 完全一样才命中”，太严格了。  
所以需要把 CAM 改成：

```text
找出所有 HD <= T 的行
```

这就是 HD-CAM 的作用。

---

## 4. HD-CAM cell

HD-CAM 的关键不是把 CAM 改成更复杂的存储单元，而是让它的 match line 能反映“差了几位”。

参考的 HD-CAM cell 如下：

![HD-CAM bitcell](figure/HD-CAM_cell.png)

这里最关键的是新增的：

- `Meval`
- `Veval`

普通 CAM 只判断“有没有不匹配”；  
HD-CAM 则让“不匹配位的数量”影响 match line 的放电速度。

直觉上就是：

- 不匹配位少，ML 放电慢
- 不匹配位多，ML 放电快

因此，match line 不再只是二值信号，而是一个近似距离的物理载体。

---

## 5. 物理判决逻辑

HD-CAM 的基本判决可以抽象成：

```text
V_ML = VDD * exp(-G_total * t_eval / C_ML)
```

其中：

- `VDD`：供电电压
- `G_total`：总放电导通
- `t_eval`：评估时间
- `C_ML`：match line 电容

核心关系很简单：

```text
汉明距离越大 -> 放电越快 -> V_ML 越低
汉明距离越小 -> 放电越慢 -> V_ML 越高
```

然后用参考阈值 `Vref` 做比较：

```text
V_ML >= Vref -> hit
V_ML <  Vref -> miss
```

如果把 `Vref` 放在 `d = 2` 和 `d = 3` 之间，就能近似实现：

```text
HD <= 2 -> 命中
HD >= 3 -> 不命中
```

这比普通 CAM 的 exact match 更适合哈希复用。

---

## 6. 8 个 16bit head 的组织方式

我们不把一个长哈希直接丢进 CAM，而是采用：

```text
8 个独立 head
每个 head 16bit
```

对应到硬件上，就是 8 个并行 bank：

```text
Head0 -> HD-CAM bank 0
Head1 -> HD-CAM bank 1
...
Head7 -> HD-CAM bank 7
```

这样做有三个好处：

1. 单个 bank 字长短，阈值更容易调
2. 8 个 head 可以天然提供 support 置信度
3. 前端候选筛选和后端复用决策可以解耦

---

## 7. 我们的完整哈希复用架构

我们把上面的思想组合成一张完整架构图：

![HD-CAM 哈希复用架构图](figure/HD-CAM_hash_reuse_architecture.png)

这张图表达的是整条链路：

1. 输入节点进入前端
2. 生成 8 个 16bit 哈希头
3. 8 个 HD-CAM bank 并行做阈值搜索
4. 候选按 `node_id` 聚合，统计 `support count`
5. 进入三段式决策：
   - `support 高` -> 直接复用
   - `support 中` -> 残差复用
   - `support 低` -> 重新计算
6. 结果写回缓存

这张图对应的不是单纯的 CAM 检索，而是一个完整的哈希复用前端。

---

## 8. 三段式决策

HD-CAM 前端返回的不是最终 embedding，而是候选和 support。

因此后端按 support 做三段式决策：

```text
support 高  -> hard direct reuse
support 中  -> residual reuse
support 低  -> compute
```

在当前 Cora/PubMed 共同参数探索后，主线固定切法是：

```text
R = 2
8 heads x 16 bits
score threshold T = 40
hard_direct >= 5
residual_soft = 4
compute < 4
```

含义很直接：

- `5~8` 个 head 同时支持：直接复用，风险低
- `4` 个 head 同时支持：进入 residual correction
- `0~3` 个 head 同时支持：置信度不够，重算

这组固定参数在 3-run 结果里得到：

```text
Cora:   reuse = 25.7%, drop = 0.45%
PubMed: reuse = 50.3%, drop = 2.52%
```

`hard_direct >= 6, residual_soft = 4` 的总 reuse 相同，但 PubMed drop 略高，说明 support=5 的节点已经足够可靠，直接复用比强制 residual correction 更稳。

所以，HD-CAM 只负责“找近似候选”，最终能不能复用，要看多头 support 聚合后的结果。

---

## 9. 这种设计为什么适合 GraphhopSimhash

这套设计和我们的任务是对齐的。

### 9.1 复用需要的是近似，不是精确

GraphhopSimhash 不追求“完全相同”，而是要找“足够相似”的旧节点。

### 9.2 多头 support 比单次命中更稳

一个候选如果在多个 head 上都命中，说明它不是偶然碰撞，而是更像真正可复用的锚点。

### 9.3 16bit/head 更适合做阈值感测

短 word 更容易把 `d=2` 和 `d=3` 拉开；这比直接做一个超长 128bit CAM 更现实。

### 9.4 前端与后端职责清晰

- HD-CAM 前端：负责近似搜索和候选筛选
- 数字后端：负责 support 聚合、残差修正、最终决策

这样系统更清楚，也更容易继续优化。

---

## 10. 一句话总结

我们这里的 HD-CAM 设计可以概括成一句话：

```text
把普通 CAM 的“完全相同才命中”，改成“match line 放电速度反映汉明距离”，
再让 8 个 16bit head 并行搜索，最后通过 support 聚合决定 direct / residual / compute。
```
