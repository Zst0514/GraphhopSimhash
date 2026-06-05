# Codex Execution Discipline

本文档记录 GraphhopSimhash 项目中 Codex 后续执行任务时的默认协作规则。目标是减少无关上下文、控制日志输出、避免重复解释，让每次修改更直接、更可复现。

## 0. 启动规则

进入新任务时，先确认当前任务范围和仓库位置：

```bash
pwd
git status --short
```

如果用户指定“只改某个文件/只看某个结果”，则不主动扫描全仓库。

如果当前工作目录不在 `GraphhopSimhash` 仓库内，先切到：

```text
/home/zhangshangtong/Transformer/OFA/GraphhopSimhash
```

## 1. 任务范围

如果用户明确限定任务，例如：

```text
只改这个 md
只看这个日志
不要解释历史
只给结论
```

则只处理该范围内的文件和信息，不主动展开历史背景、不引入额外实验脉络。

如果用户是在问概念问题，优先直接回答；只有用户明确要求“修改/更新/实现/跑实验”时才动文件或跑长任务。

## 2. 日志读取

默认不整段展开大日志。优先使用：

```bash
rg "FINAL|SUMMARY|FullP8|AllP4|Deg|TSER" <log>
tail -n 80 <log>
```

只有在需要定位错误、复现实验或核对表格来源时，才读取更长日志。

命令输出默认限流：

```text
短检查: 2000-5000 tokens
日志摘要: 8000-12000 tokens
完整 diff / 大表: 写入文件，不在对话中展开
```

不要用 `cat` 直接输出大日志或大表。

## 3. 表格与结果

大表格优先写入对应 `.md` 文档，不在对话里完整展开。

对话回复只保留：

```text
改了什么
关键结论
文件位置
是否已 push
```

如果用户要求解释表格，再按列解释。

表格需要包含对照项时，优先补齐以下关系：

```text
reference / oracle
low-cost baseline
proposed policy
random baseline
best rescue / cost saving
```

## 4. 已定结论

对已经确认过的设计和参数，不反复复述背景。

例如以下内容默认视为当前主线背景：

```text
SimHash / LRU-CAM 前端
TSER-guided residual reuse
Progressive BFP encoder path
UNIFIED_FRONTEND_POLICY_RESULT.md 中的当前参数表
```

除非用户要求，否则后续回复只引用结论，不重新讲完整来龙去脉。

如果主线参数发生变化，要优先更新对应结果文档，而不是在聊天里只口头说明。

## 5. Git 操作

提交前只 stage 当前任务相关文件，不带入其它本地改动。

如果用户没有要求 push，只在本地修改并说明尚未 push。

如果用户要求 push，执行顺序默认是：


```bash
git status --short
git diff --check -- <changed-files>
git add <changed-files>
git commit -m "<message>"
git pull --rebase --autostash origin main
git push origin main
```

push 后回复 commit hash。若仓库中还有其它未提交改动，要明确说明这些改动没有被本次提交带上。

## 6. 实验执行

不主动启动长时间实验，除非用户明确要求。

运行实验时要记录：

```text
命令
输出目录
日志路径
核心结果
是否后台运行
```

长任务优先写入 `output/<experiment_name>/...`，避免散落临时文件。

## 7. 回复风格

默认短回复：

```text
已完成 / 未完成
结果在哪里
关键数值是什么
下一步建议
```

避免无必要的长背景解释、重复辩解或大段历史复盘。

默认不把长表格贴到对话里。用户要求“怎么看”时，解释列含义和关键取舍即可。
