# Paper Docs

这个目录对应的是“论文与外部参考层”。

它的目标不是解释当前代码怎么跑，而是回答：

- 这个项目参考了哪些 CAM / HD-CAM / NVM / hybrid retrieval 文献；
- 各条技术路线的器件、匹配语义和优缺点是什么；
- 现有工程和论文世界之间怎么对齐。

## 当前主文档

- Survey 主文档：
  [CAM_SURVEY.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/paper/survey/CAM_SURVEY.md)

## 子目录说明

- `CAM/`
  参考论文 PDF 库
- `survey/`
  结构化整理和归纳文档
- `verify/`
  工具链或外部验证相关材料

## 建议维护规则

- PDF 库本身不承担结论表达职责，结论应写进 survey
- 如果一篇论文被项目正式引用，最好在 survey 里有明确条目
- 和当前实现强相关的论文，优先在 survey 里标出“与 CAM_sim 的关系”
