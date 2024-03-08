# Bug Detection based on Graph Neural Networks

GNN4bug detection

## 前言

+ [写在前面](ch0/intro.md)
+ [主流程序分析技术](ch0/program_analysis.md)
+ [为什么是GNN](ch0/why-GNN.md)

## 论文整理

+ [Self-Supervised Bug Detection and Repair](ch1/Self-Supervised Bug Detection and Repair.md)*（精读）*
+ [HOPPITY: LEARNING GRAPH TRANSFORMATIONS TO DETECT AND FIX BUGS IN PROGRAMS](ch1/HOPPITY.md)
+ 对比总结
  - Hoppity 专注于通过图转换操作直接修复程序中的错误，而 BUGLAB 则侧重于通过自监督学习提高 bug 检测和修复的泛化能力。
  - 在 Hoppity 中，GNN 用于学习图的表示，以便执行修复操作。在 BUGLAB 中，GNN 用于学习代码实体的表示，这些表示随后用于训练选择和检测模型。
  - GNN 能够处理程序表示为图的数据结构，其中节点代表程序元素（如变量、函数声明等），边代表它们之间的关系（如调用、赋值等）。这种结构化表示有助于 GNN 学习程序的深层结构特征。
  - GNN 通过在图上进行消息传递和聚合，能够捕捉程序中的复杂模式和依赖关系。这对于识别那些需要深入理解程序逻辑的错误至关重要。

## 代码复现

+ **正在更新···**

