## HOPPITY: LEARNING GRAPH TRANSFORMATIONS TO DETECT AND FIX BUGS IN PROGRAMS

> Hoppity:学习图形转换以检测和修复程序中的错误
>
> ICLR 2020

### 概述

**问题定义**：现代代码库的庞大和复杂性使得它们几乎不可能完全没有错误。自动化工具在开发过程中检测和修复错误变得越来越重要。Hoppity 旨在解决这一挑战，特别是在 JavaScript 这种动态、弱类型语言中，错误形式多样，缺乏工具支持。

**方法论**：Hoppity 使用图结构来表示程序，并通过图神经网络（GNN）来学习如何转换这些图以修复错误。这种方法允许模型捕捉程序的结构和数据流，从而更好地理解代码的语义。

**模型架构**：Hoppity 的模型包括一个外部记忆（GNN）用于嵌入有缺陷的程序，以及一个中央控制器（LSTM）来执行一系列动作（例如，预测类型、生成补丁等）以执行修复。这个多步决策过程通过自回归模型实现。

**训练和推理**：Hoppity 在 GitHub 上收集的 290,715 个 JavaScript 代码变更提交上进行训练。在推理过程中，它使用束搜索（beam search）来近似地找到具有最高概率的修复。

**实验结果**：Hoppity 在 36,361 个程序中正确检测并修复了 9,490 个错误。在给定错误位置和修复类型的端到端修复任务中，Hoppity 的性能优于基线方法。

**与现有技术的比较**：Hoppity 与现有的自动化程序修复技术（如 Getafix 和 Zoncolan）相比，展示了更高的准确性和泛化能力。此外，与静态分析工具（如 TAJS）相比，Hoppity 在检测和修复错误方面表现出更好的性能。

**局限性和未来工作**：尽管 Hoppity 在实验中取得了显著的成果，但作者指出，模型在处理更复杂的错误（如涉及多个文件或需要多步修复的错误）方面仍有改进空间。未来的工作可能包括扩展目标错误类型、在集成开发环境（IDE）中部署 Hoppity 以进一步评估其准确性和实用性，以及将学习框架扩展到其他编程语言。

+++

### 图（Graph）在程序表示中的应用：

在 Hoppity 中，程序被表示为一个图结构，这个图包含了程序的抽象语法树（AST）节点以及它们之间的关系。图的节点可以代表程序中的各种元素，如变量、函数、表达式等，而边则表示这些元素之间的控制流、数据流或其他关系。例如，一个函数调用可能会在图中创建一个节点到另一个节点的边，表示调用关系。

#### 图神经网络（GNN）的作用：

GNN 在 Hoppity 中的作用是学习图结构的表示，以便捕捉程序中的复杂模式和关系。GNN 通过在图的节点上执行消息传递和聚合操作来更新节点的表示。这些表示随后可以用于执行各种任务，如错误检测、定位和修复。

在 Hoppity 的上下文中，GNN 的具体作用包括：

1. **节点嵌入**：GNN 为图中的每个节点生成一个低维向量表示。这些表示捕捉了节点的局部邻域信息，使得模型能够理解节点在程序中的作用和上下文。
2. **图表示**：通过聚合所有节点的表示，GNN 生成了整个图的表示。这个全局表示可以用来理解程序的整体结构和行为。
3. **错误修复**：GNN 的输出被用于指导图转换操作，包括添加、删除、替换节点值和类型等，这些操作旨在修复程序中的错误。例如，如果模型确定某个节点（代表一个变量）应该被替换，它会执行相应的图转换操作来生成修复后的程序。

#### GNN 的训练和推理：

在训练阶段，Hoppity 使用大量的代码变更数据来训练 GNN。这些数据包含了错误修复前后的程序图，模型通过学习这些图的差异来理解如何修复错误。

在推理阶段，当给定一个有缺陷的程序时，Hoppity 使用训练好的 GNN 来生成图的表示，并执行一系列图转换操作来尝试修复错误。这个过程涉及到预测错误的位置、选择合适的修复操作以及执行这些操作。

#### GNN 的优势：

使用 GNN 的优势在于其能够捕捉程序中的复杂结构和语义信息。与传统的基于规则的方法相比，GNN 能够从数据中学习到更丰富的模式，这使得 Hoppity 能够处理更多样化和复杂的错误类型。