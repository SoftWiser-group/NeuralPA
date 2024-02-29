# FastSMT

This is the tool released in the NeurIPS'18 paper：[Learning to Solve SMT Formulas](https://www.sri.inf.ethz.ch/publications/balunovic2018learnsmt)
repo：https://github.com/eth-sri/fastsmt.git

## 方法介绍

FastSMT是使用学习方法在特定SMT公式集上训练，以此生成能够快速解决该公式集公式的求解策略的程序。其方法主要分为两个阶段，即带参Tactic序列生成阶段和决策树合成Strategy阶段。

![image](https://github.com/Wang-hn/imgs/blob/main/NeuralPA/fastsmt.png)

第一个阶段通过进化搜索方法探索可能的Tactic决策序列，根据求解公式的情况，以Tactic的embedding和公式的probe信息等内容生成训练数据，并用这些数据训练神经网络，从而指导公式对当前所用的Tactic进行选择。在该过程中，程序会对每个公式收集在生成过程中表现最好的Tactic序列，并用这些序列作为第二个阶段的输入，通过一种类决策树的数据结构处理序列，合成出一个在数据集上表现良好的可解释的Strategy。

## 运行说明

源仓库中对的代码如何运行已经进行了较为详细的说明，并附有其所使用的数据集和浮现实验所需的shell脚本，只需参照源程序运行即可。值得注意的是，FastSMT的性能与Z3的版本有强相关性，不同的版本表现可能大不相同，建议在复现时根据源repo中的说明逐步执行，确保复现实验过程中的数据准确。

## Cite

```
@incollection{NIPS2018_8233,
  title = {Learning to Solve SMT Formulas},
  author = {Balunovic, Mislav and Bielik, Pavol and Vechev, Martin},
  booktitle = {Advances in Neural Information Processing Systems 31},
  editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
  pages = {10337--10348},
  year = {2018},
  publisher = {Curran Associates, Inc.},
  url = {http://papers.nips.cc/paper/8233-learning-to-solve-smt-formulas.pdf}
}
```

