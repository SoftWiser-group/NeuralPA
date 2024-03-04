# Learning to Represent Programs with Graphs

This is an official implementation of the ICLR'18 paper ["Learning to Represent Programs with Graphs"](https://arxiv.org/abs/1711.00740) 
using the PyTorch GNN library [ptgnn](https://github.com/microsoft/ptgnn) developed by Microsoft.

The variable misuse task is the problem of detecting variable misuse bugs in source code. The task is formulated as a classification problem for picking the correct node among a few candidates nodes for a given location in a program (a sort of fill in the blank task). Each candidate node represents a single variable that could be placed at a given location in the program. The decision needs to be made by considering the context (a graph representation of a program) for a given location.

Cite as
```
@article{allamanis2017learning,
  title={Learning to represent programs with graphs},
  author={Allamanis, Miltiadis and Brockschmidt, Marc and Khademi, Mahmoud},
  journal={arXiv preprint arXiv:1711.00740},
  year={2017}
}
```

## 方法说明
本文提出的方法使用GGNN学习程序在图上的表示，具体来说，通过学习待预测节点位置的context representation和候选节点（每个候选节点对应于一个变量）的usage representation来预测并修复程序中的变量误用。

## 数据准备
本文使用的数据集为从Github收集的top-starred且没有fork的C#项目。官方提供了一个已经提取成图并且划分好的数据集，可以在[此处](https://www.microsoft.com/en-us/download/details.aspx?id=56844)下载。

数据集中的图可以使用如下的格式来读取：
```Python
class VarMisuseGraph(TypedDict):
    Edges: Dict[str, List[Tuple[int, int]]]
    NodeLabels: Dict[str, str]
    NodeTypes: Dict[str, str]
```

## 运行说明
在`src/varmisuse`目录下, 运行
```commandline
python -m train.py TRAIN_DATA_PATH VALID_DATA_PATH TEST_DATA_PATH MODEL_FILENAME
```
进行训练和测试。其中`TRAIN_DATA_PATH` `VALID_DATA_PATH` `TEST_DATA_PATH`指向 train/validation/test 数据集路径， `MODEL_FILENAME` 为模型的保存路径，后缀名需要为`pkl.gz`。

更详细的说明可以参见Microsoft的[原仓库](https://github.com/microsoft/ptgnn)。