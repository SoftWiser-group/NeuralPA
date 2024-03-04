# Typilus
This is a self-implemented version of the PLDI'20 paper ["Typilus: Neural Type Hints"](https://arxiv.org/abs/2004.10657)
using the PyTorch GNN library [ptgnn](https://github.com/microsoft/ptgnn) developed by Microsoft.

Typilus is a deep metric-learning method for predicting types in Python code.

At a high level, Typilus consists of two parts:

1. Front-end, which parses Python programs, represents them in graphs, and
   splits the graphs into training set, validation set, and test set.
2. Back-end, which takes the graphs as inputs and trains a model or outputs
   predictions.

The official implementation of Typilus is [here](https://github.com/typilus/typilus). Cite as 
```
@inproceedings{allamanis2020typilus,
  title={Typilus: Neural Type Hints},
  author={Allamanis, Miltiadis and Barr, Earl T and Ducousso, Soline and Gao, Zheng},
  booktitle={PLDI},
  year={2020}
}
```

## 方法说明

<img src="data/typilusPoster.jpg"  alt="Typilus">

Typilus通过将classification和metric-learning结合的方式，在GNN上学习graph的表示，其中的supernode表示待预测类型的变量。获取supernode的表示后，Typilus使用K-近邻算法来对supernode的类型进行预测。

## 数据准备

Typilus接受从Python程序提取的graph作为输入，数据集文件后缀名应为`jsonl.gz`，其中每一行表示格式如下的一张graph：
```json
{
  "nodes": [list-of-nodes-str],
  "edges": {
    "EDGE_TYPE_NAME_1": {
        "from_node_idx": [to_node_idx1, to_node_idx2, ...],
        ...
    },
    "EDGE_TYPE_NAME_2": {...},
    ...
  },
  "token-sequence": [node-idxs-in-token-sequence],
  "supernodes": {
    "supernode1-node-idx": {
      "name": "the-name-of-the-supernode",
      "annotation": null or str,
      ...
    },
    "supernode2-node-idx": { ... },
    ...
  }
 "filename": "provenance-info-str"
}
```
具体描述可参见ptgnn的官方[tutorial](https://github.com/microsoft/ptgnn/blob/master/docs/tutorial.md)。

Typilus使用从Github上获取的repo来作为原始数据集，通过对原始数据集进行预处理和graph提取来获得train, validation和test数据集，具体使用的repo列表和graph提取的过程可参见原仓库的`data_preparation`部分。

## 运行说明
在`src/typilus`目录下，使用
```commandline
python train.py [options] TRAIN_DATA_PATH VALID_DATA_PATH TEST_DATA_PATH MODEL_FILENAME
```
进行训练和测试。其中`TRAIN_DATA_PATH`，`VALID_DATA_PATH`，`TEST_DATA_PATH`分别对应train, validation和test数据集的路径，`MODEL_FILENAME`为模型的保存路径

注：模型保存路径的后缀名必须为`pkl.gz`

使用
```commandline
python predict.py [options] MODEL_FILENAME DATA_PATH
```
指定训练好的模型并对给定数据进行预测。其中`MODEL_FILENAME`表示模型路径，`DATA_PATH`表示数据集的路径。