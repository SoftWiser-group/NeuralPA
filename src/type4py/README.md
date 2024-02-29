# Type4Py

This is the tool released in the ICSE'22 paper [Type4Py: practical deep similarity learning-based type inference for python](https://dl.acm.org/doi/abs/10.1145/3510003.3510124)
Github repo: [https://github.com/saltudelft/type4py/](https://github.com/saltudelft/type4py/)

## 方法说明

![imgs from type4py](https://github.com/Wang-hn/imgs/blob/main/NeuralPA/fastsmt.png)

该工具为了解决动态语言中缺乏静态类型支持的问题，提出了一种基于学习的针对python语言的类型推断方案。其将python源代码中的上下文信息分割作为语料，先使用如stop word removal、lemmatization等传统NLP方法对数据进行清洗后，使用Word2Vec将其转化为词向量表。

而后，程序针对不同标识符，可以通过对函数名、参数名、变量名等信息进行拼接后获得词向量，而变量的上下文信息则能够通过对使用点（或返回点）的拼接（具有窗口大小）后再通过向量表得到对应的向量。这两类向量经过不同的RNN处理后能分别获得一个固定长度的向量，再与能够从源代码中提取的显式提示信息相结合，使用回归方法输出一个新的向量。程序使用Triplet loss作为损失函数对数据进行训练，以此对具有相同类型的预测向量进行聚类。在进行预测时使用k-nearest neighbor方法选取临近点后计算可能的类型概率并由此做出推断。

该方法在传统的类型标注之外，通过寻找变量的上下文信息对各个变量进行建模，并使用回归而非分类方法定义输出，通过聚类算法进行训练并使用邻近算法得到预测信息。

## 运行说明

源代码仓库已经对如何使用该工具进行类型推断做出了详细的说明，各个部分的代码也分别在对应的python文件中写明入口方法。同时，项目已经在服务器上进行了部署，具有对应的VS code插件，本节内容仅对使用过程中出现的问题进行说明，如需更深入了解请参考源repo。

首先，在使用工具前请注意程序对内存和显存具有一定要求（若使用ManyTypes4Py数据集）。其次，在特定运行环境下，原repo中的preprocessing部分可能抛出FileNotFoundError、如使用GPU训练可能因Tensor不在同一个device上出现RuntimeError，在复现过程中已对相应报错点进行对应更改（仅能保证在本机环境下能够正常运行）。

## Cite

```
@inproceedings{mir2022type4py,
  title={Type4Py: practical deep similarity learning-based type inference for python},
  author={Mir, Amir M and Lato{\v{s}}kinas, Evaldas and Proksch, Sebastian and Gousios, Georgios},
  booktitle={Proceedings of the 44th International Conference on Software Engineering},
  pages={2241--2252},
  year={2022}
}
```