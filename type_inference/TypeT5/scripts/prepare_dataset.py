import os
from typet5.utils import proj_root, read_file, write_file, get_dataroot
from typet5 import proj_root
from typet5.data import get_dataset_dir, get_tk_dataset_name, PreprocessArgs
from typet5.data import create_tokenized_srcsets, load_tokenized_srcsets

"""
Note: This notebook preprocess the downloaded dataset into a format used by the training pipeline. 
This is useful if you only want to preprocess the data but not performing the training right away. 
Otherwise, you should directly run the `train_model.py` script and it will automatically preprocess
or load the dataset for you according to the experiment configuration.

处理下载好的数据集, 生成 tokenized_src_sets
"""
# 将当前工作目录更改为项目根目录的路径。
os.chdir(proj_root())
# 数据集名称
dataset_name = "ManyTypes4Py"
# 仓库名称
# repos_split_path = proj_root() /  "data/repos_split.pkl"
repos_dir = get_dataset_dir("ManyTypes4Py") / "repos"

recreate = False
func_only = True  # whether to create functional data (for TypeT5) or chunk data (for CodeT5)
pre_args = PreprocessArgs()  # 初始的配置参数
data_reduction = 1

# 构造加载数据集名称： 'func-ManyTypes4Py-v7-PreprocessArgs()'
tk_src_name = get_tk_dataset_name(
    dataset_name, pre_args, func_only, data_reduction=data_reduction,
)
# 构造数据集路径： /home/csc/Code/TypeInference/TypeT5/SPOT-data/func-ManyTypes4Py-v7-PreprocessArgs()
datasets_path = get_dataroot() / "SPOT-data" / tk_src_name

# 创建数据集
if recreate or not datasets_path.exists():
    create_tokenized_srcsets(
        # proj_root() / "data/repos_split.pkl", # 这里源代码应该有错
        proj_root() / "ManyTypes4Py",
        datasets_path,
        func_only=func_only,
        pre_args=pre_args,
        data_reduction=data_reduction,
    )
# 加载数据集; load_tokenized_srcsets 方法参数如下
#   path: Path,
#   quicktest: bool = False,
#   sets_to_load=["test", "train", "valid"],
# 这里将 tk_src_name 传入给 quicktest(成为 true)，会选择较少的 data
tk_dataset = load_tokenized_srcsets(datasets_path, tk_src_name)

# 输出数据集的基本情况
print("dataset:", datasets_path)
tk_dataset["train"].print_stats()

long_files = sorted(tk_dataset["train"].all_srcs, key=lambda s: len(s.tokenized_code), reverse=True)
print(long_files[8].preamble_code)
