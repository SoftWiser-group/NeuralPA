readonly SRC_BASE="/usr/data/tools/typilus/src/data_preparation/scripts/"
export PYTHONPATH="$SRC_BASE"
dataset_dir="./typilus"
mkdir -p $dataset_dir/graph-dataset
python3 "$SRC_BASE"graph_generator/extract_graphs.py $dataset_dir/raw_repos/ $dataset_dir/corpus_duplicates.json $dataset_dir/graph-dataset $SRC_BASE../metadata/typingRules.json --debug
mkdir -p $dataset_dir/graph-dataset-split
python3 "$SRC_BASE"utils/split.py -data-dir $dataset_dir/graph-dataset -out-dir $dataset_dir/graph-dataset-split