# Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree
Code for paper "Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree", SANER 2020  
Requires:   
pytorch    
javalang  
pytorch-geometric  

## Data
The data is available at https://github.com/jacobwwh/graphmatch_clone/blob/master/BCB.zip and https://github.com/jacobwwh/graphmatch_clone/blob/master/googlejam4_src.zip

## Running
Run experiments on Google Code Jam:  
python main.py --dataset="GCJ" 
For BigCloneBench:  
python main.py  --dataset="BCB"

This operation include training, validation, testing and writing test results to files.   

Arguments:  
nextsib, ifedge, whileedge, foredge, blockedge, nexttoken, nextuse: whether to include these edge types in FA-AST  
data_setting: whether to perform data balance on training set  
  '0': no data balance  
  '11': pos:neg = 1:1  
  '13': pos:neg = 1:3  
  '0'/'11'/'13'/+'small': use a smaller version of the training set
