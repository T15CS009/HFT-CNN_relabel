#! /bin/bash
DataDIR=./c_Data #c15_c1511_Data
Train=${DataDIR}/train.out
Test=${DataDIR}/c_test_inputdata.txt
Valid=${DataDIR}/valid.out

## Embedding Weights Type (fastText .bin)
# EmbeddingWeightsPath=./Word_embedding_fastText
EmbeddingWeightsPath=./Word_embedding_fastText/model.bin
## Network Type (XML-CNN,  CNN-Flat,  CNN-Hierarchy,  CNN-fine-tuning or Pre-process)
ModelType=CNN-Hierarchy
### the limit of the sequence 
USE_WORDS=100
### TrEE FILE PATH
TreefilePath=./Tree/c.tree

mkdir -p CNN
mkdir -p CNN/PARAMS
mkdir -p CNN/LOG
mkdir -p CNN/RESULT
mkdir -p Word_embedding

python train.py ${Train} ${Test} ${Valid} ${EmbeddingWeightsPath} ${ModelType} ${TreefilePath} ${USE_WORDS} 
 #cat CNN/RESULT/f_score.txt | ./notices_result_slack.sh
