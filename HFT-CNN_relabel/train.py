#!/usr/bin/env python

#HFT-CNN
#
import sys
from collections import defaultdict
import numpy as np
import data_helper
import cnn_train
import random
import scipy.sparse as sp
import tree
import os
import pdb

#学習を呼び出す関数
def train_problem(currentDepth, upperDepth, classNum, fineTuning, embeddingWeights, inputData, modelType, learning_categories):
    params = {"gpu":0, 
                "outchannels":128,
                # "embedding-dimensions":300,
                "embedding-dimensions":100, 
                "epoch":40, 
                "batchsize":32,
                "unit":1024, 
                "output-dimensions":int(classNum), 
                "fineTuning":int(fineTuning), 
                "currentDepth":currentDepth, 
                "upperDepth":upperDepth, 
                "embeddingWeights": embeddingWeights,
                "inputData": inputData,
                "model-type": modelType,
                "learning_categories": learning_categories
                }
    if params["model-type"] == "XML-CNN":
        params["unit"] = 512 # compact representation
    if (params["model-type"] == "CNN-fine-tuning") and (currentDepth == "1st"):
        params["fineTuning"] = 0

    if (currentDepth == "1st") and ((params["model-type"] == "CNN-fine-tuning") or  (params["model-type"] == "CNN-Hierarchy")):
        network_output = cnn_train.load_top_level_weights(params)
    else:
        network_output = cnn_train.main(params)
    
    return network_output

#ラベルを変更する場合のtrain_probrem
def train_problem_relabel(currentDepth, upperDepth, classNum, fineTuning, embeddingWeights, inputData, modelType, learning_categories,value):
    params = {"gpu":0, 
                "outchannels":128,
                "embedding-dimensions":100, 
                "epoch":40, 
                "batchsize":32,
                "unit":1024, 
                "output-dimensions":int(classNum), 
                "fineTuning":int(fineTuning), 
                "currentDepth":currentDepth, 
                "upperDepth":upperDepth, 
                "embeddingWeights": embeddingWeights,
                "inputData": inputData,
                "model-type": modelType,
                "learning_categories": learning_categories,
                "value":value
                }
    if params["model-type"] == "XML-CNN":
        params["unit"] = 512 # compact representation
    if (params["model-type"] == "CNN-fine-tuning") and (currentDepth == "1st"):
        params["fineTuning"] = 0

    if (currentDepth == "1st") and ((params["model-type"] == "CNN-fine-tuning") or  (params["model-type"] == "CNN-Hierarchy")):
        network_output = cnn_train.load_top_level_weights(params)
    else:
        network_output = cnn_train.main_relabel(params)
    
    return network_output

#階層構造の情報を取得
def make_labels_hie_info_dic(treePath):
        label_hierarchical_info_dic = {}
        with open(treePath, "r") as f:
            for line in f:
                line = line[:-1]
                category = line.split("<")[-1]
                level = len(line.split("<"))
                if category not in label_hierarchical_info_dic:
                        label_hierarchical_info_dic[category] = level
        # pdb.set_trace()
        return label_hierarchical_info_dic

#階層構造の情報を使用しカテゴリーのリストを作成
def make_labels_hie_list_dic(labels, label_hierarchical_info_dic):
        layer_category_list_dic = {}
        for i in range(1,max(label_hierarchical_info_dic.values())+1):
                a_set = set([])
                layer_category_list_dic[i] = a_set
        for label in labels:
            layer_category_list_dic[int(label_hierarchical_info_dic[label])].add(label)
        # pdb.set_trace()
        return layer_category_list_dic

#読み込んだ階層のデータから階層構造を取得
def make_tree(treeFile_path):
    Tree = tree.make()
    with open(treeFile_path, mode="r") as f:
        for line in f:
            line = line[:-1]
            line = line.split("\t")[0]
            line = line.split("<")
            tree.add(Tree, line)
    return Tree


# メインの学習処理 
# ==================================================================
def main():
    random.seed(0)
    np.random.seed(0)

    # データの読み込み
    # ==========================================================
    print ('-'*50)
    print ('Loading data...')
    train = sys.argv[1]
    test = sys.argv[2]
    validation = sys.argv[3]
    embeddingWeights_path = sys.argv[4]
    modelType = sys.argv[5]
    treeFile_path = sys.argv[6]
    useWords = int(sys.argv[7])

    f_train = open(train, 'r')
    train_lines = f_train.readlines()
    f_test = open(test, 'r')
    test_lines = f_test.readlines()
    f_valid = open(validation, 'r')
    valid_lines = f_valid.readlines()
    f_train.close()
    f_test.close()
    f_valid.close()
   

    # モデルの選択
    # =========================================================
    fineTuning = 0
    if modelType == "XML-CNN" or modelType == "CNN-Flat":
        categorizationType="flat"
        fineTuning = 0
    elif modelType == "CNN-Hierarchy":
        categorizationType="hierarchy"
        fineTuning = 0
    elif modelType == "CNN-fine-tuning":
        categorizationType="hierarchy"
        fineTuning = 1
    elif modelType == "Pre-process":
        categorizationType = "pre-process"
        fineTuning = 0
    else:
        raise TypeError('Unknown model type: %s!' % (modelType))

    
    # pre-processingのときの処理
    # ========================================================
    if categorizationType == "pre-process":
        # 階層構造の情報を取得
        # =========================================================
      
        category_hie_info_dic = make_labels_hie_info_dic(treeFile_path)
        input_data_dic = data_helper.data_load(train_lines, valid_lines, test_lines, category_hie_info_dic, useWords)
        category_hie_list_dic = make_labels_hie_list_dic(list(input_data_dic['catgy'].keys()), category_hie_info_dic)
        # 分散表現の読み込み  
        # =========================================================
        print ('-'*50)
        print ("Loading Word embedings...")
        embeddingWeights=data_helper.embedding_weights_load(input_data_dic['vocab'], embeddingWeights_path)
   

        #メインの学習処理
        print ('-'*50)
        print ("Pre-process for hierarchical categorization...")
        Tree = make_tree(treeFile_path)
        layer = 1
        depth = data_helper.order_n(1)#深さの指定
        upperDepth = None
        learning_categories = sorted(category_hie_list_dic[layer])#ソートした新たなリストを生成
        X_trn, Y_trn, X_val, Y_val, X_tst, Y_tst,origin_Y_trn =  data_helper.build_problem(learning_categories=learning_categories,depth=depth, input_data_dic=input_data_dic)#データの整形
        input_network_data = {"X_trn":X_trn, "Y_trn":Y_trn, "X_val":X_val, "Y_val":Y_val, "X_tst":X_tst, "Y_tst":Y_tst,"origin_Y_trn":origin_Y_trn}#データをまとめる
        Y_pred = train_problem(currentDepth=depth, upperDepth=upperDepth, classNum=len(learning_categories), fineTuning=fineTuning, embeddingWeights=embeddingWeights, inputData=input_network_data, modelType=modelType, learning_categories=learning_categoriesm)#学習を呼び出し，その後学習結果がY_predに格納される
        print ("Please change model-type to CNN-Hierarchy of CNN-fine-tuning.")
    
    
    #  flat のモデルの時の処理
    # ========================================================
    elif categorizationType == "flat":
        print ('-'*50)
        print ("Processing in case of flat categorization...")
        from itertools import chain
        learning_categories = sorted(input_data_dic['catgy'].keys()) ## this order is network's output order.
        X_trn, Y_trn, X_val, Y_val, X_tst, Y_tst, origin_Y_trn =  data_helper.build_problem(learning_categories=learning_categories,depth="flat", input_data_dic=input_data_dic)
        input_network_data = {"X_trn":X_trn, "Y_trn":Y_trn, "X_val":X_val, "Y_val":Y_val, "X_tst":X_tst, "Y_tst":Y_tst,"origin_Y_trn":origin_Y_trn}
        Y_pred = train_problem(currentDepth="flat", upperDepth=None, classNum=len(learning_categories), fineTuning=fineTuning, embeddingWeights=embeddingWeights, inputData=input_network_data, modelType=modelType, learning_categories=learning_categories)
        GrandLabels, PredResult = data_helper.get_catgy_mapping(learning_categories, Y_tst, Y_pred, "flat")#ネットワークの予測結果と真の正解をファイルに書き出す
        data_helper.write_out_prediction(GrandLabels, PredResult, input_data_dic)#ネットワークの予測確率をファイルに書き出す
        
    # hierarchical のモデルの時の処理
    # ========================================================
    elif categorizationType == "hierarchy":
        if not os.path.exists("./CNN/PARAMS/parameters_for_multi_label_model_1st.npz"):
            raise FileNotFoundError('Please change "ModelType=CNN-Hierarchy" or "ModelType=CNN-fine-tuning" to "ModelType=Pre-process" in example.sh.')
        
        #階層構造の情報を取得
        category_hie_info_dic = make_labels_hie_info_dic(treeFile_path)
        input_data_dic = data_helper.data_load(train_lines, valid_lines, test_lines, category_hie_info_dic, useWords)
        category_hie_list_dic = make_labels_hie_list_dic(list(input_data_dic['catgy'].keys()), category_hie_info_dic)

        # 分散表現の読み込み
        # =========================================================
        print ('-'*50)
        print ("Loading Word embedings...")
        embeddingWeights=data_helper.embedding_weights_load(input_data_dic['vocab'], embeddingWeights_path)    
       
        #メインの学習処理
        print ('-'*50)
        print ("Processing in case of hierarchical categorization...")
        upperDepth = None
        Y_tst_concat = [[] for i in range(len(input_data_dic['test']))]
        Y_pred_concat = [[] for i in range(len(input_data_dic['test']))]
        all_categories = []
        Tree = make_tree(treeFile_path)
        layers =list(category_hie_list_dic.keys())
        flag = 0
       
        #層ごとの学習を行う 
        for layer in layers:
            values = sorted(category_hie_list_dic.get(layer))#今いる層のカテゴリ−
            child_list = tree.search_child(Tree,str(values[0]))#今いる層の子のカテゴリーのリスト
            depth = data_helper.order_n(layer)
            print ('-'*50)
            print ('Learning and categorization processing of ' + depth + ' layer')
            #2層目以降の学習を行う
            if not layer == 1 :
                #ラベルの変更をおこなうため一つのデータごとに見ていく
                for value in values:
                    
                    #子の再ラベル付け
                    #カテゴリーを変換
                    parent = tree.search_parent(Tree,str(value))
                   
                    print('-'*50)
                    print('Relabeling and Learning of '+ value + ' category')    
                    relabel = True
                    category_hie_info_dic = make_labels_hie_info_dic(treeFile_path)
                    
                    input_data_ori_dic = data_helper.data_load_relabel(train_lines,category_hie_info_dic,useWords,value,parent,relabel)#valueのカテゴリーを変更する
                    category_hie_list_dic = make_labels_hie_list_dic(list(input_data_ori_dic['catgy'].keys()), category_hie_info_dic)#階層の情報を取得する
                   
                    learning_categories = sorted(category_hie_list_dic[layer])#現在の層のカテゴリ−を取得
                    X_trn, Y_trn, X_val, Y_val, X_tst, Y_tst, origin_Y_trn, changeflag =  data_helper.build_problem_relabel(learning_categories=learning_categories,depth=depth, input_data_dic=input_data_dic, input_data_ori_dic=input_data_ori_dic) #flag
                    input_network_data = {"X_trn":X_trn, "Y_trn":Y_trn, "X_val":X_val, "Y_val":Y_val, "X_tst":X_tst, "Y_tst":Y_tst, "origin_Y_trn":origin_Y_trn,"changeflag":changeflag}
                    Y_pred = train_problem_relabel(currentDepth=depth, upperDepth=upperDepth, classNum=len(learning_categories), fineTuning=fineTuning, embeddingWeights=embeddingWeights, inputData=input_network_data, modelType=modelType, learning_categories=learning_categories,value=value)

                    GrandLabels, PredResult = data_helper.get_catgy_mapping(learning_categories, Y_tst, Y_pred, depth)
                    upperDepth = depth
                    for i in range(len(input_data_dic['test'])):
                        Y_tst_concat[i].extend(GrandLabels[i])#テストデータのカテゴリーを格納
                    for i in range(len(input_data_dic['test'])):
                        for y in PredResult[i]:
                           if (tree.search_parent(Tree, y) in Y_pred_concat[i]) or (tree.search_parent(Tree, y) == 'root'):
                              Y_pred_concat[i].append(y)#予測結果のカテゴリーを格納
                    all_categories += learning_categories
                    data_helper.write_out_f_score_relabel(learning_categories,GrandLabels, PredResult,depth,value)#Fスコアを取得する
                    data_helper.write_out_prediction(GrandLabels, PredResult, input_data_dic,value)#予測結果をファイルに出力
            #1層目の学習を行う
            else:
                learning_categories = sorted(category_hie_list_dic[layer])
                X_trn, Y_trn, X_val, Y_val, X_tst, Y_tst, origin_Y_trn = data_helper.build_problem(learning_categories=learning_categories,depth=depth, input_data_dic=input_data_dic) #flag
                # pdb.set_trace()
                input_network_data = {"X_trn":X_trn, "Y_trn":Y_trn, "X_val":X_val, "Y_val":Y_val, "X_tst":X_tst, "Y_tst":Y_tst, "origin_Y_trn":origin_Y_trn}
                Y_pred = train_problem(currentDepth=depth, upperDepth=upperDepth, classNum=len(learning_categories), fineTuning=fineTuning, embeddingWeights=embeddingWeights, inputData=input_network_data, modelType=modelType, learning_categories=learning_categories)
                GrandLabels, PredResult = data_helper.get_catgy_mapping(learning_categories, Y_tst, Y_pred, depth)
       	        upperDepth = depth
                for i in range(len(input_data_dic['test'])):
                    Y_tst_concat[i].extend(GrandLabels[i])
                for i in range(len(input_data_dic['test'])):
                   for y in PredResult[i]:
                       if (tree.search_parent(Tree, y) in Y_pred_concat[i]) or (tree.search_parent(Tree, y) == 'root'):
                           Y_pred_concat[i].append(y)
                all_categories += learning_categories
                data_helper.write_out_f_score(learning_categories,GrandLabels, PredResult,depth)
       
        print ('-'*50)
        print ('Final Result')

#メインの呼び出し
if __name__ == "__main__":
        main()


