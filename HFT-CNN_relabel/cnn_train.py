#!/usr/bin/env python

#HFT-CNN#
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import numpy as np
import os.path
import six
from chainer.datasets import tuple_dataset
from tqdm import tqdm
import shutil
import random
import cnn_model
import xml_cnn_model
import pdb

USE_CUDNN = 'always' ## always, auto, or never
#予測結果が閾値より大きい数を取る関数
def select_function(scores):
    scores = chainer.cuda.to_cpu(scores)
    np_predicts = np.zeros(scores.shape,dtype=np.int8)
    for i in tqdm(range(len(scores)),desc="select labels based on threshold loop"):
        np_predicts[i] = (scores[i] >= 0.5)
    return np_predicts

#シード値の固定
def set_seed_random(seed):
        random.seed(seed)
        np.random.seed(seed)
        if chainer.cuda.available:
            chainer.cuda.cupy.random.seed(seed)

#メイン関数
def main(params):
    #ハイパーパラメータの表示
    print("")   
    print('# gpu: {}'.format(params["gpu"]))
    print('# unit: {}'.format(params["unit"]))
    print('# batch-size: {}'.format(params["batchsize"]))
    print('# epoch: {}'.format(params["epoch"]))
    print('# number of category: {}'.format(params["output-dimensions"]))
    print('# embedding dimension: {}'.format(params["embedding-dimensions"]))
    print('# current layer: {}'.format(params["currentDepth"])) 
    print('# model-type: {}'.format(params["model-type"])) 
    print('')
    #ハイパーパラメータの書き出し
    f = open('./CNN/LOG/configuration_' + params["currentDepth"] + '.txt', 'w')
    f.write('# gpu: {}'.format(params["gpu"])+"\n")
    f.write('# unit: {}'.format(params["unit"])+"\n")
    f.write('# batch-size: {}'.format(params["batchsize"])+"\n")
    f.write('# epoch: {}'.format(params["epoch"])+"\n")
    f.write('# number of category: {}'.format(params["output-dimensions"])+"\n")
    f.write('# embedding dimension: {}'.format(params["embedding-dimensions"])+"\n")
    f.write('# current layer: {}'.format(params["currentDepth"])+"\n")
    f.write('# model-type: {}'.format(params["model-type"])+"\n")
    f.write("\n")
    f.close()
    #分散表現
    embeddingWeights = params["embeddingWeights"]
    embeddingDimensions = params["embedding-dimensions"]
    #訓練データと検証データxがテキスト、yが正解クラス
    inputData = params["inputData"]
    x_train = inputData['X_trn']
    x_test = inputData['X_val']
    y_train = inputData['Y_trn']
    y_test = inputData['Y_val']
    y_train_e = inputData['origin_Y_trn']
    #CNNのハイパーパラメータ
    cnn_params = {"cudnn":USE_CUDNN, 
                "out_channels":params["outchannels"],
                "row_dim":embeddingDimensions, 
                "batch_size":params["batchsize"],
                "hidden_dim":params["unit"],
                "n_classes":params["output-dimensions"],
                "embeddingWeights":embeddingWeights,
                }
    #CNNのモデルを定義
    if params["fineTuning"] == 0:
        cnn_params['mode'] = 'scratch'
    elif params["fineTuning"] == 1:
        cnn_params['mode'] = 'fine-tuning'
        cnn_params['load_param_node_name'] = params['upperDepth']
        
    if params["model-type"] == "XML-CNN":
        model = xml_cnn_model.CNN(**cnn_params)
    else:
        model = cnn_model.CNN(**cnn_params)

    if params["gpu"] >= 0:
        chainer.cuda.get_device_from_id(params["gpu"]).use()
        model.to_gpu()
    #optimizerの設定
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    #テキストデータと正解クラスのタプルを作成する
    train = tuple_dataset.TupleDataset(x_train, y_train)
    test = tuple_dataset.TupleDataset(x_test, y_test)
    #ミニバッチを自動で生成する
    train_iter = chainer.iterators.SerialIterator(train, params["batchsize"], repeat=True, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(test, params["batchsize"], repeat = False, shuffle=False)
    
    #早期終了の設定
    #過学習を起こしてきたら学習を早めに切り上げる
    stop_trigger = training.triggers.EarlyStoppingTrigger(
    monitor='validation/main/loss',
    max_trigger=(params["epoch"], 'epoch'))

    #Updaterは学習の核となるもの
    from MyUpdater import MyUpdater
    updater = MyUpdater(train_iter, optimizer, params["output-dimensions"], device=params["gpu"])
    trainer = training.Trainer(updater, stop_trigger, out='./CNN/')
    
    #拡張機能
    from MyEvaluator import MyEvaluator
    trainer.extend(MyEvaluator(test_iter, model, class_dim=params["output-dimensions"], device=params["gpu"]))
    trainer.extend(extensions.dump_graph('main/loss'))
    #modelのパラメータ保存に関する部分
    #誤差が最小になった時に全パラメータを保存
    trainer.extend(extensions.snapshot_object(model, 'parameters_for_multi_label_model_' + params["value"] + '.npz'),trigger=training.triggers.MinValueTrigger('validation/main/loss',trigger=(1,'epoch')))
    #学習のログを出力
    trainer.extend(extensions.LogReport(log_name='LOG/log_' + params["currentDepth"] + ".txt", trigger=(1, 'epoch')))
    #端末画面に表示させる
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'elapsed_time']))    
    trainer.extend(extensions.ProgressBar())

    trainer.extend(
    extensions.PlotReport(['main/loss', 'validation/main/loss'],
                          'epoch', file_name='LOG/loss_' + params["currentDepth"] + '.png'))
    #学習開始
    trainer.run()

    #保存されるパラメータのパスの指定
    filename = 'parameters_for_multi_label_model_' + params["currentDepth"] + '.npz'
    src = './CNN/'
    dst = './CNN/PARAMS'
    shutil.move(os.path.join(src, filename), os.path.join(dst, filename))
    #テスト（推論）
    print ("-"*50)
    print ("Testing...")
    
    X_tst = inputData['X_tst']
    Y_tst = inputData['Y_tst']
    N_eval = len(X_tst)
    #上の層の学習でのパラメータを読み込む
    cnn_params['mode'] = 'test-predict'
    cnn_params['load_param_node_name'] = params["currentDepth"]
    #モデルの指定
    if params["model-type"] == "XML-CNN":
        model = xml_cnn_model.CNN(**cnn_params)
    else:
        model = cnn_model.CNN(**cnn_params)

    model.to_gpu()
    #ネットワークの出力を保存する行列を確保
    output = np.zeros([N_eval,params["output-dimensions"]],dtype=np.int8)
    #予測確率を書き出すためのファイル手続き
    output_probability_file_name = "CNN/RESULT/probability_" + params["currentDepth"] + ".csv"
    with open(output_probability_file_name, 'w') as f:
        f.write(','.join(params["learning_categories"])+"\n")
    #推論本体
    test_batch_size = params["batchsize"]
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        for i in tqdm(six.moves.range(0, N_eval, test_batch_size),desc="Predict Test loop"):
            x = chainer.Variable(chainer.cuda.to_gpu(X_tst[i:i + test_batch_size]))
            t = Y_tst[i:i + test_batch_size]
            net_output = F.sigmoid(model(x))
            output[i: i + test_batch_size] = select_function(net_output.data)
            with open(output_probability_file_name , 'a') as f:
                tmp = chainer.cuda.to_cpu(net_output.data)
                low_values_flags = tmp < 0.001
                tmp[low_values_flags] = 0
                np.savetxt(f,tmp,fmt='%.4g',delimiter=",")
    return output

#ラベルを変更する場合のmain
def main_relabel(params):
    print("")   
    print('# gpu: {}'.format(params["gpu"]))
    print('# unit: {}'.format(params["unit"]))
    print('# batch-size: {}'.format(params["batchsize"]))
    print('# epoch: {}'.format(params["epoch"]))
    print('# number of category: {}'.format(params["output-dimensions"]))
    print('# embedding dimension: {}'.format(params["embedding-dimensions"]))
    print('# current layer: {}'.format(params["currentDepth"]))
    print('# model-type: {}'.format(params["model-type"]))
    print('')


    f = open('./CNN/LOG/configuration_' + params["currentDepth"] + '.txt', 'w')
    f.write('# gpu: {}'.format(params["gpu"])+"\n")
    f.write('# unit: {}'.format(params["unit"])+"\n")
    f.write('# batch-size: {}'.format(params["batchsize"])+"\n")
    f.write('# epoch: {}'.format(params["epoch"])+"\n")
    f.write('# number of category: {}'.format(params["output-dimensions"])+"\n")
    f.write('# embedding dimension: {}'.format(params["embedding-dimensions"])+"\n")
    f.write('# current layer: {}'.format(params["currentDepth"])+"\n")
    f.write('# model-type: {}'.format(params["model-type"])+"\n")
    f.write("\n")
    f.close()

    embeddingWeights = params["embeddingWeights"]
    embeddingDimensions = params["embedding-dimensions"]
    value = params["value"]#現在のカテゴリー
    inputData = params["inputData"]
    x_train = inputData['X_trn']
    x_test = inputData['X_val']
    y_train = inputData['Y_trn']
    y_test = inputData['Y_val']
    changeflag = inputData['changeflag']#ラベルを変更するかどうかのフラグ
    y_train_e = inputData['origin_Y_trn']#訓練データのテキスト
    
    cnn_params = {"cudnn":USE_CUDNN, 
                "out_channels":params["outchannels"],
                "row_dim":embeddingDimensions, 
                "batch_size":params["batchsize"],
                "hidden_dim":params["unit"],
                "n_classes":params["output-dimensions"],
                "embeddingWeights":embeddingWeights,
                }
    if params["fineTuning"] == 0:
        cnn_params['mode'] = 'scratch'
    elif params["fineTuning"] == 1:
        cnn_params['mode'] = 'fine-tuning'
        cnn_params['load_param_node_name'] = params['upperDepth']
        
    if params["model-type"] == "XML-CNN":
        model = xml_cnn_model.CNN(**cnn_params)
    else:
        model = cnn_model.CNN(**cnn_params)

    if params["gpu"] >= 0:
        chainer.cuda.get_device_from_id(params["gpu"]).use()
        model.to_gpu()
    
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    train = tuple_dataset.TupleDataset(x_train, y_train,y_train_e,changeflag)#y_train_e, changeflagを含めたタプルを作成
    test = tuple_dataset.TupleDataset(x_test, y_test)
    train_iter = chainer.iterators.SerialIterator(train, params["batchsize"], repeat=True, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(test, params["batchsize"], repeat = False, shuffle=False)
    
    stop_trigger = training.triggers.EarlyStoppingTrigger(
    monitor='validation/main/loss',
    max_trigger=(params["epoch"], 'epoch'))

    value = params["value"]
    from MyUpdater import MyUpdater_relabel

    updater = MyUpdater_relabel(train_iter, optimizer, params["output-dimensions"], value, device=params["gpu"])
    trainer = training.Trainer(updater, stop_trigger, out='./CNN/')
    
    from MyEvaluator import MyEvaluator
    trainer.extend(MyEvaluator(test_iter, model, class_dim=params["output-dimensions"], device=params["gpu"]))
    trainer.extend(extensions.dump_graph('main/loss'))
    #valueごとにパラメータを保存
    trainer.extend(extensions.snapshot_object(model, 'parameters_for_multi_label_model_' + params["value"] + '.npz'),trigger=training.triggers.MinValueTrigger('validation/main/loss',trigger=(1,'epoch')))

    trainer.extend(extensions.LogReport(log_name='LOG/log_' + params["currentDepth"] + ".txt", trigger=(1, 'epoch')))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'elapsed_time']))    
    trainer.extend(extensions.ProgressBar())

    trainer.extend(
    extensions.PlotReport(['main/loss', 'validation/main/loss'],
                          'epoch', file_name='LOG/loss_' + params["currentDepth"] + '.png'))
    
    trainer.run()


    filename = 'parameters_for_multi_label_model_' + params["value"] + '.npz'
    src = './CNN/'
    dst = './CNN/PARAMS'
    shutil.move(os.path.join(src, filename), os.path.join(dst, filename))

    print ("-"*50)
    print ("Testing...")
    
    X_tst = inputData['X_tst']
    Y_tst = inputData['Y_tst']
    N_eval = len(X_tst)

    cnn_params['mode'] = 'test-predict'
    cnn_params['load_param_node_name'] = params["currentDepth"]
    
    if params["model-type"] == "XML-CNN":
        model = xml_cnn_model.CNN(**cnn_params)
    else:
        model = cnn_model.CNN(**cnn_params)

    model.to_gpu()
    output = np.zeros([N_eval,params["output-dimensions"]],dtype=np.int8)
    output_probability_file_name = "CNN/RESULT/probability_" + params["currentDepth"] + ".csv"
    with open(output_probability_file_name, 'w') as f:
        f.write(','.join(params["learning_categories"])+"\n")
 
    test_batch_size = params["batchsize"]
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        for i in tqdm(six.moves.range(0, N_eval, test_batch_size),desc="Predict Test loop"):
            x = chainer.Variable(chainer.cuda.to_gpu(X_tst[i:i + test_batch_size]))
            t = Y_tst[i:i + test_batch_size]
            net_output = F.sigmoid(model(x))
            output[i: i + test_batch_size] = select_function(net_output.data)
            with open(output_probability_file_name , 'a') as f:
                tmp = chainer.cuda.to_cpu(net_output.data)
                low_values_flags = tmp < 0.001
                tmp[low_values_flags] = 0
                np.savetxt(f,tmp,fmt='%.4g',delimiter=",")
    return output

#最上位の層の推論
def  load_top_level_weights(params):
    print ("-"*50)
    print ("Testing...")

    embeddingWeights = params["embeddingWeights"]
    embeddingDimensions = params["embedding-dimensions"]
    inputData = params["inputData"]

    cnn_params = {"cudnn":USE_CUDNN, 
                "out_channels":params["outchannels"],
                "row_dim":embeddingDimensions, 
                "batch_size":params["batchsize"],
                "hidden_dim":params["unit"],
                "n_classes":params["output-dimensions"],
                "embeddingWeights":embeddingWeights,
                }
                   
    X_tst = inputData['X_tst']
    Y_tst = inputData['Y_tst']
    N_eval = len(X_tst)
    cnn_params['mode'] = 'test-predict'
    cnn_params['load_param_node_name'] = params["currentDepth"]
    if params["model-type"] == "XML-CNN":
        model = xml_cnn_model.CNN(**cnn_params)
    else:
        model = cnn_model.CNN(**cnn_params)

    model.to_gpu()
    output = np.zeros([N_eval,params["output-dimensions"]],dtype=np.int8)
    output_probability_file_name = "CNN/RESULT/probability_" + params["currentDepth"] + ".csv"
    with open(output_probability_file_name, 'w') as f:
        f.write(','.join(params["learning_categories"])+"\n")
        
    test_batch_size = params["batchsize"]
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        for i in tqdm(six.moves.range(0, N_eval, test_batch_size),desc="Predict Test loop"):
            x = chainer.Variable(chainer.cuda.to_gpu(X_tst[i:i + params["batchsize"]]))
            t = Y_tst[i:i + test_batch_size]
            net_output = F.sigmoid(model(x))
            output[i: i + test_batch_size] = select_function(net_output.data)
            with open(output_probability_file_name , 'a') as f:
                tmp = chainer.cuda.to_cpu(net_output.data)
                low_values_flags = tmp < 0.001
                tmp[low_values_flags] = 0
                np.savetxt(f,tmp,fmt='%.4g',delimiter=",")
    return output



