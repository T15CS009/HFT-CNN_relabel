#! /usr/bin/env python
#
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import random


class CNN(chainer.Chain):
    #CNNを定義する　__init__は一度だけ実行される
    def __init__(self, **params):
        self.in_channels = 1
        self.out_channels = params["out_channels"]
        self.row_dim = params["row_dim"]
        self.batch_size = params["batch_size"] if "batch_size" in params else 100
        self.hidden_dim = params["hidden_dim"]
        self.n_classes = params["n_classes"]
        self.mode = params["mode"] if "mode" in params else None
        self.load_param_node_name = params["load_param_node_name"] if "load_param_node_name" in params else None 
        self.cudnn = params["cudnn"] if "cudnn" in params else 'never'
        self.embeddingWeights = params["embeddingWeights"]
        self.initializer = chainer.initializers.HeNormal()
        #モデルごとの層の定義
        if self.mode == "fine-tuning":
            parameters = np.load('./CNN/PARAMS/parameters_for_multi_label_model_' + self.load_param_node_name + '.npz')
            super(CNN, self).__init__()
            set_seed_random(0)
            with self.init_scope():
                self.lookup = L.EmbedID(in_size = parameters['lookup/W'].shape[0], out_size = parameters['lookup/W'].shape[1], initialW = parameters['lookup/W'], ignore_label = -1)
                self.conv1 = L.Convolution2D(self.in_channels,self.out_channels,(2, self.row_dim),stride=1,initialW=parameters['conv1/W'],initial_bias=parameters['conv1/b'])   
                self.conv2 = L.Convolution2D(self.in_channels,self.out_channels,(3, self.row_dim),stride=1,initialW=parameters['conv2/W'],initial_bias=parameters['conv2/b'])
                self.conv3 = L.Convolution2D(self.in_channels,self.out_channels,(4, self.row_dim),stride=1,initialW=parameters['conv3/W'],initial_bias=parameters['conv3/b'])
                self.l1=L.Linear(in_size = None, out_size = self.hidden_dim, initialW=self.initializer)
                self.l2=L.Linear(in_size = self.hidden_dim, out_size = self.n_classes, initialW=self.initializer)                
            
        elif self.mode == "scratch":
            super(CNN, self).__init__()
            set_seed_random(0)
            with self.init_scope():
                self.lookup = L.EmbedID(in_size = self.embeddingWeights.shape[0], out_size = self.embeddingWeights.shape[1], initialW = self.embeddingWeights, ignore_label = -1)
                self.conv1 = L.Convolution2D(self.in_channels,self.out_channels,(2, self.row_dim), stride=1,initialW=self.initializer)
                self.conv2 = L.Convolution2D(self.in_channels,self.out_channels,(3, self.row_dim), stride=1,initialW=self.initializer)
                self.conv3 = L.Convolution2D(self.in_channels,self.out_channels,(4, self.row_dim), stride=1,initialW=self.initializer)
                self.l1=L.Linear(in_size = None, out_size = self.hidden_dim, initialW=self.initializer)
                self.l2=L.Linear(in_size = self.hidden_dim, out_size = self.n_classes, initialW=self.initializer)
                 
        elif self.mode == "test-predict":
            parameters = np.load('./CNN/PARAMS/parameters_for_multi_label_model_' + self.load_param_node_name +'.npz')
            super(CNN, self).__init__()
            set_seed_random(0)
            with self.init_scope():
                self.lookup = L.EmbedID(in_size = self.embeddingWeights.shape[0], out_size = self.embeddingWeights.shape[1], initialW = parameters['lookup/W'], ignore_label = -1)
                self.conv1 = L.Convolution2D(self.in_channels,self.out_channels,(2, self.row_dim),stride=1,initialW=parameters['conv1/W'],initial_bias=parameters['conv1/b'])   
                self.conv2 = L.Convolution2D(self.in_channels,self.out_channels,(3, self.row_dim),stride=1,initialW=parameters['conv2/W'],initial_bias=parameters['conv2/b'])
                self.conv3 = L.Convolution2D(self.in_channels,self.out_channels,(4, self.row_dim),stride=1,initialW=parameters['conv3/W'],initial_bias=parameters['conv3/b']) 
                self.l1=L.Linear(in_size = None, out_size = self.hidden_dim, initialW=parameters['l1/W'], initial_bias=parameters['l1/b'])
                self.l2=L.Linear(self.hidden_dim, self.n_classes, initialW=parameters['l2/W'], initial_bias = parameters['l2/b'])

    #__call__はCNNが呼び出されるたびに実行される。xには文書データが入る                
    def __call__(self, x):
        with chainer.using_config('use_cudnn', self.cudnn):
            with chainer.using_config('cudnn_deterministic', True):
                h_non_static = F.dropout(self.lookup(x),0.25)
                h_non_static = F.reshape(h_non_static, (h_non_static.shape[0], 1, h_non_static.shape[1], h_non_static.shape[2]))
                #畳み込み
                h1 = self.conv1(h_non_static)
                h2 = self.conv2(h_non_static)
                h3 = self.conv3(h_non_static)
                #畳み込んだ結果をReLU関数で活性化。その後Max-Pooling
                h1 = F.max_pooling_2d(F.relu(h1), (h1.shape[2],h1.shape[3]))
                h2 = F.max_pooling_2d(F.relu(h2), (h2.shape[2],h2.shape[3]))
                h3 = F.max_pooling_2d(F.relu(h3), (h3.shape[2],h3.shape[3]))
                #3種類の畳み込み層で畳み込んだ結果をつなげる
                h = F.concat((h1,h2,h3),axis=2)
                #全結合層に通す(dropout付き)
                h = F.dropout(F.relu(self.l1(h)), ratio=0.5)
                #全結合層の通す
                y = self.l2(h)
                
        return y
#シード値の固定
def set_seed_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)
