#HFT-CNN
#
import six
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, training, reporter
from chainer.datasets import get_mnist
from chainer.training import trainer, extensions
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer.datasets import get_mnist
from chainer import optimizer as optimizer_module
import scipy.sparse as sp
import pdb
import math
from itertools import zip_longest

class MyUpdater(training.StandardUpdater):
    def __init__(self, iterator, optimizer, class_dim, converter=convert.concat_examples,
                device=None, loss_func=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if not isinstance(optimizer, dict):
            optimizer = {'main': optimizer}
        self._optimizers = optimizer

        if device is not None and device >= 0:
            for optimizer in six.itervalues(self._optimizers):
                optimizer.target.to_gpu(device)

        self.converter = converter
        self.loss_func = loss_func
        self.device = device
        self.iteration = 0
        self.class_dim = class_dim
        
    def update_core(self):
        batch = self._iterators['main'].next()
        x = chainer.cuda.to_gpu(np.array([i[0] for i in batch]))
        

        labels = [l[1] for l in batch] 
        #疎行列を作成
        row_idx, col_idx, val_idx = [], [], []
        for i in range(len(labels)):#行のループ
            l_list = list(set(labels[i])) # 重複するラベルがないように一旦集合にしてからリストに戻す
            for y in l_list:#列のループ
                row_idx.append(i)#行の位置
                col_idx.append(y)#列の位置
                val_idx.append(1)#値(1)
        #m×nの行列を作る
        m = len(labels)
        n = self.class_dim
        t = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n), dtype=np.int8).todense()#データを密行列に変換
        

        t = chainer.cuda.to_gpu(t) #GPUに変換
        
        optimizer = self._optimizers['main']
        y = optimizer.target(x)
        optimizer.target.cleargrads()
        
        #誤差を計算 
        loss = F.sigmoid_cross_entropy(y, t)
        
        chainer.reporter.report({'main/loss':loss})#lossを書き出す
        loss.backward()#逆伝播
        optimizer.update()#パラメータの更新

#ラベルの変更をするときのMyUpdater
class MyUpdater_relabel(training.StandardUpdater):
    def __init__(self, iterator, optimizer, class_dim, value, converter=convert.concat_examples,
                device=None, loss_func=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if not isinstance(optimizer, dict):
            optimizer = {'main': optimizer}
        self._optimizers = optimizer

        if device is not None and device >= 0:
            for optimizer in six.itervalues(self._optimizers):
                optimizer.target.to_gpu(device)
        self.value = value
        self.converter = converter
        self.loss_func = loss_func
        self.device = device
        self.iteration = 0
        self.class_dim = class_dim
        
    def update_core(self):
        batch = self._iterators['main'].next()
        l_batch_origin = [i[2] for i in batch]
        changeflag = [i[3] for i in batch]

        x = chainer.cuda.to_gpu(np.array([i[0] for i in batch]))
        value = self.value

        labels = [l[1] for l in batch] 
        row_idx, col_idx, val_idx = [], [], []
        for i in range(len(labels)):
            l_list = list(set(labels[i])) # remove duplicate cateories to avoid double count
            for y in l_list:
                row_idx.append(i)
                col_idx.append(y)
                val_idx.append(1)
        m = len(labels)
        n = self.class_dim
        t = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n), dtype=np.int8).todense()
        

        t = chainer.cuda.to_gpu(t)
        
        optimizer = self._optimizers['main']
        y = optimizer.target(x)
        loss = F.sigmoid_cross_entropy(y, t)
        flag=1
        #それぞれのカテゴリーに対して誤差を計算する
        for i1,i2,i3,i4 in zip_longest(t,y,l_batch_origin,changeflag):
            if changeflag == '1':  
                for j1,j2 in zip_longest(i1,i2):
                    myloss = F.sigmoid_cross_entropy(j2,j1) + (j1*(F.log(F.sigmoid(j2))))#ラベルの変更がされている場合のlossの計算
                    if math.isnan(myloss.data):
                    
                        myloss = F.sigmoid_cross_entropy(j2,j1)+(j1*(F.log1p(F.sigmoid(j2))))
                    
                    if flag == 1:
                        all_loss = myloss
                        flag = 0
                        
                    else:
                        all_loss += myloss
            else:
                for j1,j2 in zip_longest(i1,i2):
                    myloss = F.sigmoid_cross_entropy(j2,j1) #ラベルの変更がされていない場合のlossの計算
                    
                    if flag == 1:
                        all_loss = myloss
                        flag = 0
                        
                    else:
                        all_loss += myloss
        all_loss = all_loss/t.size
        optimizer.target.cleargrads()
        all_loss.backward()#逆伝播
        optimizer.update()#パラメータの更新
        chainer.reporter.report({'main/loss':all_loss})#lossの書き出し
