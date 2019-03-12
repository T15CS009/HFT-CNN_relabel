from gensim.models.wrappers.fasttext import FastText
from gensim.models import KeyedVectors
from tqdm import tqdm
from collections import defaultdict
import scipy.sparse as sp
import numpy as np
from  sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from itertools import chain
import re
import chakin
import os
import pdb
import tree
#

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
#データを整形する関数
def  make_data_list(data, kind_of_data, tree_info, max_sen_len, vocab, catgy, articleID, useWords):
    data_list = []#空のリスト
    for line in tqdm(data,desc="Loading " + kind_of_data + " data"):
        tmp_dict = dict()#空の辞書
        line = line[:-1]#改行文字を削除
        tmp_dict['text'] = ' '.join(clean_str(' '.join(line.split("\t")[1].split(" "))).split(" ")[:useWords])#タブ区切りでリストを作成
        [vocab[word] for word in tmp_dict['text'].split(" ")]#textをスペースで区切ったリストを作成
        tmp_dict['num_words'] = len(tmp_dict['text'].split(" "))#文書の単語数
        max_sen_len = max(max_sen_len, tmp_dict['num_words'])#最大単語長を調べる
        tmp_dict['split'] = kind_of_data
        tmp_dict['hie_info'] = list(set([tree_info[cat] for cat in line.split("\t")[0].split(",")]))#階層の情報
        tmp_dict['catgy'] = [cat for cat in line.split("\t")[0].split(",")]#カテゴリ名
        
        [catgy[cat] for cat in line.split("\t")[0].split(",")]#全カテゴリを保存するための処理
        tmp_dict['id'] = str(articleID) #記事の通し番号
        articleID += 1
        data_list.append(tmp_dict)#辞書をリストに追加
        del tmp_dict
    return data_list, max_sen_len, vocab, catgy, articleID

#ラベルを変更する場合のmake_data_list
def  make_data_list_relabel(data, kind_of_data, tree_info, max_sen_len, vocab, catgy, articleID, useWords,value,parent,relabel_flag):
    data_list = []
    change = []
    for line in tqdm(data,desc="Loading " + kind_of_data + " data"):
        tmp_dict = dict()
        line = line[:-1]
        tmp_dict['text'] = ' '.join(clean_str(' '.join(line.split("\t")[1].split(" "))).split(" ")[:useWords])
        [vocab[word] for word in tmp_dict['text'].split(" ")]
        tmp_dict['num_words'] = len(tmp_dict['text'].split(" "))
        max_sen_len = max(max_sen_len, tmp_dict['num_words'])
        tmp_dict['split'] = kind_of_data
        tmp_dict['catgy'] = [cat for cat in line.split("\t")[0].split(",")]
        #ラベルを変更
        if relabel_flag:
            
            flag2 = False
            if len(tmp_dict['catgy']) == 1:#カテゴリーがひとつしかない
                for s in tmp_dict['catgy']:
                    if s == str(parent):#親がいる
                        flag2 = True     
            if flag2: 
                tmp_dict["catgy"]=[str(value)]
                tmp_dict["change"]=['1']#ラベルの変更を行った場合'1'
            else:
                tmp_dict["change"]=['0']#ラベルの変更を行わなかった場合'0'
        
        tmp_dict['hie_info'] = list(set([tree_info[cat] for cat in tmp_dict['catgy']]))
        [catgy[cat] for cat in line.split("\t")[0].split(",")]
        tmp_dict['id'] = str(articleID)        
        articleID += 1
        data_list.append(tmp_dict)
        del tmp_dict
    return data_list, max_sen_len, vocab, catgy, articleID,change

#データ読み込みのメインの処理
def data_load(train, valid, test, tree_info, useWords):
    vocab = defaultdict( lambda: len(vocab) )#辞書内に存在しない語彙は追加される
    catgy = defaultdict( lambda: len(catgy) )#辞書内に存在しないカテゴリーは追加される
    articleID = 0
    max_sen_len = 0
   
    #各種データの読みこみ
    train_list, max_sen_len, vocab, catgy, articleID = make_data_list(train, 'train', tree_info, max_sen_len, vocab, catgy, articleID, useWords) 
    relabel_flag = False
    valid_list, max_sen_len, vocab, catgy, articleID = make_data_list(valid, 'valid', tree_info, max_sen_len, vocab, catgy, articleID, useWords) 
    test_list, max_sen_len, vocab, catgy, articleID = make_data_list(test, 'test', tree_info, max_sen_len, vocab, catgy, articleID, useWords) 
    #クラス数
    class_dim = len(catgy)
    #辞書のリストを更に辞書形式で保存
    data = {}
    data['train'] = train_list
    data['test'] = test_list
    data['valid'] = valid_list
    data['vocab'] = vocab
    data['catgy'] = catgy
    data['max_sen_len'] = max_sen_len
    data['class_dim'] = class_dim
    return data

#ラベルの変更を行うときのdata_load
def data_load_relabel(train, tree_info, useWords, value ,parent,relabel_flag):
    vocab = defaultdict( lambda: len(vocab) )
    catgy = defaultdict( lambda: len(catgy) )
    articleID = 0
    max_sen_len = 0
    re_value = value
    re_parent = parent
    #ラベル変更前のデータを保持
    train_list, max_sen_len, vocab, catgy, articleID ,changeflag = make_data_list_relabel(train, 'train', tree_info, max_sen_len, vocab, catgy, articleID, useWords,re_value,re_parent,relabel_flag) 
    relabel_flag = False
 
    class_dim = len(catgy)

    data = {}
    data['train'] = train_list
    data['catgy'] = catgy
    data['max_sen_len'] = max_sen_len
    data['class_dim'] = class_dim
    data['changeflag'] = changeflag#ラベルの変更を行ったかどうか
    return data

#分散表現を読み込む関数
def embedding_weights_load(words_map,embeddingWeights_path):
    pre_trained_embedding = None
    try:
        
        model = FastText.load_fasttext_format(embeddingWeights_path)#binファイルがある場合はそちらを読み込む
        pre_trained_embedding = "bin"
        
    except:
        print ("fastText binary file (.bin) is not found!")#ない場合はwikipediaの分散表現を使用する
        if os.path.exists("./Word_embedding/wiki.en.vec"):
            print ("Using wikipedia(en) pre-trained word vectors.")
        else:
            print ("Downloading wikipedia(en) pre-trained word vectors.")
            chakin.download(number=2, save_dir="./Word_embedding")
        print ("Loading vectors...")
        model =  KeyedVectors.load_word2vec_format('./Word_embedding/wiki.en.vec')
        pre_trained_embedding = "txt"

    vocab_size = len(words_map)
    word_dimension = model['a'].shape[0]#次元数を取得
    W = np.zeros((vocab_size,word_dimension),dtype=np.float32)#分散表現を格納するための行列 
    for k,v in words_map.items():#kには単語，vには単語ID
        word = k
        word_number = v
        #モデル中に存在しないチャンゴがある場合には、その単語の分散表現は乱数となる
        try:
                W[word_number][:] = model[word]
        except KeyError as e:
                if pre_trained_embedding == "bin":
                    W[word_number][:] = model.seeded_vector(word)
                else:
                    np.random.seed(word_number)
                    W[word_number][:] = np.random.uniform(-0.25, 0.25, word_dimension)
    return W
#ネットワークの出力結果をカテゴリ名に復元する関数
def get_catgy_mapping(network_output_order_list, test_labels, prediction,currentDepth):
    
    predictResult = []
    grandLabels = []
    
    for i in range(len(test_labels)):
        predictResult.append([])
        grandLabels.append([])

    class_dim = prediction.shape[1]

    row_idx, col_idx, val_idx = [], [], []
    for i in range(len(test_labels)):
        l_list = list(set(test_labels[i]))
        for y in l_list:
            row_idx.append(i)
            col_idx.append(y)
            val_idx.append(1)
    m = max(row_idx) + 1
    n = max(col_idx) + 1
    n = max(class_dim, n)
    test_labels = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n), dtype=np.int8).todense()

    np_orderList = np.array(network_output_order_list)

    for i,j in tqdm(enumerate(prediction), desc="Generating predict labels..."):
        one_hots = np.where(j == 1)[0]
        if len(one_hots) >= 1:
            predictResult[i] = np_orderList[one_hots].tolist()

    output_grand_truth_file_name = "CNN/RESULT/grand_truth_" + currentDepth + ".csv"
    with open(output_grand_truth_file_name, 'w') as f:
        f.write(','.join(network_output_order_list)+"\n")

    with open(output_grand_truth_file_name, 'a') as f:
        for i,j in tqdm(enumerate(test_labels), desc="Generating grand truth labels..."):
            one_hots = np.where(j == 1)[1]
            if len(one_hots) >= 1:
                grandLabels[i] = np_orderList[one_hots].tolist()
                f.write(",".join(grandLabels[i])+"\n")
            else:
                f.write("\n")

    return grandLabels,predictResult

#結果を書き出す関数
def write_out_prediction(GrandLabels, PredResult, input_data_dic,catname):

    # Writing out prediction
    # ===================================================
    print ("-"*50)
    print ("Writing out prediction...")
    test_data = input_data_dic['test']
    result_file = open("./CNN/RESULT/Prediction_"+catname+".txt", mode="w")
    result_file.write("Grand-truth-label\tPrediction-labels\tInput-text\n")
    for g,p,t in zip(GrandLabels, PredResult, test_data):
        result_file.write("{}\t{}\t{}\n".format(','.join(sorted(g)), ','.join(sorted(p)), t['text']))
    result_file.close()

# Making Problems
#========================================================
# テキストデータをndarrayに変換する関数
def build_input_sentence_data(sentences):
    x = np.array(sentences)
    return x

#ラベルを整形する関数    
def build_input_label_data(labels, class_order):
    from sklearn.preprocessing import MultiLabelBinarizer
    from itertools import chain

    bml = MultiLabelBinarizer(classes=class_order, sparse_output=True)
    indexes = sp.find(bml.fit_transform(labels)) 
    Y = []

    for i in range(len(labels)):
        Y.append([])
    for i,j in zip(indexes[0], indexes[1]):
        Y[i].append(j)
    return Y

#最大単語長を満たしていない文書を-1によってパディングする関数
def pad_sentences(sentences, padding_word=-1, max_length=50):
    sequence_length = max(max(len(x) for x in sentences), max_length)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < max_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:max_length]
        padded_sentences.append(new_sentence)
    return padded_sentences

#データを組み立てるメイン処理
def build_problem(learning_categories, depth, input_data_dic): #flag
    train_data = input_data_dic['train']
    
    validation_data = input_data_dic['valid']
    test_data = input_data_dic['test']
    vocab = input_data_dic['vocab']
    max_sen_len = input_data_dic['max_sen_len']
    if depth == "flat":
        trn_text = [[vocab[word] for word in doc['text'].split()] for doc in train_data]
        trn_labels = [doc['catgy'] for doc in train_data]
        val_text = [[vocab[word] for word in doc['text'].split()] for doc in validation_data]
        val_labels = [doc['catgy'] for doc in validation_data]
        tst_text = [[vocab[word] for word in doc['text'].split()] for doc in test_data]
        tst_labels = [doc['catgy'] for doc in test_data]
       
    else:
        layer = int(depth[:-2])
        origin_trn_labels = [list( set(doc['catgy']) & set(learning_categories)) for doc in train_data if (layer in doc['hie_info']) or ((layer-1) in doc['hie_info'])]
        trn_text = [[vocab[word] for word in doc['text'].split()] for doc in train_data if (layer in doc['hie_info']) or ((layer-1) in doc['hie_info'])]
        trn_labels = [list( set(doc['catgy']) & set(learning_categories)) for doc in train_data if (layer in doc['hie_info']) or ((layer-1) in doc['hie_info'])]
        #pdb.set_trace()
        val_text = [[vocab[word] for word in doc['text'].split()] for doc in validation_data if (layer in doc['hie_info']) or ((layer-1) in doc['hie_info'])]
        val_labels = [list( set(doc['catgy']) & set(learning_categories)) for doc in validation_data if (layer in doc['hie_info']) or ((layer-1) in doc['hie_info'])]
        tst_text = [[vocab[word] for word in doc['text'].split()] for doc in test_data]
        tst_labels = [list( set(doc['catgy']) & set(learning_categories)) if layer in doc['hie_info'] else [] for doc in test_data]
    
    trn_padded = pad_sentences(trn_text, max_length=max_sen_len)
    val_padded = pad_sentences(val_text, max_length=max_sen_len)
    tst_padded = pad_sentences(tst_text, max_length=max_sen_len)
    X_trn = build_input_sentence_data(trn_padded)
    X_val = build_input_sentence_data(val_padded)
    X_tst = build_input_sentence_data(tst_padded)
    Y_trn = build_input_label_data(trn_labels,learning_categories)
    Y_val = build_input_label_data(val_labels, learning_categories)
    Y_tst = build_input_label_data(tst_labels, learning_categories)
    # pdb.set_trace()
    return X_trn, Y_trn, X_val, Y_val, X_tst, Y_tst, origin_trn_labels, 
    # oriori_trn_labels

#ラベルを変更する場合のbuild_problem
def build_problem_relabel(learning_categories, depth, input_data_dic,input_data_ori_dic): #flag
    train_data = input_data_ori_dic['train']
    validation_data = input_data_dic['valid']
    test_data = input_data_dic['test']
    vocab = input_data_dic['vocab']
    max_sen_len = input_data_dic['max_sen_len']
    changeflag = input_data_ori_dic['changeflag']


    if depth == "flat":
        trn_text = [[vocab[word] for word in doc['text'].split()] for doc in train_data]
        trn_labels = [doc['catgy'] for doc in train_data]
        val_text = [[vocab[word] for word in doc['text'].split()] for doc in validation_data]
        val_labels = [doc['catgy'] for doc in validation_data]
        tst_text = [[vocab[word] for word in doc['text'].split()] for doc in test_data]
        tst_labels = [doc['catgy'] for doc in test_data]
    
    else:
        layer = int(depth[:-2])
        changeflag_id = []
        changeflag_id =  [list( set(doc['change'])) for doc in train_data if (layer in doc['hie_info']) or ((layer-1) in doc['hie_info'])]
        origin_trn_labels = [list( set(doc['catgy']) & set(learning_categories)) for doc in train_data if (layer in doc['hie_info']) or ((layer-1) in doc['hie_info'])]
        trn_text = [[vocab[word] for word in doc['text'].split()] for doc in train_data if (layer in doc['hie_info']) or ((layer-1) in doc['hie_info'])]
        trn_labels = [list( set(doc['catgy']) & set(learning_categories)) for doc in train_data if (layer in doc['hie_info']) or ((layer-1) in doc['hie_info'])]
        val_text = [[vocab[word] for word in doc['text'].split()] for doc in validation_data if (layer in doc['hie_info']) or ((layer-1) in doc['hie_info'])]
        val_labels = [list( set(doc['catgy']) & set(learning_categories)) for doc in validation_data if (layer in doc['hie_info']) or ((layer-1) in doc['hie_info'])]
        tst_text = [[vocab[word] for word in doc['text'].split()] for doc in test_data]
        tst_labels = [list( set(doc['catgy']) & set(learning_categories)) if layer in doc['hie_info'] else [] for doc in test_data]
    
    trn_padded = pad_sentences(trn_text, max_length=max_sen_len)
    val_padded = pad_sentences(val_text, max_length=max_sen_len)
    tst_padded = pad_sentences(tst_text, max_length=max_sen_len)
    X_trn = build_input_sentence_data(trn_padded)
    X_val = build_input_sentence_data(val_padded)
    X_tst = build_input_sentence_data(tst_padded)
    Y_trn = build_input_label_data(trn_labels,learning_categories)
    Y_val = build_input_label_data(val_labels, learning_categories)
    Y_tst = build_input_label_data(tst_labels, learning_categories)
    # pdb.set_trace()
    return X_trn, Y_trn, X_val, Y_val, X_tst, Y_tst, origin_trn_labels, changeflag_id

#層の深さを取得する関数
def order_n(i): return {1:"1st", 2:"2nd", 3:"3rd"}.get(i) or "%dth"%i


#fスコアを計算する関数
def write_out_f_score_relabel(labels, Y_true, Y_pred,dep,value):
    mlb = MultiLabelBinarizer(labels)
    Y_true = mlb.fit_transform(Y_true)
    Y_pred = mlb.fit_transform(Y_pred)
    with open("./CNN/RESULT/f_score.txt", mode="a") as result_file:
        result_file.write("-------"+ dep +" layer result and " + value +" category relaveled --------\n")
        result_file.write("macro_average : " + str(f1_score(Y_true, Y_pred, average='macro')) + "\n")
        result_file.write("micro_average : " + str(f1_score(Y_true, Y_pred, average='micro')) + "\n\n")
        result_file.write(" ---------- class ---------- \n" + str("\n".join([str(i) + '\t' + str(j)for i ,j in zip(labels, f1_score(Y_true, Y_pred, average=None))])))
        result_file.write("\n")



def write_out_f_score(labels, Y_true, Y_pred,dep):
    mlb = MultiLabelBinarizer(labels)
    Y_true = mlb.fit_transform(Y_true)
    Y_pred = mlb.fit_transform(Y_pred)
    with open("./CNN/RESULT/f_score.txt", mode="a") as result_file:
        result_file.write("-------"+ dep +" layer result  --------\n")
        result_file.write("macro_average : " + str(f1_score(Y_true, Y_pred, average='macro')) + "\n")
        result_file.write("micro_average : " + str(f1_score(Y_true, Y_pred, average='micro')) + "\n\n")
        result_file.write(" ---------- class ---------- \n" + str("\n".join([str(i) + '\t' + str(j)for i ,j in zip(labels, f1_score(Y_true, Y_pred, average=None))])))
        result_file.write("\n")

