# -*- coding: utf-8 -*-
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

with open("topics.txt", 'r') as topics_txt:
    with open("Prediction.txt", 'r') as prediction_txt:
        topics_txt_list = topics_txt.readlines()
        prediction_txt_list = prediction_txt.readlines()
        topics_txt_list = [i.replace("\n", "") for i in topics_txt_list]
        prediction_txt_list = [i.replace("\t\t", "\tNone\t").split('\t')for i in prediction_txt_list][1:]
        
        topics_txt_list.append("None")

        g_t_l = [i[0].split(",") for i in prediction_txt_list]
        p_l = [i[1].split(",") for i in prediction_txt_list]

        mlb = MultiLabelBinarizer(topics_txt_list)
        # g_t_l = [i[:-1]for i in mlb.fit_transform(g_t_l)]
        # p_l = [i[:-1]for i in mlb.fit_transform(p_l)]

        # for i, j in zip(g_t_l, p_l):
        #     print('macro : ' + str(f1_score(i, j, average='macro')))
        #     print('micro : ' + str(f1_score(i, j, average='micro')))

        print('macro :' + str(f1_score(mlb.fit_transform(g_t_l), mlb.fit_transform(p_l), average='macro')))
        print('micro :' + str(f1_score(mlb.fit_transform(g_t_l), mlb.fit_transform(p_l), average='micro')))


        # print(mean([f1_score(i, j, average='macro')for i, j in zip(g_t_l, p_l)]))
