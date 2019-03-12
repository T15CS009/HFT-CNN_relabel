#!/usr/bin/env python
# -* coding: utf-8 -*-

#new new----------------------
import sys, random
random.seed(0)

with open(sys.argv[1], 'r') as input_data:
    #datalist = [i for i in input_data]
    datalist = input_data.readlines()

    percent = int(len(datalist)*0.25)
    #ran = random.sample(datalist, percent)
    random.shuffle(datalist)
 
    #valid = [i for i in datalist if i in ran]
    #train = [i for i in datalist if(not i in ran)]
    
    valid = datalist[:percent]
    train = datalist[percent:]

    with open('valid.out', 'w') as valid_out:
        valid_out.writelines(valid)
    with open('train.out', 'w') as train_out:
        train_out.writelines(train)
