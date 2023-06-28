# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:49:45 2023

@author: shala
"""

import json
import os

os.chdir(r"C:\Users\shala\Downloads")
test = json.load(open("dev-v2.json"))

pred = json.load(open("pred.json"))

from allennlp_models import pretrained
model = pretrained.load_predictor("rc-bidaf")

pred_final = {}
n = 1

for d in test['data'][1:]:
    paragraphs = d['paragraphs']
    for d1 in paragraphs:
        context = d1['context']
        questions_l = d1['qas']
        print("questions in dictionary no. {} are ".format(n),len(questions_l))
        for d2 in questions_l:
            question = d2['question']
            ids = d2['id']
            
            answer = model.predict(question,context)['best_span_str']
            
            pred_final[ids]=answer 
            
    n +=1
    print("dict {} over".format(n) )
    
json.dump(pred_final,open("pred2.json",'w+'))