# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: make_ft_rerank_corpus.py
# @time: 2024/6/19 11:31
# Python script for making ReRank model training dataset
import os
import json
from random import choices
import pandas as pd


project_dir = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]

with open(os.path.join(project_dir, 'data/ft_train_dataset.json')) as f:
    train_corpus = json.load(f)

rerank_train_data = {"query_id": [], "queries": [], "passages": [], "labels": []}
relevant_docs = train_corpus['relevant_docs']
corpus = train_corpus['corpus']
# each query has 1 relevant doc and 4 negative docs
for query_id, query in train_corpus['queries'].items():
    rerank_train_data["query_id"].append(query_id)
    rerank_train_data["queries"].append(query)
    relevant_passage_id = relevant_docs[query_id][0]
    rerank_train_data["passages"].append(corpus[relevant_passage_id])
    rerank_train_data["labels"].append(1)

    cnt = 0
    while cnt < 4:
        negative_id = choices(list(corpus.keys()), k=1)[0]
        if negative_id != relevant_passage_id:
            rerank_train_data["query_id"].append(query_id)
            rerank_train_data["queries"].append(query)
            rerank_train_data["passages"].append(corpus[negative_id])
            rerank_train_data["labels"].append(0)
            cnt += 1

df = pd.DataFrame(rerank_train_data)
df.to_csv(os.path.join(project_dir, 'data/ft_rerank_train.csv'), index=False)
