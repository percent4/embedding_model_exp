# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: datasets_tranfer.py
# @time: 2024/6/6 23:01
# load train dataset
import os
import json
from datasets import Dataset

project_dir = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]

# load dataset
with open(os.path.join(project_dir, "data/ft_val_dataset.json"), "r", encoding="utf-8") as f:
    train_content = json.loads(f.read())

train_anchor, train_positive = [], []
for query_id, context_id in train_content['relevant_docs'].items():
    train_anchor.append(train_content['queries'][query_id])
    train_positive.append(train_content['corpus'][context_id[0]])

train_dataset = Dataset.from_dict({"positive": train_positive, "anchor": train_anchor})

train_dataset.to_json(os.path.join(project_dir, "data/auto_val_dataset.json"))
