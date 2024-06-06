# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: datasets_push.py
# @time: 2024/6/6 23:20
import os
from datasets import load_dataset

project_dir = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]

data_files = {"train": f"{project_dir}/data/auto_train_dataset.json", "dev": f"{project_dir}/data/auto_val_dataset.json"}
raw_dataset = load_dataset("json", data_files=data_files)

print(raw_dataset)
raw_dataset.push_to_hub("jclian91/embedding_exp_semiconductor")

"""
huggingface-cli login
#  using write access token
"""