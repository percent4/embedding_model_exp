# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: bge_base_zh_eval.py
# @time: 2024/6/6 16:32
import os
import json
import time
import torch
from pprint import pprint
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim

project_dir = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]

# data process
# load dataset, get corpus, queries, relevant_docs
with open(os.path.join(project_dir, "data/doc_qa.json"), "r", encoding="utf-8") as f:
    content = json.loads(f.read())

corpus = content['corpus']
queries = content['queries']
relevant_docs = content['relevant_docs']

# # Load a model
# 替换成自己的模型完整路径或使用huggingface modl id
model_name = "bge-base-zh-v1.5"
model_path = os.path.join(project_dir, f"models/{model_name}")
model = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded")

s_time = time.time()

# # Evaluate the model
evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name=f"{os.path.basename(model_path)}",
    score_functions={"cosine": cos_sim}
)

# Evaluate the model
result = evaluator(model)
pprint(result)
print(f"Time cost: {time.time() - s_time:.2f}s")
