# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: bge_base_zh_eval.py
# @time: 2024/6/6 16:32
import os
import json
import torch
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
model_path = os.path.join(project_dir, "models/ft_bge-base-zh-v1.5")
model = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded")

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
print(result)

"""
{'bge-base-zh-v1.5_cosine_accuracy@1': 0.6043613707165109,
 'bge-base-zh-v1.5_cosine_accuracy@10': 0.881619937694704,
 'bge-base-zh-v1.5_cosine_accuracy@3': 0.7538940809968847,
 'bge-base-zh-v1.5_cosine_accuracy@5': 0.8099688473520249,
 'bge-base-zh-v1.5_cosine_map@100': 0.6998100636043242,
 'bge-base-zh-v1.5_cosine_mrr@10': 0.694540869307224,
 'bge-base-zh-v1.5_cosine_ndcg@10': 0.7396433891138803,
 'bge-base-zh-v1.5_cosine_precision@1': 0.6043613707165109,
 'bge-base-zh-v1.5_cosine_precision@10': 0.0881619937694704,
 'bge-base-zh-v1.5_cosine_precision@3': 0.25129802699896153,
 'bge-base-zh-v1.5_cosine_precision@5': 0.16199376947040497,
 'bge-base-zh-v1.5_cosine_recall@1': 0.6043613707165109,
 'bge-base-zh-v1.5_cosine_recall@10': 0.881619937694704,
 'bge-base-zh-v1.5_cosine_recall@3': 0.7538940809968847,
 'bge-base-zh-v1.5_cosine_recall@5': 0.8099688473520249}
"""