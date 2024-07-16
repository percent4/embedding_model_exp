# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: es_embedding_demo.py
# @time: 2024/7/16 10:05
import os
import json
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from src.utils.get_text_embedding import get_embedding

es_client = Elasticsearch(['http://localhost:9200'])
index_name = "dengyue"

if not es_client.indices.exists(index=index_name):
    # create index
    mapping = {
            "properties": {
                "content": {
                  "type": "text",
                  "fields": {
                    "keyword": {
                      "type": "keyword",
                      "ignore_above": 256
                    }
                  }
                },
                "content_id": {
                  "type": "integer"
                },
                "embedding": {
                  "type": "dense_vector",
                  "dims": 1536,
                  "index": True,
                  "similarity": "cosine"
                }
            }
    }
    es_client.indices.create(index=index_name, mappings=mapping)
    # read content and its openai embedding
    project_dir = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]
    corpus_embeddings = np.load(os.path.join(project_dir, "data/dengyue_embedding.npz"))['arr_0']
    print("corpus embeddings: ", corpus_embeddings.shape)
    with open(os.path.join(project_dir, 'data/dengyue.json'), 'r') as f:
        corpus_list = list(json.loads(f.read()).values())

    # insert data into ElasticSearch
    requests = []
    for i, doc in enumerate(corpus_list):
        request = {"_op_type": "index",
                   "_index": index_name,
                   "content_id": i + 1,
                   "content": doc,
                   "embedding": corpus_embeddings[i].tolist()}
        requests.append(request)

    bulk(es_client, requests)
    print("insert data into ElasticSearch successfully!")

# search data from ElasticSearch
query = "阿波罗登月计划"
# query = "神舟五号 杨利伟"
# query = "北京航天城"
query_embedding = get_embedding([query])[0]
result = es_client.search(index=index_name,
                          knn={
                              "field": "embedding",
                              "k": 3,
                              "num_candidates": 10,
                              "query_vector": query_embedding
                          })

for record in result['hits']['hits']:
    print(record['_source'])
