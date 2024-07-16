# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: simple_embed_sq8.py
# @time: 2024/7/11 10:03
# simple scalar embedding quantization using Milvus
# reference: https://milvus.io/docs
import os
import json
import time
import numpy as np
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType

from src.utils.get_text_embedding import get_embedding

project_dir = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]

with open(os.path.join(project_dir, 'data/dengyue.json'), 'r') as f:
    sentences = list(json.loads(f.read()).values())

# add vector
sentences_embeddings = np.load(os.path.join(project_dir, "data/dengyue_embedding.npz"))['arr_0']

collection_name = "dengyue"
# Connects to a server
client = MilvusClient(uri="http://localhost:19530", db_name="default")
# List all collection names
collections = client.list_collections()
print("exist collections: ", collections)

# create collection if not exists
if collection_name not in collections:
    # Creates a collection
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
    ]
    schema = CollectionSchema(fields, "text search demo")
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_SQ8",
        metric_type="L2",
        params={"nlist": 128}
    )
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    time.sleep(3)
    res = client.get_load_state(
        collection_name=collection_name
    )
    print("load state: ", res)

    # Inserts vectors in the collection
    entities = [
        {"id": i + 1,
         "text": sentences[i],
         "embedding": sentences_embeddings[i].tolist()}
        for i in range(len(sentences))
    ]
    client.insert(collection_name=collection_name, data=entities)

# Single vector search
query = "阿波罗登月计划"
# query = "神舟五号 杨利伟"
# query = "北京航天城"
query_embedding = get_embedding([query])

start_time = time.time()
res = client.search(
    collection_name=collection_name,
    data=query_embedding,
    limit=3,
    search_params={"metric_type": "IP", "params": {}},
    output_fields=['text']
)
print(len(res[0]), res[0][0])
print(f"cost time: {(time.time() - start_time) * 1000:.2f} ms")

# Convert the output to a formatted JSON string
result = json.dumps(res, indent=4, ensure_ascii=False)
print(result)

