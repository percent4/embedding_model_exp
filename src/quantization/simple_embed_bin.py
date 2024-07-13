# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: simple_embed_bin.py
# @time: 2024/7/11 10:28
# simple binary embedding quantization using Milvus
import os
import json
import time
import numpy as np
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, Collection

from src.utils.get_text_embedding import get_embedding

project_dir = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]

with open(os.path.join(project_dir, 'data/dengyue.json'), 'r') as f:
    sentences = list(json.loads(f.read()).values())

# add vector
sentences_embeddings = np.load(os.path.join(project_dir, "data/dengyue_embedding.npz"))['arr_0']

collection_name = "dengyue_bin"
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
        FieldSchema(name="embedding_bin", dtype=DataType.BINARY_VECTOR, dim=1536)
    ]
    schema = CollectionSchema(fields, "Embedding quantization demo")
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding_bin",
        index_type="BIN_FLAT",
        metric_type="JACCARD",
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
    for i in range(len(sentences)):
        entities = {
            "id": i + 1,
            "text": sentences[i],
            "embedding_bin": np.packbits(np.where(sentences_embeddings[i] > 0, 1, 0)).tobytes()
        }
        client.insert(collection_name=collection_name, data=entities)

# Single vector search
# query = "阿波罗登月计划"
# query = "神舟五号 杨利伟"
query = "北京航天城"
query_embedding = get_embedding([query])
embedding_bin = np.packbits(np.where(np.array(query_embedding[0]) > 0, 1, 0)).tobytes()

start_time = time.time()
res = client.search(
    collection_name=collection_name,
    data=[embedding_bin],
    limit=3,
    search_params={"metric_type": "JACCARD", "params": {"nprobe": 128}},
    output_fields=['text']
)
print(f"cost time: {(time.time() - start_time) * 1000:.2f} ms")

# Convert the output to a formatted JSON string
result = json.dumps(res, indent=4, ensure_ascii=False)
print(result)

