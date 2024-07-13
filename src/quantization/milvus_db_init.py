# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: milvus_db_init.py
# @time: 2024/7/11 11:30
import json
import os
import time
import numpy as np
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType

project_dir = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]
corpus_embeddings = np.load(os.path.join(project_dir, "data/corpus_openai_embedding.npy"))
print(corpus_embeddings.shape)
with open(os.path.join(project_dir, 'data/doc_qa_test.json'), 'r', encoding='utf-8') as f:
    corpus_list = list(json.loads(f.read())['corpus'].values())

collection_name_list = ["semi_conductor", "semi_conductor_sq8", "semi_conductor_bin"]
dtype_list = [DataType.FLOAT_VECTOR, DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]
schema_list = [{"index_type": "IVF_FLAT", "metric_type": "IP"},
               {"index_type": "IVF_SQ8", "metric_type": "L2"},
               {"index_type": "BIN_FLAT", "metric_type": "JACCARD"}
               ]

# Connects to a server
client = MilvusClient(uri="http://localhost:19530", db_name="default")
# List all collection names
collections = client.list_collections()
print("exist collections: ", collections)

for i, collection_name in enumerate(collection_name_list):
    # create collection if not exists
    if collection_name not in collections:
        # Creates a collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="embedding", dtype=dtype_list[i], dim=1536)
        ]
        schema = CollectionSchema(fields, "text search demo")
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type=schema_list[i]["index_type"],
            metric_type=schema_list[i]["metric_type"],
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
        if i < 2:
            entities = [
                {"id": j + 1,
                 "text": corpus_list[j],
                 "embedding": corpus_embeddings[j]}
                for j in range(corpus_embeddings.shape[0])
            ]
        else:
            entities = [
                {"id": j + 1,
                 "text": corpus_list[j],
                 "embedding": np.packbits(np.where(corpus_embeddings[j] > 0, 1, 0)).tobytes()}
                for j in range(corpus_embeddings.shape[0])
            ]
        client.insert(collection_name=collection_name, data=entities)
