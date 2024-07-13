# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: milvus_embed_quantized_eval.py
# @time: 2024/7/11 16:19
import os
import time
import json
import asyncio
import numpy as np
import pandas as pd
from typing import List
from datetime import datetime
from pymilvus import MilvusClient
from llama_index.core import QueryBundle
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.indices.query.schema import QueryType
from llama_index.core.evaluation import RetrieverEvaluator, EmbeddingQAFinetuneDataset


# display result in pandas format
def display_results(name_list, eval_results_list):
    pd.set_option('display.precision', 4)
    columns = {"retrievers": [], "hit_rate": [], "mrr": []}
    for name, eval_results in zip(name_list, eval_results_list):
        metric_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)

        full_df = pd.DataFrame(metric_dicts)

        hit_rate = full_df["hit_rate"].mean()
        mrr = full_df["mrr"].mean()

        columns["retrievers"].append(name)
        columns["hit_rate"].append(hit_rate)
        columns["mrr"].append(mrr)

    metric_df = pd.DataFrame(columns)

    return metric_df


# Embedding file
project_dir = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]
query_embeddings = np.load(os.path.join(project_dir, "data/queries_openai_embedding.npy"))
with open(os.path.join(project_dir, "data/doc_qa_test.json"), "r", encoding="utf-8") as f:
    content = json.loads(f.read())
queries = list(content["queries"].values())
query_id_dict = {v: k for k, v in content["queries"].items()}
corpus_id_dict = {v: k for k, v in content["corpus"].items()}
corpus = list(content["corpus"].values())
query_embedding_dict = {}
for i in range(len(queries)):
    query_embedding_dict[queries[i]] = query_embeddings[i]


# Milvus Config
client = MilvusClient(uri="http://localhost:19530", db_name="default")
collection_name_list = ["semi_conductor", "semi_conductor_sq8", "semi_conductor_bin"]
schema_list = [{"index_type": "IVF_FLAT", "metric_type": "IP"},
               {"index_type": "IVF_SQ8", "metric_type": "L2"},
               {"index_type": "BIN_FLAT", "metric_type": "JACCARD"}
               ]


# Embedding Retriever
class EmbeddingRetriever(BaseRetriever):
    def __init__(self, top_k) -> None:
        super().__init__()
        self.top_k = top_k

    def _retrieve(self, query: QueryType) -> List[NodeWithScore]:
        query = QueryBundle(query) if isinstance(query, str) else query
        # vector search using Milvus
        query_embedding = query_embedding_dict[query.query_str].tolist()
        res = client.search(
            collection_name="semi_conductor",
            data=[query_embedding],
            limit=self.top_k,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=['text']
        )

        result = []
        for data in res[0]:
            text = data['entity']['text']
            node_with_score = NodeWithScore(
                node=TextNode(
                    text=text,
                    id_=corpus_id_dict[text],
                    score=data['distance']
                )
            )
            result.append(node_with_score)

        return result


# Embedding SQ8 Retriever
class EmbeddingSQ8Retriever(BaseRetriever):
    def __init__(self, top_k) -> None:
        super().__init__()
        self.top_k = top_k

    def _retrieve(self, query: QueryType) -> List[NodeWithScore]:
        query = QueryBundle(query) if isinstance(query, str) else query
        # vector search using Milvus
        query_embedding = query_embedding_dict[query.query_str].tolist()
        res = client.search(
            collection_name="semi_conductor_sq8",
            data=[query_embedding],
            limit=self.top_k,
            search_params={"metric_type": "L2", "params": {"nprobe": 8}},
            output_fields=['text']
        )

        result = []
        for data in res[0]:
            text = data['entity']['text']
            node_with_score = NodeWithScore(
                node=TextNode(
                    text=text,
                    id_=corpus_id_dict[text]
                ),
                score=data['distance']
            )
            result.append(node_with_score)

        return result


# binary embedding retriever
class EmbeddingBinRetriever(BaseRetriever):
    def __init__(self, top_k) -> None:
        super().__init__()
        self.top_k = top_k

    def _retrieve(self, query: QueryType) -> List[NodeWithScore]:
        query = QueryBundle(query) if isinstance(query, str) else query
        # vector search using Milvus
        query_embedding = np.packbits(np.where(np.array(query_embedding_dict[query.query_str]) > 0, 1, 0)).tobytes()
        res = client.search(
            collection_name="semi_conductor_bin",
            data=[query_embedding],
            limit=self.top_k,
            search_params={"metric_type": "JACCARD", "params": {}},
            output_fields=['text']
        )

        result = []
        for data in res[0]:
            text = data['entity']['text']
            node_with_score = NodeWithScore(
                node=TextNode(
                    text=text,
                    id_=corpus_id_dict[text],
                    score=data['distance']
                )
            )
            result.append(node_with_score)

        return result


if __name__ == '__main__':
    # dict: key: embedding name, value: its retriever
    embedding_name = "embedding_bin"
    embedding_retriever_dict = {"embedding": EmbeddingRetriever,
                                "embedding_sq8": EmbeddingSQ8Retriever,
                                "embedding_bin": EmbeddingBinRetriever}
    custom_embedding_retriever = embedding_retriever_dict[embedding_name]

    # evaluation experiment
    doc_qa_dataset = EmbeddingQAFinetuneDataset.from_json(os.path.join(project_dir, "data/doc_qa_test.json"))
    metrics = ["mrr", "hit_rate"]
    evaluation_name_list = []
    evaluation_result_list = []
    cost_time_list = []
    for top_n in [3, 5, 10]:
        start_time = time.time()
        embedding_retriever = custom_embedding_retriever(top_k=top_n)
        embedding_retriever_evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=embedding_retriever)
        embedding_eval_results = asyncio.run(embedding_retriever_evaluator.aevaluate_dataset(doc_qa_dataset,
                                                                                             show_progress=True))
        evaluation_name_list.append(f"{embedding_name}_top_{top_n}_eval")
        evaluation_result_list.append(embedding_eval_results)
        cost_time_list.append((time.time() - start_time) * 1000)
        print(f"done for top_{top_n} {embedding_name} evaluation!")

    df = display_results(evaluation_name_list, evaluation_result_list)
    df['cost_time'] = cost_time_list
    print(df.head())
    csv_save_path = os.path.join(project_dir,
                                 f"data/eval_{embedding_name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.csv")
    df.to_csv(csv_save_path, encoding="utf-8", index=False)
