# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: make_ft_embedding_corpus.py
# @time: 2024/6/6 16:56
import os
from llama_index.legacy.finetuning import (
    generate_qa_embedding_pairs
)
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

project_dir = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]

TRAIN_FILES = [os.path.join(project_dir, "data/ft_train.txt")]
VAL_FILES = [os.path.join(project_dir, "data/ft_test.txt")]

TRAIN_CORPUS_FPATH = os.path.join(project_dir, "data/ft_train_corpus.json")
VAL_CORPUS_FPATH = os.path.join(project_dir, "data/ft_val_corpus.json")


def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter(chunk_size=250, chunk_overlap=0)
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes


train_nodes = load_corpus(TRAIN_FILES, verbose=True)
val_nodes = load_corpus(VAL_FILES, verbose=True)

llm = OpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

qa_generate_prompt_tmpl = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination in Chinese. The questions should be diverse in nature \
across the document in Chinese. The questions should not contain options, not start with Q1/ Q2. \
Restrict the questions to the context information provided.
"""

train_dataset = generate_qa_embedding_pairs(nodes=train_nodes, llm=llm, num_questions_per_chunk=1, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl)
val_dataset = generate_qa_embedding_pairs(nodes=val_nodes, llm=llm, num_questions_per_chunk=1, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl)

train_dataset.save_json(TRAIN_CORPUS_FPATH)
val_dataset.save_json(VAL_CORPUS_FPATH)

"""
Output:

Loading files ['/Users/admin/PycharmProjects/embedding_model_exp/data/ft_train.txt']
Loaded 1 docs
Parsing nodes: 100%|██████████| 1/1 [00:00<00:00, 23.54it/s]
Parsing nodes:   0%|          | 0/1 [00:00<?, ?it/s]Parsed 137 nodes
Loading files ['/Users/admin/PycharmProjects/embedding_model_exp/data/ft_test.txt']
Loaded 1 docs
Parsing nodes: 100%|██████████| 1/1 [00:00<00:00, 45.84it/s]
  0%|          | 0/137 [00:00<?, ?it/s]Parsed 111 nodes
100%|██████████| 137/137 [03:34<00:00,  1.57s/it]
100%|██████████| 111/111 [01:55<00:00,  1.04s/it]
"""
