# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: ft_rerank.py
# @time: 2024/6/19 11:30
# Python script for ReRank model fine-tuning using Sentence Transformers
import os
import logging
import pandas as pd

from torch.utils.data import DataLoader

from sentence_transformers import InputExample, LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

# logger
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

# First, we define the transformer model we want to fine-tune
model_path = "/workspace/models/bge-reranker-large"
train_batch_size = 8
num_epochs = 5
model_save_path = "ft_" + os.path.basename(model_path)


# We set num_labels=1, which predicts a continuous score between 0 and 1
model = CrossEncoder(model_path, num_labels=1, max_length=512)


# Prepare datasets for model training and evaluation
train_samples = []
dev_samples = {}

project_dir = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]
train_df = pd.read_csv(os.path.join(project_dir, "data/ft_rerank_train.csv"))
print(train_df.shape)
for i, row in train_df.iterrows():
    train_samples.append(InputExample(texts=[row["queries"], row["passages"]], label=row["labels"]))

val_df = pd.read_csv(os.path.join(project_dir, "data/ft_rerank_val.csv"))
for i, row in val_df.iterrows():
    query_id = row["query_id"]
    if query_id not in dev_samples:
        dev_samples[query_id] = {"query": row["queries"], "positive": set(), "negative": set()}
    
    if row["labels"]:
        dev_samples[query_id]["positive"].add(row["passages"])
    else:
        dev_samples[query_id]["negative"].add(row["passages"])

# We create a DataLoader to load our train samples
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# We add an evaluator, which evaluates the performance during training
# It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
evaluator = CERerankingEvaluator(dev_samples, name="train-eval")

# Configure the training
warmup_steps = 100
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=100,
    optimizer_params={'lr': 1e-5},
    warmup_steps=warmup_steps,
    output_path=model_save_path,
    use_amp=True
)

# Save the model
model.save(model_save_path)
