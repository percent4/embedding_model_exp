# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: ft_sentence_transformers_trainer.py
# @time: 2024/6/6 17:54
import os
import json
import time
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformerTrainer

start_time = time.time()
project_dir = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]

# load eval dataset
with open(os.path.join(project_dir, "data/ft_val_dataset.json"), "r", encoding="utf-8") as f:
    eval_content = json.loads(f.read())

corpus, queries, relevant_docs = eval_content['corpus'], eval_content['queries'], eval_content['relevant_docs']
# load train dataset
with open(os.path.join(project_dir, "data/ft_train_dataset.json"), "r", encoding="utf-8") as f:
    train_content = json.loads(f.read())

train_anchor, train_positive = [], []
for query_id, context_id in eval_content['relevant_docs'].items():
    train_anchor.append(eval_content['queries'][query_id])
    train_positive.append(eval_content['corpus'][context_id[0]])

train_dataset = Dataset.from_dict({"positive": train_positive, "anchor": train_anchor})

print(train_dataset[0:5])

# Load a model
model_name = 'bge-base-zh-v1.5'
# 替换成自己的模型完整路径或使用huggingface modl id
model_path = os.path.join(project_dir, f"models/{model_name}")
model = SentenceTransformer(model_path, device="cuda:0" if torch.cuda.is_available() else "cpu")
print("Model loaded")

# # Evaluate the model
evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name=f"{model_name}",
    score_functions={"cosine": cos_sim}
)
train_loss = MultipleNegativesRankingLoss(model)

# define training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=f"ft_{model_name}",  # output directory and hugging face model ID
    num_train_epochs=5,  # number of epochs
    per_device_train_batch_size=4,  # train batch size
    gradient_accumulation_steps=2,  # for a global batch size of 512
    per_device_eval_batch_size=4,  # evaluation batch size
    warmup_ratio=0.1,  # warmup ratio
    learning_rate=2e-5,  # learning rate, 2e-5 is a good value
    lr_scheduler_type="cosine",  # use constant learning rate scheduler
    optim="adamw_torch_fused",  # use fused adamw optimizer
    tf32=True,  # use tf32 precision
    bf16=True,  # use bf16 precision
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="epoch",  # evaluate after each epoch
    save_strategy="epoch",  # save after each epoch
    logging_steps=5,  # log every 10 steps
    save_total_limit=3,  # save only the last 3 models
    load_best_model_at_end=True,  # load the best model when training ends
    metric_for_best_model=f"eval_{model_name}_cosine_ndcg@10",  # Optimizing for the best ndcg@10 score
)

# train the model
trainer = SentenceTransformerTrainer(
    model=model,    # the model to train
    args=args,      # training arguments
    train_dataset=train_dataset.select_columns(
        ["positive", "anchor"]
    ),  # training dataset
    loss=train_loss,
    evaluator=evaluator
)

trainer.train()
trainer.save_model()
print(f"cost time: {time.time() - start_time:.2f}s")
