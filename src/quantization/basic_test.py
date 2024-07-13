# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: basic_test.py
# @time: 2024/6/14 10:47
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings

# 1. Load an embedding model
model_name = 'bge-base-zh-v1.5'
project_dir = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]
model_path = os.path.join(project_dir, f"models/{model_name}")
model = SentenceTransformer(model_path)

# 2a. Encode some text using "binary" quantization
sentences = ["小米手机销量不错。", "人工智能技术发展迅速。"]
binary_embeddings = model.encode(sentences, precision="ubinary")
print('*' * 50)
print(binary_embeddings.shape)
print(binary_embeddings.nbytes)
print(binary_embeddings)

# 2b. or, encode some text without quantization & apply quantization after wards
embeddings = model.encode(sentences)
binary_embeddings_v2 = quantize_embeddings(embeddings, precision="ubinary")
print('*' * 50)
print(embeddings.shape)
print(embeddings.nbytes)
print(embeddings)
binary_embeddings_cal = np.packbits(np.where(embeddings > 0, 1, 0), axis=-1)
print('*' * 50)
print(binary_embeddings_v2.shape)
print(binary_embeddings_v2.nbytes)
print(binary_embeddings_v2)
print('*' * 50)
print(np.array_equal(binary_embeddings, binary_embeddings_v2))
print(np.array_equal(binary_embeddings, binary_embeddings_cal))
