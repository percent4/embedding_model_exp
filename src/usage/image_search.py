# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: image_search.py
# @time: 2024/7/2 17:21
import os
from PIL import Image
from sentence_transformers import SentenceTransformer

project_dir = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]

# Load CLIP model
model_path = os.path.join(project_dir, "models/clip-ViT-B-32")
model = SentenceTransformer(model_path)

# Encode an image:
image_path = os.path.join(project_dir, "data/two_dogs_in_snow.jpg")
img_emb = model.encode(Image.open(image_path))

# Encode text descriptions
text_emb = model.encode(
    ["Two dogs in the snow", "A cat on a table", "A picture of London at night"]
)

# Compute similarities
similarity_scores = model.similarity(img_emb, text_emb)
print(similarity_scores)
print(type(similarity_scores))

"""
tensor([[0.3072, 0.1016, 0.1095]])
<class 'torch.Tensor'>
"""