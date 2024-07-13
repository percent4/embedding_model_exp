# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: get_text_embedding.py
# @time: 2024/7/11 10:19
import os
import json
import requests
from typing import List
from dotenv import load_dotenv


load_dotenv()


def get_embedding(texts: List[str]):
    url = "https://api.openai.com/v1/embeddings"
    payload = json.dumps({
        "model": "text-embedding-ada-002",
        "input": texts,
        "encoding_format": "float"
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    embedding = [_["embedding"] for _ in response.json()['data']]
    response.close()
    return embedding
