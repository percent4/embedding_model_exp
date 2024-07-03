本项目用于Embedding模型的相关实验，包括Embedding模型评估、ReRank模型微调、Embedding模型微调、Embedding模型量化等。

### 1. Embedding模型评估

参考脚本: `src/baseline_eval`目录：

- bge_base_zh_eval.py: BGE-base-zh-v1.5模型评估，作为基线评估（baseline）

评估结果参考 `docs/model_evaluation.md` 文档。


### 2. Embedding模型微调

- Using `sentence-transformers v3`:

```commandline
python src/finetune/ft_embedding.py
```

- Using AutoTrain:

```commandline
cd ./src/finetune
CUDA_VISIBLE_DEVICES=0 autotrain --config config.yml
```

- Using LlamaIndex Finetune Embeddings:

可查阅`参考文献5`。

### 3. ReRank模型微调

- 数据合成: src/utils/make_ft_rerank_corpus.py
- 模型微调: src/finetune/ft_rerank.py
- 评估实验: [https://github.com/percent4/embedding_rerank_retrieval](https://github.com/percent4/embedding_rerank_retrieval)，评估结果参考 `docs/model_evaluation.md` 文档。

### 4. Embedding模型量化

- 基础测试: src/quantization/basic_test.py

### 5. Embedding Usage(应用)

- 图片搜索示例: src/usage/image_search.py

### 参考文献

1. Training and Finetuning Embedding Models with Sentence Transformers v3: [https://huggingface.co/blog/train-sentence-transformers](https://huggingface.co/blog/train-sentence-transformers)
2. Fine-tune Embedding models for Retrieval Augmented Generation (RAG): [https://www.philschmid.de/fine-tune-embedding-model-for-rag](https://www.philschmid.de/fine-tune-embedding-model-for-rag)
3. 俄罗斯套娃 (Matryoshka) 嵌入模型概述: [https://huggingface.co/blog/zh/matryoshka](https://huggingface.co/blog/zh/matryoshka)
4. Finetune Embeddings: [https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding/](https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding/)
5. NLP（八十六）RAG框架Retrieve阶段的Embedding模型微调: [https://mp.weixin.qq.com/s?__biz=MzU2NTYyMDk5MQ==&mid=2247486333&idx=1&sn=29d00d472647bc5d6e336bec22c88139&chksm=fcb9b2edcbce3bfb42ea149d96fb1296b10a79a60db7ad2da01b85ab223394191205426bc025&token=1376257911&lang=zh_CN#rd](https://mp.weixin.qq.com/s?__biz=MzU2NTYyMDk5MQ==&mid=2247486333&idx=1&sn=29d00d472647bc5d6e336bec22c88139&chksm=fcb9b2edcbce3bfb42ea149d96fb1296b10a79a60db7ad2da01b85ab223394191205426bc025&token=1376257911&lang=zh_CN#rd)
6. How to Fine-Tune Custom Embedding Models Using AutoTrain: [https://huggingface.co/blog/abhishek/finetune-custom-embeddings-autotrain](https://huggingface.co/blog/abhishek/finetune-custom-embeddings-autotrain)
7. Upload a dataset to the Hub: [https://huggingface.co/docs/datasets/v1.16.0/upload_dataset.html](https://huggingface.co/docs/datasets/v1.16.0/upload_dataset.html)
8. Training Examples » MS MARCO: [https://sbert.net/examples/training/ms_marco/cross_encoder_README.html](https://sbert.net/examples/training/ms_marco/cross_encoder_README.html)
9. train_cross-encoder_scratch.py: [https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_cross-encoder_scratch.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_cross-encoder_scratch.py)
10. NLP（八十三）RAG框架中的Rerank算法评估: [https://mp.weixin.qq.com/s/ZqBbrrZxlMtn2ohttAGDIQ](https://mp.weixin.qq.com/s/ZqBbrrZxlMtn2ohttAGDIQ)
11. NLP（一百零一）Embedding模型微调实践: [https://mp.weixin.qq.com/s/lJ3Mycjw1G99T08r8c7dSQ](https://mp.weixin.qq.com/s/lJ3Mycjw1G99T08r8c7dSQ)
