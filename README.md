本项目用于Embedding模型的相关实验，包括Embedding模型评估、Embedding模型微调、Embedding模型量化等。

### 1. Embedding模型评估

参考脚本: `src/baseline_eval`目录：

- bge_base_zh_eval.py: BGE-base-zh-v1.5模型评估，作为基线评估（baseline）

评估结果参考 `docs/model_evaluation.md` 文档。


### 2. Embedding模型微调

- Using `sentence-transformers v3`:

```commandline
python src/finetune/ft_sentence_transformers_trainer.py
```

- Using AutoTrain:

```commandline
cd ./src/finetune
CUDA_VISIBLE_DEVICES=0 autotrain --config config.yml
```

- Using LlamaIndex Finetune Embeddings:

可查阅`参考文献5`。

### 3. Embedding模型量化

### 参考文献

1. Training and Finetuning Embedding Models with Sentence Transformers v3: [https://huggingface.co/blog/train-sentence-transformers](https://huggingface.co/blog/train-sentence-transformers)
2. Fine-tune Embedding models for Retrieval Augmented Generation (RAG): [https://www.philschmid.de/fine-tune-embedding-model-for-rag](https://www.philschmid.de/fine-tune-embedding-model-for-rag)
3. 俄罗斯套娃 (Matryoshka) 嵌入模型概述: [https://huggingface.co/blog/zh/matryoshka](https://huggingface.co/blog/zh/matryoshka)
4. Finetune Embeddings: [https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding/](https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding/)
5. NLP（八十六）RAG框架Retrieve阶段的Embedding模型微调: [https://mp.weixin.qq.com/s?__biz=MzU2NTYyMDk5MQ==&mid=2247486333&idx=1&sn=29d00d472647bc5d6e336bec22c88139&chksm=fcb9b2edcbce3bfb42ea149d96fb1296b10a79a60db7ad2da01b85ab223394191205426bc025&token=1376257911&lang=zh_CN#rd](https://mp.weixin.qq.com/s?__biz=MzU2NTYyMDk5MQ==&mid=2247486333&idx=1&sn=29d00d472647bc5d6e336bec22c88139&chksm=fcb9b2edcbce3bfb42ea149d96fb1296b10a79a60db7ad2da01b85ab223394191205426bc025&token=1376257911&lang=zh_CN#rd)
6. How to Fine-Tune Custom Embedding Models Using AutoTrain: [https://huggingface.co/blog/abhishek/finetune-custom-embeddings-autotrain](https://huggingface.co/blog/abhishek/finetune-custom-embeddings-autotrain)
7. Upload a dataset to the Hub: [https://huggingface.co/docs/datasets/v1.16.0/upload_dataset.html](https://huggingface.co/docs/datasets/v1.16.0/upload_dataset.html)
