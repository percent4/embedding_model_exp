不同模型的评估指标如下：

| 模型                        | accuracy@5 | accuracy@10 | map@100 | mrr@10 | ndcg@10  | cost time |
|---------------------------|------------|-------------|---------|--------|----------|-----------|
| bge-base-zh-v1.5          | 0.8100     | 0.8816      | 0.6998  | 0.6945 | 0.7396   | 15.44s    |
| ft_bge-base-zh-v1.5       | 0.9128     | 0.9408      | 0.8052  | 0.8018 | 0.8362   | 15.14s    |
| autotrain-bge-base-zh-v15 | 0.9159     | 0.9346      | 0.7952  | 0.7918 | 0.8272   | 15.07s    |

注意：

- ft_bge-base-zh-v1.5和autotrain-bge-base-zh-v15都是对基准模型bge-base-zh-v1.5进行微调得到的模型，前者使用sentence-transformers微调，后者使用AutoTrain微调。
- 评估脚本为 src/baseline_eval/bge_base_zh_eval.py，使用Mac CPU测试，CPU型号为 `Apple M2 Pro` 。