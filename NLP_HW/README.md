# 基于卷积神经网络的文本情感分类

## 模型下载
+ FastText 预训练词向量下载地址：https://fasttext.cc/docs/en/crawl-vectors.html
+ Glove 预训练词向量下载地址：https://nlp.stanford.edu/projects/glove/
+ Word2Vec预训练词向量下载地址：https://code.google.com/archive/p/word2vec/
+ Bert 预训练模型下载地址：https://huggingface.co/google-bert/bert-base-uncased

```bash
pip install huggingface-hub
export HF_ENDPOINT=https://hf-mirror.com
hf download [model] --local-dir [model_dir]
```

## 模型评估

|   Word_Vec   | Model |   Accuracy    | Model | Accuracy |
|:------------:|:-----:|:-------------:|:-----:|:--------:|
|    Random    |  CNN  |    43.80%     |  RNN  |  43.03%  |
|   FastText   |  CNN  |    %     |  RNN  |    %     |
|    Glove     |  CNN  |    46.92%     |  RNN  |  45.25%  |
|   Word2Vec   |  CNN  | % |  RNN  |    %     |
|     Bert     |  CNN  |    52.08%     |  RNN  |  50.72%  |
