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

| Word_Vec | Model | Accuracy | Model | Accuracy |
|:--------:|:-----:|:--------:|:-----:|:--------:|
|  Random  |  CNN  |  43.80%  |  RNN  |  43.03%  |
| FastText |  CNN  |  44.71%  |  RNN  |  44.57%  |
|  Glove   |  CNN  |  46.79%  |  RNN  |  45.20%  |
| Word2Vec |  CNN  |  48.73%  |  RNN  |  45.61%  |
|   BERT   |  CNN  |  52.85%  |  RNN  |  51.31%  |
