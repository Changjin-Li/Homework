# 基于卷积神经网络的文本情感分类

## 模型下载
+ Word2Vec预训练词向量下载地址：https://code.google.com/archive/p/word2vec/
+ Glove 预训练词向量下载地址：https://nlp.stanford.edu/projects/glove/
+ FastText 预训练词向量下载地址：https://fasttext.cc/docs/en/crawl-vectors.html
+ Bert 预训练模型下载地址：https://huggingface.co/google-bert/bert-base-uncased

## 模型评估

|   Word_Vec   | Model |   Accuracy    |
|:------------:|:-----:|:-------------:|
|    Random    |  CNN  |    43.67%     |
|   FastText   |  CNN  | 43.39 ± 0.95% |
|    Glove     |  CNN  | 46.11 ± 0.82% |
|   Word2Vec   |  CNN  | 48.19 ± 0.59% |
|     Bert     |   -   |      ± %      |

### Pretrained Model

If you can't successfully download pretrained model, you can use the code:

```bash
pip install huggingface-hub
export HF_ENDPOINT=https://hf-mirror.com
hf download airkingbd/dplm_650m --local-dir ./dplm_650m
```
