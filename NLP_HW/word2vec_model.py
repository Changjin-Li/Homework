import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from utils import split_sentence, process_dataset, random_word_vector, train, test
from gensim.models import Word2Vec


class Config:
    def __init__(self, mode = "train"):
        self.tokens2id = pd.read_csv('dataset/tokens2id.csv').set_index('tokens')
        # the param of the network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.epochs = 50
        self.batch_size = 64
        self.num_workers = 4
        self.dropout = 0.2
        self.num_classes = 5
        self.filter_size = [3, 4, 5]
        self.num_filters = 100
        self.save_path = "model/word2vec_model.pth"
        self.mode = mode
        # the param of sentence embedding
        self.sentence_length = 50
        self.embedding_dim = 300
        self.vocab_size = self.tokens2id.shape[0] + 1
        # the param of word2vec
        self.window_size = 5
        self.negative_samples = 20
        self.min_count = 3


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.word2vec_init()
        self.conv = nn.ModuleList([
            nn.Conv2d(1, self.config.num_filters, (k, self.config.embedding_dim))
            for k in self.config.filter_size
        ])
        self.fc = nn.Sequential(
            nn.Linear(len(self.config.filter_size) * self.config.num_filters, self.config.num_classes),
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.dropout)

    def word2vec_init(self):
        if self.config.mode == "train":
            word2vec(self.config.embedding_dim)
        zero_init_weight = random_word_vector(self.config.embedding_dim, 0, 0)
        init_weight = [zero_init_weight]
        words_vector = Word2Vec.load('model/self_word2vec/word2vec.model').wv
        tokens = self.config.tokens2id.index.tolist()
        for token in tokens:
            word_vector = words_vector[token] if token in words_vector else random_word_vector(self.config.embedding_dim, 0, 0.1)
            init_weight.append(word_vector)
        init_weight = np.array(init_weight, dtype=np.float32)
        self.embedding.weight.data.copy_(torch.Tensor(init_weight))
        self.embedding.weight.requires_grad = True

    def forward(self, x):
        vec = self.embedding(x)                     # BxLxD
        vec = vec.unsqueeze(1)                      # Bx1xLxD
        out = []
        for conv in self.conv:
            h = self.relu(conv(vec)).squeeze(-1)    # Bx1xLxD -> BxCxLx1 -> BxCxL -> BxCx1 -> BxC
            h = F.max_pool1d(h, h.size(-1)).squeeze(-1)
            out.append(h)
        out = torch.cat(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def word2vec(vector_size):
    config = Config()
    sentences = pd.concat([
        pd.read_csv('dataset/train.csv'),
        pd.read_csv('dataset/dev.csv'),
        pd.read_csv('dataset/test.csv'),
    ])['sentences'].tolist()
    sentences = [split_sentence(sentence, mode='token') for sentence in sentences]
    model = Word2Vec(
        sentences = sentences,
        vector_size = vector_size,
        window = config.window_size,
        negative = config.negative_samples,
        min_count = config.min_count,
        seed = config.seed,
    )
    model.save('model/self_word2vec/word2vec.model')
    return model


def main():
    config = Config()
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    train_data = process_dataset(config, 'dataset/train.csv')
    test_data  = process_dataset(config, 'dataset/test.csv')
    dev_data   = process_dataset(config, 'dataset/dev.csv')
    model = Model(config).to(config.device)
    train(config, model, train_data, dev_data)
    model.load_state_dict(torch.load(config.save_path, weights_only=True))
    test_acc = test(config, model, test_data)
    print(f"Test Accuracy: {test_acc * 100:.2f}%.")


if __name__ == "__main__":
    main()