import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from utils import random_word_vector, split_vector, process_dataset, train, test


class Config:
    def __init__(self):
        self.tokens2id = pd.read_csv('dataset/tokens2id.csv').set_index('tokens')
        # the param of the network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = 1e-4
        self.weight_decay = 1e-7
        self.epochs = 50
        self.batch_size = 64
        self.num_workers = 4
        self.dropout = 0.5
        self.num_classes = 5
        self.filter_size = [3, 4, 5]
        self.num_filters = 100
        self.save_path = "model/googlenews_model.pth"
        # the param of sentence embedding
        self.sentence_length = 50
        self.embedding_dim = 300
        self.vocab_size = self.tokens2id.shape[0] + 1


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
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

    def load_googlenews(self):
        print("GoogleNews loading...")
        words_vector = KeyedVectors.load_word2vec_format('model/googlenews.300d/GoogleNews-vectors-negative300.bin', binary=True)
        tokens = self.config.tokens2id.index.tolist()
        zero_init_weight = random_word_vector(self.config.embedding_dim, 0, 0)
        init_weight = [zero_init_weight]
        for token in tokens:
            try:
                word_vector = words_vector.get_vector(token)
            except:
                word_vector = random_word_vector(self.config.embedding_dim, 0, 0)
            init_weight.append(word_vector)
        csv_file = open('model/googlenews.300d/word_vector.csv', "w", newline='')
        name = ['vector']
        try:
            writer = csv.writer(csv_file)
            writer.writerow(name)
            for i in range(len(init_weight)):
                writer.writerows([[init_weight[i]]])
        finally:
            csv_file.close()
        print(f"GoogleNews load successfully, total {len(words_vector)}.")

    def word2vec_init(self):
        self.load_googlenews()
        init_weight = pd.read_csv('model/googlenews.300d/word_vector.csv')['vector'].tolist()
        init_weight = [np.array(split_vector(v), dtype=np.float32) for v in init_weight]
        init_weight = np.array(init_weight)
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


def main():
    config = Config()
    train_data = process_dataset(config, 'dataset/train.csv')
    test_data  = process_dataset(config, 'dataset/test.csv')
    dev_data   = process_dataset(config, 'dataset/dev.csv')
    model = Model().to(config.device)
    train(config, model, train_data, dev_data)
    model.load_state_dict(torch.load(config.save_path, weights_only=True))
    test_acc = test(config, model, test_data)
    print(f"Test Accuracy: {test_acc * 100:.2f}%.")


if __name__ == "__main__":
    main()