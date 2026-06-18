import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import csv
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch.nn.utils.rnn import pack_padded_sequence
import fasttext
from gensim.models import KeyedVectors
from utils import process_dataset, split_vector, random_word_vector, train, test


lr = {
    "random": {"CNN": 5e-4, "RNN": 5e-4},
    "fasttext": {"CNN": 1e-3, "RNN": 1e-3},
    "glove": {"CNN": 1e-4, "RNN": 1e-4},
    "word2vec": {"CNN": 1e-4, "RNN": 5e-4},
}
weight_decay = {
    "random": {"CNN": 1e-2, "RNN": 1e-2},
    "fasttext": {"CNN": 1e-5, "RNN": 1e-2},
    "glove": {"CNN": 1e-6, "RNN": 1e-6},
    "word2vec": {"CNN": 1e-6, "RNN": 1e-3},
}
dropout = {
    "random": {"CNN": 0.2, "RNN": 0.2},
    "fasttext": {"CNN": 0.5, "RNN": 0.5},
    "glove": {"CNN": 0.5, "RNN": 0.5},
    "word2vec": {"CNN": 0.5, "RNN": 0.2},
}
sentence_length = {
    "random": 128,
    "fasttext": 128,
    "glove": 50,
    "word2vec": 128,
}
model_name = ['random', 'fasttext', 'glove', 'word2vec'][3]


class Config:
    def __init__(self, mode = "train", word_vector = model_name, net = "CNN"):
        self.net = net
        self.mode = mode
        self.word_vector = word_vector
        self.tokens2id = pd.read_csv('dataset/tokens2id.csv').set_index('tokens')
        # the param of the network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42
        self.lr = lr[word_vector][net]
        self.weight_decay = weight_decay[word_vector][net]
        self.dropout = dropout[word_vector][net]
        self.epochs = 30
        self.batch_size = 64
        self.num_workers = 4
        self.num_classes = 5
        self.save_path = f"model_{net}/{word_vector}_model.pth"
        # ---------- CNN -----------
        self.filter_size = [3, 4, 5]
        self.num_filters = 100
        # ---------- RNN ----------
        self.num_layers = 1
        self.hidden_size = 128
        self.direction = 2
        # the param of sentence embedding
        self.sentence_length = sentence_length[word_vector]
        self.embedding_dim = 300
        self.vocab_size = self.tokens2id.shape[0] + 1


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        if config.mode == "train":
            if config.word_vector == "random":
                self.random_init()
            elif config.word_vector == "fasttext":
                self.fasttext_init()
            elif config.word_vector == "glove":
                self.glove_init()
            elif config.word_vector == "word2vec":
                self.word2vec_init()
            self.embedding.weight.requires_grad = True

        self.conv = None
        if config.net == "CNN":
            self.conv = nn.ModuleList([
                nn.Conv2d(1, self.config.num_filters, (k, self.config.embedding_dim))
                for k in self.config.filter_size
            ])

        self.lstm = None
        if config.net == "RNN":
            self.lstm = nn.LSTM(
            input_size = self.config.embedding_dim,
            hidden_size = self.config.hidden_size,
            num_layers = self.config.num_layers,
            dropout = self.config.dropout if self.config.num_layers > 1 else 0,
            bidirectional = self.config.direction > 1,
            batch_first = True,
            )

        self.fc = None
        if config.net == "CNN":
            self.fc = nn.Linear(len(self.config.filter_size) * self.config.num_filters, self.config.num_classes)
        elif config.net == "RNN":
            self.fc = nn.Linear(self.config.hidden_size * self.config.direction, self.config.num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.dropout)

    def random_init(self):
        zero_init_weight = random_word_vector(self.config.embedding_dim, 0, 0)
        init_weight = [zero_init_weight]
        tokens = self.config.tokens2id.index.tolist()
        for _ in tokens:
            word_vector = np.array(random_word_vector(self.config.embedding_dim, 0, 0.1))
            init_weight.append(word_vector)
        init_weight = np.array(init_weight)
        self.embedding.weight.data.copy_(torch.Tensor(init_weight))

    def fasttext_init(self):
        print("FastText loading...")
        tokens = self.config.tokens2id.index.tolist()
        zero_init_weight = random_word_vector(self.config.embedding_dim, 0, 0)
        init_weight = [zero_init_weight]
        model = fasttext.load_model("model/fasttext.300d/cc.en.300.bin")
        for token in tokens:
            try:
                word_vector = model.get_word_vector(token)
            except:
                word_vector = np.array(random_word_vector(self.config.embedding_dim, 0, 0.1))
            init_weight.append(word_vector)
        print(f"FastText load successfully, total {len(model.get_words())}.")
        init_weight = np.array(init_weight)
        self.embedding.weight.data.copy_(torch.Tensor(init_weight))

    def glove_init(self):
        print("Glove loading...")
        if not os.path.exists('model/glove.300d/word_vector.csv'):
            words_vector = dict()
            tokens = self.config.tokens2id.index.tolist()
            with open('model/glove.300d/glove.txt', encoding='utf-8') as f:
                for line in f:
                    vectors = line.split()
                    word = vectors[0]
                    if word not in tokens: continue
                    vector = []
                    for v in vectors[1:]:
                        try:
                            v_ = float(v)
                        except:
                            v_ = 0
                        vector.append(v_)
                    if len(vector) == self.config.embedding_dim:
                        words_vector[word] = vector
            zero_init_weight = random_word_vector(self.config.embedding_dim, 0, 0)
            init_weight = [zero_init_weight]
            for token in tokens:
                word_vector = words_vector[token] if token in words_vector else random_word_vector(
                    self.config.embedding_dim, 0, 0.1)
                init_weight.append(word_vector)
            csv_file = open('model/glove.300d/word_vector.csv', "w", newline='')
            name = ['vector']
            try:
                writer = csv.writer(csv_file)
                writer.writerow(name)
                for i in range(len(init_weight)):
                    writer.writerows([[init_weight[i]]])
            finally:
                csv_file.close()
            print(f"Glove load successfully, total {len(words_vector)}.")

        init_weight = pd.read_csv('model/glove.300d/word_vector.csv')['vector'].tolist()
        init_weight = [np.array(split_vector(v), dtype=np.float32) for v in init_weight]
        init_weight = np.array(init_weight)
        self.embedding.weight.data.copy_(torch.Tensor(init_weight))

    def word2vec_init(self):
        print("Word2Vec loading...")
        words_vector = KeyedVectors.load_word2vec_format('model/googlenews.300d/GoogleNews-vectors-negative300.bin', binary=True)
        tokens = self.config.tokens2id.index.tolist()
        zero_init_weight = random_word_vector(self.config.embedding_dim, 0, 0)
        init_weight = [zero_init_weight]
        for token in tokens:
            word_vector = words_vector[token] if token in words_vector else random_word_vector(
                self.config.embedding_dim, 0, 0.1)
            init_weight.append(word_vector)
        print(f"Word2Vec load successfully, total {len(words_vector)}.")
        init_weight = np.array(init_weight)
        self.embedding.weight.data.copy_(torch.Tensor(init_weight))

    def forward(self, x):
        out = []

        if self.config.net == "CNN":
            x = self.embedding(x)
            x = x.unsqueeze(1)
            for conv in self.conv:
                h = self.relu(conv(x)).squeeze(-1)
                h = F.max_pool1d(h, h.size(-1)).squeeze(-1)
                out.append(h)
            out = torch.cat(out, 1)
            out = self.dropout(out)
            out = self.fc(out)

        elif self.config.net == "RNN":
            lengths = (x != 0).sum(dim=1).view(-1).cpu().tolist()
            x = self.embedding(x)
            x = pack_padded_sequence(
                x,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            x_n, (h_n, c_n) = self.lstm(x)
            out = [h_n[-1], h_n[-2]] if self.config.direction > 1 else [h_n[-1]]
            out = torch.cat(out, 1)
            out = self.dropout(out)
            out = self.fc(out)

        return out


def main():
    config = Config()
    pl.seed_everything(config.seed)

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
