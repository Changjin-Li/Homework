import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import fasttext
import csv
from gensim.models import KeyedVectors, Word2Vec
from utils import random_word_vector, process_dataset, train, test, split_vector


class Config:
    def __init__(self, mode = "train"):
        self.tokens2id = pd.read_csv('dataset/tokens2id.csv').set_index('tokens')
        # the param of the network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42
        self.lr = 1e-4
        self.weight_decay = 1e-6
        self.epochs = 30
        self.batch_size = 64
        self.num_workers = 4
        self.dropout = 0.5
        self.num_classes = 5
        # CNN
        self.kernel_size = 3
        self.channels = 256
        # RNN
        self.num_layers = 1
        self.hidden_size = 128
        self.directions = 2
        self.save_path = "model/ablation_model.pth"
        self.mode = mode
        self.word2vec = "self"
        # the param of sentence embedding
        self.sentence_length = 64
        self.embedding_dim = 300
        self.vocab_size = self.tokens2id.shape[0] + 1


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        if config.mode == "train":
            if config.word2vec == "word2vec":
                self.word2vec_init()
            elif config.word2vec == "fasttext":
                self.fasttext_init()
            elif config.word2vec == "glove":
                self.glove_init()
            elif config.word2vec == "random":
                self.random_init()
            elif config.word2vec == "self":
                self.self_word2vec_init()

        self.conv = nn.Conv1d(
            in_channels = config.embedding_dim,
            out_channels = config.channels,
            kernel_size = config.kernel_size,
            padding = config.kernel_size // 2,
        )
        self.lstm = nn.LSTM(
            input_size = config.channels,
            hidden_size = config.hidden_size,
            num_layers = config.num_layers,
            batch_first = True,
            dropout = config.dropout if config.num_layers > 1 else 0,
            bidirectional = config.directions > 1,
        )
        self.fc = nn.Linear(config.hidden_size * config.directions, config.num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.dropout)

    def load_word2vec(self):
        print("Word2Vec loading...")
        words_vector = KeyedVectors.load_word2vec_format('model/googlenews.300d/GoogleNews-vectors-negative300.bin', binary=True)
        tokens = self.config.tokens2id.index.tolist()
        zero_init_weight = random_word_vector(self.config.embedding_dim, 0, 0)
        init_weight = [zero_init_weight]
        for token in tokens:
            word_vector = words_vector[token] if token in words_vector else random_word_vector(self.config.embedding_dim, 0, 0.1)
            init_weight.append(word_vector)
        print(f"Word2Vec load successfully, total {len(words_vector)}.")
        return init_weight

    def word2vec_init(self):
        init_weight = self.load_word2vec()
        init_weight = np.array(init_weight)
        self.embedding.weight.data.copy_(torch.Tensor(init_weight))
        self.embedding.weight.requires_grad = True

    def load_fasttext(self):
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
        return init_weight

    def fasttext_init(self):
        init_weight = self.load_fasttext()
        init_weight = np.array(init_weight)
        self.embedding.weight.data.copy_(torch.Tensor(init_weight))
        self.embedding.weight.requires_grad = True

    def load_glove(self):
        print("Glove loading...")
        words_vector = dict()
        tokens = self.config.tokens2id.index.tolist()
        zero_init_weight = random_word_vector(self.config.embedding_dim, 0, 0)
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
        init_weight = [zero_init_weight]
        for token in tokens:
            word_vector = words_vector[token] if token in words_vector else random_word_vector(self.config.embedding_dim, 0, 0.1)
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

    def glove_init(self):
        # self.load_glove()
        init_weight = pd.read_csv('model/glove.300d/word_vector.csv')['vector'].tolist()
        init_weight = [np.array(split_vector(v), dtype=np.float32) for v in init_weight]
        init_weight = np.array(init_weight)
        self.embedding.weight.data.copy_(torch.Tensor(init_weight))
        self.embedding.weight.requires_grad = True

    def random_init(self):
        zero_init_weight = random_word_vector(self.config.embedding_dim, 0, 0)
        init_weight = [zero_init_weight]
        tokens = self.config.tokens2id.index.tolist()
        for _ in tokens:
            word_vector = np.array(random_word_vector(self.config.embedding_dim, 0, 0.1))
            init_weight.append(word_vector)
        init_weight = np.array(init_weight)
        self.embedding.weight.data.copy_(torch.Tensor(init_weight))
        self.embedding.weight.requires_grad = True

    def self_word2vec_init(self):
        zero_init_weight = random_word_vector(self.config.embedding_dim, 0, 0)
        init_weight = [zero_init_weight]
        words_vector = Word2Vec.load('model/self_word2vec/word2vec.model').wv
        tokens = self.config.tokens2id.index.tolist()
        for token in tokens:
            if token in words_vector:
                word_vector = words_vector[token]
                word_vector = np.array(word_vector)
                word_vector = word_vector / np.linalg.norm(word_vector)
            else:
                word_vector = np.array(random_word_vector(self.config.embedding_dim, 0, 0.1))
            init_weight.append(word_vector)
        init_weight = np.array(init_weight)
        self.embedding.weight.data.copy_(torch.Tensor(init_weight))
        self.embedding.weight.requires_grad = True

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


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