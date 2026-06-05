import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import pandas as pd
import time


class TextDataset(data.Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def random_word_vector(length, mean: float = 0, std: float = 1):
    if std == 0: return [0] * length
    vector = np.random.normal(mean, std, length)
    vector = vector / np.linalg.norm(vector)
    return vector.tolist()


def split_sentence(sentence, tokens2id = None, mode = 'token'):
    """
    Split a sentence from dataset into a list of tokens.
    :param sentence: the sentence to split
    :param tokens2id: the table for turning tokens into id
    :param mode: 'token' or 'id', if 'token' then return ['a', 'b', 'c'] else return [1, 2, 3]
    """
    words = sentence.split()
    tokens = [words[0][2:-2]]
    if len(words) > 1:
        for idx in range(1, len(words)):
            tokens.append(words[idx][1:-2])
    if mode == 'token':
        return tokens
    elif mode == 'id':
        tokens = [int(tokens2id.loc[token, ['id']].tolist()[0]) + 1 for token in tokens]
        return tokens
    else:
        raise ValueError


def split_vector(vector):
    """
    Split a vector into a list of nums.
    :param vector: the vector to split
    :return: [num1, num2, ...]
    """
    vectors = vector.split()
    word_vector = [float(vectors[0][1:-1])]
    for idx in range(1, len(vectors)):
        word_vector.append(float(vectors[idx][:-1]))
    return word_vector


def process_dataset(config, file_path):
    """mode: 'token' or 'id', if 'token' then return ['a', 'b', 'c'] else return [1, 2, 3]"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    sentences = df['sentences'].apply(lambda x: split_sentence(x, config.tokens2id, 'id')).tolist()
    sentence_length = config.sentence_length
    sentences = [s[:sentence_length] if len(s) >= sentence_length else s + [0] * (sentence_length - len(s)) for s in sentences]
    labels = df['label'].apply(lambda x: int(x)).tolist()
    sentences, labels = torch.LongTensor(sentences), torch.LongTensor(labels)
    print("Successfully load data from", file_path)
    return TextDataset(sentences, labels)


def train(config, model, train_data: TextDataset, test_data: TextDataset):
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    best_acc = 0
    now_time = time.time()
    print("Start Training...")
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for text, label in train_loader:
            text = text.to(config.device)
            label = label.to(config.device)
            optimizer.zero_grad()
            output = model(text)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        use_time = time.time() - now_time
        now_time = time.time()
        acc = test(config, model, test_data)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), config.save_path)
        print(f"[{epoch + 1} / {config.epochs}] \tloss: {total_loss / len(train_loader):.4f}, \tacc: {acc * 100:.2f}%, \tuse_time: {use_time:.2f}s")
    print(f"Finish Training. Best Accuracy: {best_acc * 100:.2f}%.")


def test(config, model, test_data: TextDataset):
    correct, total = 0, 0
    test_loader = data.DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model.to(config.device)
    model.eval()
    with torch.no_grad():
        for text, label in test_loader:
            text = text.to(config.device)
            prediction = model(text)
            prediction = prediction.argmax(dim=1)
            prediction = prediction.cpu().numpy()
            label = label.numpy().squeeze()
            correct += (prediction == label).sum()
            total += label.shape[0]

    return correct / total


if __name__ == "__main__":
    pass