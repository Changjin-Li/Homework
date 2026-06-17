import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
import pandas as pd
import time


class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42
        self.lr = 2e-5
        self.weight_decay = 5e-3
        self.epochs = 20
        self.batch_size = 128
        self.num_workers = 8
        self.num_classes = 5
        self.max_length = 128
        self.model_path = 'model/Bert/bert-base-uncased'
        self.save_path = "model/bert_model_rnn.pth"
        self.dropout = 0.3
        # CNN
        self.channels = 256
        self.kernel_size = 5
        # RNN
        self.lstm_hidden_size = 128
        self.lstm_num_layers = 1
        self.directions = 2


class BertClassifier(nn.Module):
    def __init__(self, config, net = "cnn + rnn"):
        super().__init__()
        self.config = config
        self.net = net

        self.bert = BertModel.from_pretrained(config.model_path)
        hidden_size = self.bert.config.hidden_size

        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=config.channels,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2,
        )

        self.lstm = nn.LSTM(
            input_size=config.channels if "cnn" in net else hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            dropout=config.dropout_rate if config.lstm_num_layers > 1 else 0,
            bidirectional=config.directions > 1,
        )

        hidden_size = config.channels if "cnn" in net else hidden_size
        hidden_size = config.lstm_hidden_size * config.directions if "rnn" in net else hidden_size
        self.fc = nn.Linear(hidden_size, config.num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, attention_mask):
        # ----- BERT -----
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state

        # ----- CNN -----
        if "cnn" in self.net:
            cnn_input = sequence_output.permute(0, 2, 1)
            conv_output = self.conv(cnn_input)
            sequence_output = self.relu(conv_output)
            sequence_output = sequence_output.permute(0, 2, 1)

        # ----- LSTM -----
        if "rnn" in self.net:
            sequence_output, (h_n, c_n) = self.lstm(sequence_output)

        # ----- FC -----
        pooled = torch.mean(sequence_output, dim=1)
        # pooled = lstm_output[:, -1, :]
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['sentences'])
        label = int(self.data.iloc[idx]['label'])
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class Model:
    def __init__(self, config):
        self.tokenizer = BertTokenizer.from_pretrained(config.model_path)
        self.model = BertClassifier(config).to(config.device)
        self.model.load_state_dict(torch.load(config.save_path, weights_only=True))
    def test(self):
        return test("test", self.tokenizer, self.model)


def train():
    config = Config()
    pl.seed_everything(config.seed)

    tokenizer = BertTokenizer.from_pretrained(config.model_path)
    model = BertClassifier(config).to(config.device)
    train_data = TextDataset(pd.read_csv("dataset/train.csv"), tokenizer, config)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print("Start Training Classifier...")
    best_acc = 0
    for epoch in range(1, config.epochs + 1):
        start_time = time.time()
        total_loss = 0
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        test_acc = test("dev", tokenizer, model)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), config.save_path)
        use_time = time.time() - start_time
        print(f"[{epoch} / {config.epochs}] \tloss: {total_loss / len(train_loader):.4f}, \tacc: {test_acc * 100:.2f}%, \tuse_time: {use_time:.2f}s")
    print(f"Finish Training. Best Accuracy: {best_acc * 100:.2f}%.")

    model.load_state_dict(torch.load(config.save_path, weights_only=True))
    test_acc = test("test", tokenizer, model)
    print(f"Test Accuracy: {test_acc * 100:.2f}%.")


def test(data, tokenizer, model):
    config = Config()

    if data == "test":
        test_data = TextDataset(pd.read_csv("dataset/test.csv"), tokenizer, config)
    elif data == "dev":
        test_data = TextDataset(pd.read_csv("dataset/dev.csv"), tokenizer, config)
    else:
        raise ValueError("Unknown dataset.")
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_labels = []
        for batch in test_loader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs, dim=-1)
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        test_acc = accuracy_score(all_labels, all_predictions)

    return test_acc


if __name__ == "__main__":
    train()
