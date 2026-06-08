import torch
import time
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score


class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42
        self.lr = [1e-3, 5e-6]
        self.weight_decay = [1e-4, 1e-2]
        self.epochs = [30, 20]
        self.batch_size = 64
        self.num_workers = 4
        self.num_classes = 5
        self.max_length = 64
        self.model_path = 'model/bert-base-uncased'
        self.save_path = "model/bert_model"


class Model:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.save_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.save_path, num_labels=config.num_classes).to(config.device)

    def test(self, test_data="test"):
        test_acc = test(test_data, self.tokenizer, self.model)
        return test_acc


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


def train():
    config = Config()
    pl.seed_everything(config.seed)

    # freeze the weight of encoder, only train Classifier
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_path, num_labels=config.num_classes).to(config.device)

    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    print("Training layers: ")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f" - {name}")

    train_data = TextDataset(pd.read_csv("dataset/train.csv"), tokenizer, config)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr[0], weight_decay=config.weight_decay[0])

    print("Start Training Classifier...")
    model.train()
    max_acc = 0
    for epoch in range(1, config.epochs[0] + 1):
        start_time = time.time()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        test_acc = test("dev", tokenizer, model)
        if test_acc > max_acc:
            max_acc = test_acc
            tokenizer.save_pretrained(config.save_path)
            model.save_pretrained(config.save_path)

        print(f"Epoch [{epoch}/{config.epochs[0]}] \tLoss: {total_loss / len(train_loader):.4f} \tAcc: {test_acc * 100:.2f}% \tTime: {time.time() - start_time:.2f}s")
    print("Training Finished!")

    # train Model
    tokenizer = AutoTokenizer.from_pretrained(config.save_path)
    model = AutoModelForSequenceClassification.from_pretrained(config.save_path, num_labels=config.num_classes).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr[1], weight_decay=config.weight_decay[1])

    model.train()
    max_acc = 0
    for epoch in range(1, config.epochs[1] + 1):
        start_time = time.time()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        test_acc = test("dev", tokenizer, model)
        if test_acc > max_acc:
            max_acc = test_acc
            tokenizer.save_pretrained(config.save_path)
            model.save_pretrained(config.save_path)

        print(f"Epoch [{epoch}/{config.epochs[1]}] \tLoss: {total_loss / len(train_loader):.4f} \tAcc: {test_acc * 100:.2f}% \tTime: {time.time() - start_time:.2f}s")
    print("Training Finished!")

    # test
    tokenizer = AutoTokenizer.from_pretrained(config.save_path)
    model = AutoModelForSequenceClassification.from_pretrained(config.save_path, num_labels=config.num_classes).to(config.device)

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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            pred = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        test_acc = accuracy_score(all_labels, all_predictions)

    return test_acc


if __name__ == "__main__":
    train()
