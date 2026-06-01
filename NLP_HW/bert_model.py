import torch
import time
import pandas as pd
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score


class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42
        self.lr = 1e-3
        self.weight_decay = 5e-4
        self.epochs = 10
        self.batch_size = 64
        self.num_workers = 4
        self.num_classes = 5
        self.max_length = 50
        self.model_path = 'model/bert-base-uncased'
        self.save_model_path = "model/bert_model"


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

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_path, num_labels=config.num_classes).to(config.device)

    train_data = TextDataset(pd.read_csv("dataset/train.csv"), tokenizer, config)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    print("Start Training...")
    model.train()
    max_acc = 0
    for epoch in range(1, config.epochs + 1):
        start_time = time.time()
        total_loss = 0
        all_predictions = []
        all_labels = []
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
            pred = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_predictions)
        test_acc = test("dev", tokenizer, model)
        if test_acc > max_acc:
            max_acc = acc
            model.save_pretrained(config.save_model_path)
            tokenizer.save_pretrained(config.save_model_path)
        print(f"Epoch [{epoch}/{config.epochs}] \tLoss: {total_loss / len(train_loader):.4f} \tAcc: {acc * 100:.2f}% \tTime: {time.time() - start_time:.2f}s")

    print("Training Finished!")
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

