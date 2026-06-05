import json
import torch
from datasets import load_dataset
import pytorch_lightning as pl
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

from utils import load_data, load_model, bleu_score, Translate


class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrain_model_path = "pretrain_models/Guwen_Translation"
        self.save_model_path = "models/Guwen_Translation_v1"
        self.dataset_path = "data/translation.json"
        self.model_path = ["models/Guwen_Translation_v1", "models/Guwen_Translation_v2"]

        self.seed = 42
        self.lr = 5e-6
        self.weight_decay = 0.01
        self.batch_size = 64
        self.epochs = 20
        self.max_input_length = 64
        self.max_output_length = 256
        self.max_length = 150
        self.num_test_data = 100
        self.translate_iteration = 0

        self.tokenizer_kwargs = dict(
            truncation = True,
            max_length = self.max_length,
            padding = "max_length",
            # return_tensors = 'pt',
        )
        self.generate_kwargs = dict(
            num_beams = 3,
            max_length = self.max_output_length,
            bos_token_id = 101,
        )


def train():
    config = Config()
    pl.seed_everything(config.seed)

    tokenizer, model = load_model(config.pretrain_model_path, config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.bos_token_id is not None:
        model.config.decoder_start_token_id = tokenizer.bos_token_id
    else:
        model.config.decoder_start_token_id = 101

    model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    model.generation_config.pad_token_id = model.config.pad_token_id

    load_data(config.dataset_path, config)
    train_dataset = load_dataset("json", data_files="data/train.json")
    test_dataset = load_dataset("json", data_files="data/test.json")

    def process_function(examples):
        model_inputs = tokenizer(examples["input"], **config.tokenizer_kwargs)
        labels = tokenizer(examples["output"], **config.tokenizer_kwargs)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = train_dataset.map(process_function, batched=True, remove_columns=["input", "output"])
    test_dataset = test_dataset.map(process_function, batched=True, remove_columns=["input", "output"])

    training_arguments = Seq2SeqTrainingArguments(
        output_dir=config.save_model_path,
        eval_strategy="epoch",
        learning_rate=config.lr,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        save_total_limit=1,
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset["train"],
        eval_dataset=test_dataset["train"],
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    trainer.train()

    model.save_pretrained(config.save_model_path)
    tokenizer.save_pretrained(config.save_model_path)
    
    test()


def test():
    config = Config()
    pl.seed_everything(config.seed)

    tokenizer, model = load_model(config.save_model_path, config)
    load_data(config.dataset_path, config)

    with open("data/test.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    BLEU = [0, 0, 0, 0]

    for example in data:
        response_sentence = Translate(example, (tokenizer, model), config)
        bleu = bleu_score(response_sentence, example["output"])
        BLEU[0] += bleu["BLEU1"]
        BLEU[1] += bleu["BLEU2"]
        BLEU[2] += bleu["BLEU3"]
        BLEU[3] += bleu["BLEU4"]

    print(f"BLEU-1: {BLEU[0] / len(data): .4f}\tBLEU-2: {BLEU[1] / len(data): .4f}\tBLEU-3: {BLEU[2] / len(data): .4f}\tBLEU-4: {BLEU[3] / len(data): .4f}")


def evaluate(mode = "self"):
    if mode == "self":
        config = Config()
        pl.seed_everything(config.seed)

        models = [load_model(config.model_path[i], config) for i in range(len(config.model_path))]

        load_data(config.dataset_path, config)

        with open("data/test.json", 'r', encoding='utf-8') as f:
            data = json.load(f)

        BLEU = [0, 0, 0, 0]

        for example in data:
            translated_text = Translate(example, models, config)[0]
            bleu = bleu_score(translated_text, example["output"])

            BLEU[0] += bleu["BLEU1"]
            BLEU[1] += bleu["BLEU2"]
            BLEU[2] += bleu["BLEU3"]
            BLEU[3] += bleu["BLEU4"]

        print(f"BLEU-1: {BLEU[0] / len(data): .4f}\tBLEU-2: {BLEU[1] / len(data): .4f}\t"
              f"BLEU-3: {BLEU[2] / len(data): .4f}\tBLEU-4: {BLEU[3] / len(data): .4f}")

    elif mode == "others":
        with open("data/data.json", 'r', encoding='utf-8') as f:
            data = json.load(f)

        BLEU = [0, 0, 0, 0]
        total_data = len(data)

        for example in data:
            bleu = bleu_score(example["response"], example["output"])

            if bleu["BLEU1"] < 0.1:
                total_data -= 1
                continue

            BLEU[0] += bleu["BLEU1"]
            BLEU[1] += bleu["BLEU2"]
            BLEU[2] += bleu["BLEU3"]
            BLEU[3] += bleu["BLEU4"]

        print(f"BLEU-1: {BLEU[0] / total_data: .4f}\tBLEU-2: {BLEU[1] / total_data: .4f}\t"
              f"BLEU-3: {BLEU[2] / total_data: .4f}\tBLEU-4: {BLEU[3] / total_data: .4f}")


if __name__ == "__main__":
    # train()
    evaluate("self")

