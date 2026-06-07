import json
import torch
import numpy as np
from typing import List, Union, Tuple
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, EncoderDecoderModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, config):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_sentence = item["input"]
        output_sentence = item["output"]
        input_tokens = self.tokenizer(input_sentence, **self.config.tokenizer_kwargs)
        output_tokens = self.tokenizer(output_sentence, **self.config.tokenizer_kwargs)

        input_ids = input_tokens["input_ids"].squeeze(0)
        attention_mask = input_tokens["attention_mask"].squeeze(0)
        labels = output_tokens["input_ids"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attn_mask": attention_mask,
            "labels": labels,
            "input": input_sentence,
            "output": output_sentence,
        }


def load_model(model_path, config):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = EncoderDecoderModel.from_pretrained(model_path).to(config.device)

    return tokenizer, model


def module_name(model):
    for name, module in model.named_modules():
        print(name, type(module))


def load_data(data_path, config):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_data, test_data = train_test_split(data, test_size=config.num_test_data / len(data), random_state=config.seed)

    process_train_data = []
    for text in train_data:
        if len(text["input"]) < config.max_input_length * 2:
            process_train_data.append(text)
        else:
            process_train_data.append({"input": text["input"][:config.max_input_length], "output": text["output"][:config.max_output_length]})

    with open("data/train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open("data/test.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)


def translate(text: str, tokenizer, model, config):
    translated_text = ""

    model.eval()
    with torch.no_grad():
        # [0, 127] + [128, 255] + ...
        for idx in range(1 + len(text) // config.max_input_length):
            inputs = tokenizer(
                text[idx * config.max_input_length : (idx + 1) * config.max_input_length],
                **config.tokenizer_kwargs,
                return_tensors = 'pt'
            ).to(config.device)
            outputs = model.generate(
                inputs.input_ids,
                attention_mask = inputs.attention_mask,
                **config.generate_kwargs,
                eos_token_id = tokenizer.sep_token_id,
                pad_token_id = tokenizer.pad_token_id,
            )
            output_text = tokenizer.decode(outputs, skip_special_tokens=True)
            translated_text += output_text[0].replace(" ", "")

    return translated_text


def translate_iterate(text: str, tokenizer, model, config):
    translated_text = translate(text, tokenizer, model, config)

    if config.translate_iteration > 0:
        for _ in range(config.translate_iteration):
            translated_text = translate(translated_text, tokenizer, model, config)

    return translated_text


def translate_with_conf(text: str, generate_text: str, tokenizer, model, config):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, truncation=True, return_tensors="pt").to(config.device)
        labels = tokenizer(generate_text, truncation=True, return_tensors="pt").to(config.device)

        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels.input_ids,
        )

        confidence = -outputs.loss.item()

    return confidence


def bleu_score(response, output):
    """
    Calculate the BLEU score between two sentences.
    :param response: The output sentence of the model.
    :param output: The ground truth sentence.
    :return: a dict including BLEU1, BLEU2, BLEU3, BLEU4
    """
    smoother = SmoothingFunction().method4
    if len(output) > len(response) + 100:
        output = output[:len(response) + 100]

    bleu_score_1 = sentence_bleu([output], response, weights=(1, 0, 0, 0), smoothing_function=smoother)
    bleu_score_2 = sentence_bleu([output], response, weights=(1/2, 1/2, 0, 0), smoothing_function=smoother)
    bleu_score_3 = sentence_bleu([output], response, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoother)
    bleu_score_4 = sentence_bleu([output], response, weights=(1/4, 1/4, 1/4, 1/4), smoothing_function=smoother)

    return {
        "BLEU1": round(bleu_score_1, 4),
        "BLEU2": round(bleu_score_2, 4),
        "BLEU3": round(bleu_score_3, 4),
        "BLEU4": round(bleu_score_4, 4)
    }


def Translate(
        texts: Union[str, dict, List[str], List[dict]],
        models: Union[Tuple[AutoTokenizer, EncoderDecoderModel], List[Tuple[AutoTokenizer, EncoderDecoderModel]]],
        config = None,
    ) -> List[str]:
    """The translating function."""
    if not isinstance(texts, list):
        texts = [texts]
    if not isinstance(models, list):
        models = [models]

    translated_texts = []

    for text in texts:
        translated_text = ""

        if isinstance(text, dict):
            max_bleu = 0
            weights = [1/4, 1/4, 1/4, 1/4]
            for model in models:
                response_text = translate_iterate(text["input"], model[0], model[1], config)
                bleu = bleu_score(response_text, text["output"])
                bleu = np.exp(weights[0] * np.log(bleu['BLEU1']) + weights[1] * np.log(bleu['BLEU2']) +
                              weights[2] * np.log(bleu['BLEU3']) + weights[3] * np.log(bleu['BLEU4']))
                if bleu > max_bleu:
                    max_bleu = bleu
                    translated_text = response_text

        elif isinstance(text, str):
            max_confidence = -float("inf")
            for model in models:
                response_text = translate_iterate(text, model[0], model[1], config)
                confidence = translate_with_conf(text, response_text, model[0], model[1], config)
                if confidence > max_confidence:
                    max_confidence = confidence
                    translated_text = response_text

        else:
            raise ValueError("Wrong texts.")

        translated_texts.append(translated_text)

    return translated_texts

