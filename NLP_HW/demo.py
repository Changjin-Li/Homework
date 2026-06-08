import torch
from utils import process_dataset, test


model_name_list = ['word2vec_model', 'glove_model_cnn', 'glove_model_rnn', 'bert_model']
model_name = model_name_list[2]

if model_name == 'word2vec_model':
    from word2vec_model import Model, Config
elif model_name == 'glove_model_cnn':
    from glove_model_cnn import Model, Config
elif model_name == 'glove_model_rnn':
    from glove_model_rnn import Model, Config
elif model_name == 'bert_model':
    from bert_model import Model, Config
else:
    raise NotImplementedError


def demo():
    if "bert_model" in model_name:
        config = Config()
        model = Model(config)
        test_acc = model.test()
    else:
        config = Config("test")
        model = Model(config).to(config.device)
        model.load_state_dict(torch.load(config.save_path, weights_only=True))
        test_data = process_dataset(config, 'dataset/test.csv')
        test_acc = test(config, model, test_data)

    print(f"Model loaded from {config.save_path}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%.")


if __name__ == '__main__':
    demo()