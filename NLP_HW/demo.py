import torch
from utils import process_dataset, test


model_name_list = ['self_trained_model', 'glove_model', 'bert_model', 'word2vec_model']
model_name = model_name_list[0]

if model_name == 'self_trained_model':
    from _self_trained_model import Model, Config
elif model_name == 'glove_model':
    from _glove_model import Model, Config
elif model_name == 'bert_model':
    from _bert_model import Model, Config
elif model_name == 'word2vec_model':
    from _word2vec_model import Model, Config
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