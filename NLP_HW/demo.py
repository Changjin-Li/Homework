import torch
from utils import process_dataset, test


model_name_list = ['random', 'fasttext', 'glove', 'word2vec', 'bert']
model_name = model_name_list[4]
model_type = ["CNN", "RNN", "RNN_CNN"][1]


if model_name in model_name_list:
    if model_name == 'bert':
        from train_BERT import Model, Config
    else:
        from train import Model, Config
else:
    raise NotImplementedError


def demo():
    if "bert" in model_name:
        config = Config(net=model_type)
        model = Model(config)
        test_acc = model.test()
    else:
        config = Config("test", model_name, model_type)
        model = Model(config).to(config.device)
        model.load_state_dict(torch.load(config.save_path, weights_only=True))
        test_data = process_dataset(config, 'dataset/test.csv')
        test_acc = test(config, model, test_data)

    print(f"Model loaded from {config.save_path}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%.")


if __name__ == '__main__':
    demo()