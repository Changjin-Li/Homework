import torch
from glove_model_cnn import Model, Config
from utils import process_dataset, test


def demo():
    config = Config("test")

    test_data = process_dataset(config, 'dataset/test.csv')

    model = Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path, weights_only=True))
    print(f"Model loaded from {config.save_path}")

    test_acc = test(config, model, test_data)
    print(f"Test Accuracy: {test_acc * 100:.2f}%.")


if __name__ == '__main__':
    demo()