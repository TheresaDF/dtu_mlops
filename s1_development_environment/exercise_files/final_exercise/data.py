import torch
from torch.utils.data import DataLoader
from datasets import Dataset 


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    train = torch.randn(30000, 784)
    train_target = torch.randn(30000, 10)
    test = torch.randn(5000, 784)
    test_target = torch.randn(5000, 10)

    for i in range(6):
        train[i*5000:(i+1)*5000] = torch.load(f"../../../data/corruptmnist/train_images_{i}.pt").reshape(5000, 784)
        train_target[i*5000:(i+1)*5000] = torch.nn.functional.one_hot(torch.load(f"../../../data/corruptmnist/train_target_{i}.pt"), 10)

    test = torch.load("../../../data/corruptmnist/test_images.pt").reshape(5000, 784)
    test_target = torch.load("../../../data/corruptmnist/test_target.pt")

    d_train = list(zip(train, train_target))
    d_test = list(zip(test, test_target))

    return DataLoader(d_train, batch_size = 128, shuffle = True), DataLoader(d_test, batch_size = 128, shuffle = True)
