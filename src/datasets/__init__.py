from src.datasets.cifar10 import CIFAR10, get_cifar10, get_cifar10_augmented, get_cifar10_corrupted

#from src.datasets.all_datasets import get_train_loaders, get_test_loaders
from src.datasets.wrapper import dataloader_from_string, augmented_dataloader_from_string
from src.datasets.utils import get_output_dim