import torch
from src.datasets.cifar10 import get_cifar10, get_cifar10_augmented

def augmented_dataloader_from_string(
        dataset_name,
        n_samples = None,
        batch_size: int = 128,
        shuffle = True,
        seed: int = 0,
        download: bool = True,
        data_path: str = "../datasets",
        split_train_val_ratio = 0.9,
    ):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if dataset_name == "CIFAR-10":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_cifar10_augmented(
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path,
            split_train_val_ratio = split_train_val_ratio
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is not implemented")
    
    return train_loader, valid_loader


def dataloader_from_string(
        dataset_name,
        n_samples = None,
        batch_size: int = 128,
        shuffle = True,
        seed: int = 0,
        download: bool = False,
        data_path: str = "../datasets",
        split_train_val_ratio = 0.9,
    ):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if dataset_name == "CIFAR-10":
        classes = list(range(10))
        train_loader, valid_loader, test_loader = get_cifar10(
            batch_size = batch_size, 
            shuffle = shuffle,
            n_samples_per_class = int(n_samples/10) if n_samples is not None else None,
            classes = classes,
            seed = seed,
            download = download, 
            data_path = data_path,
            split_train_val_ratio = split_train_val_ratio
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is not implemented")
    
    return train_loader, valid_loader, test_loader