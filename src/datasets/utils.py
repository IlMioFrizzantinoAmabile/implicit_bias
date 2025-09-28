import torch
import numpy as np


def get_subset_data(data, targets, classes, n_samples_per_class=None, seed=0):
    np.random.seed(seed)
    targets = np.array(targets)
    idxs = []
    for target in classes:
        indices = np.where(targets == target)[0]
        if n_samples_per_class is None:
            # take all elements of that class
            idxs.append(indices)
        else:
            # subset only "n_samples_per_class" elements per class
            if n_samples_per_class>len(indices):
                raise ValueError(f"Class {target} has only {len(indices)} data, you are asking for {n_samples_per_class}.")
            idxs.append(np.random.choice(indices, n_samples_per_class, replace=False))
    idxs = np.concatenate(idxs).astype(int)
    targets = targets[idxs]

    clas_to_index = { c : i for i, c in enumerate(classes)}
    targets = np.array([clas_to_index[clas.item()] for clas in targets])
    data = data[idxs]
    return data, targets


def get_loader(
        dataset,
        batch_size = 128,
        split_train_val_ratio: float = 1.0,
        shuffle: bool = False,
        drop_last: bool = True,
        seed = 0,
        collate_fn = None
    ):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if split_train_val_ratio == 1.0:
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=0, #4, 
            pin_memory=False, 
            drop_last=drop_last,
            collate_fn = collate_fn
        )
    else:
        train_size = int(split_train_val_ratio * len(dataset))
        valid_size = len(dataset) - train_size
        dataset_train, dataset_valid = torch.utils.data.random_split(
            dataset, (train_size, valid_size), generator=torch.Generator().manual_seed(0)
        )
        return (
            torch.utils.data.DataLoader(
                dataset_train, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                num_workers=0, #4, 
                pin_memory=False, 
                drop_last=drop_last,
                collate_fn=collate_fn
            ),
            torch.utils.data.DataLoader(
                dataset_valid, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                num_workers=0, #4, 
                pin_memory=False, 
                drop_last=drop_last,
                collate_fn=collate_fn
            ),
        )

def get_output_dim(dataset_name):
    if dataset_name == "CIFAR-10":
        return 10