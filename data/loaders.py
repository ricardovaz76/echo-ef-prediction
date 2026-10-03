import torch
from torch.utils.data import DataLoader


# Builds the train/val/test DataLoaders
# batch_size is 1 since every video has a different number of frames
def build_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=1, num_workers=4):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# Computes the mean and std of the EF values from the train set to normalize
# the regression target during training, which helps with training stability.
#
# Reads the EF values straight from the dataset's sample list instead of
# iterating the DataLoader, so no videos have to be loaded.
def compute_ef_stats(dataset):
    all_ef = torch.tensor([ef for _, _, ef in dataset.samples], dtype=torch.float32)

    ef_mean = all_ef.mean().item()
    ef_std = all_ef.std().item()

    return ef_mean, ef_std
