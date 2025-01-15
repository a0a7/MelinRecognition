import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import Model
from dataloader import ShorthandGenerationDataset, data_split
from config import CONFIG

def collate_fn(batch):
    print(batch) # wyd
    imgs, labels = zip(*batch)
    imgs = pad_sequence(imgs, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    return imgs, labels

# Split the data
train_files, val_files, test_files, max_H, max_W, max_seq_length = data_split()

# Initialize dataset and dataloaders
train_dataset = ShorthandGenerationDataset(train_files, max_H, max_W, aug_types=9, max_label_leng=max_seq_length, channels=1)
val_dataset = ShorthandGenerationDataset(val_files, max_H, max_W, aug_types=1, max_label_leng=max_seq_length, channels=1)
test_dataset = ShorthandGenerationDataset(test_files, max_H, max_W, aug_types=1, max_label_leng=max_seq_length, channels=1)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Initialize model
config = CONFIG()
model = Model(max_H, max_W, config)

# Example training loop
num_epochs = 10  # Define the number of epochs
for epoch in range(num_epochs):
    model.train()
    for imgs, labels in train_loader:
        # Training step
        pass

    model.eval()
    with torch.no_grad():
        for imgs, labels in val_loader:
            # Validation step
            pass