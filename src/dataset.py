import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split


class ConversationDataset(Dataset):
    def __init__(self, tokenizer, df, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = [self.construct_conv(row) for _, row in df.iterrows()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def construct_conv(self, row):
        flatten = lambda l: [item for sublist in l for item in sublist]
        conv = list(reversed([x + self.tokenizer.eos_token for x in row]))
        conv = ''.join(conv)
        return conv
    
    def collate(self, batch):
        return self.tokenizer.batch_encode_plus(batch, return_tensors='pt', 
                                                padding=True, truncation=True, max_length=self.max_len)

    
def get_dataloaders(tokenizer, df, max_len=512, batch_size=32, val_frac=0.1):
    dataset = ConversationDataset(tokenizer, df, max_len=max_len)
    n = len(dataset)
    v = int(n*val_frac)
    train_dataset, val_dataset = random_split(dataset, [n - v, v])
    print('train dataset has {} samples and val dataset has {} samples'.format(n-v, v))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12, collate_fn=dataset.collate)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=10, collate_fn=dataset.collate)
    return train_loader, val_loader