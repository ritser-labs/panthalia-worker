import torch
from torch.utils.data import Dataset

class WikipediaDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, bos=True, eos=True)
        tokens = tokens[:self.seq_len] + [self.tokenizer.pad_id] * (self.seq_len - len(tokens))
        return torch.tensor(tokens)
