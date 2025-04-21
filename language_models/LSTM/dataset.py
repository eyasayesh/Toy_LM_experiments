import torch
from torch.utils.data import Dataset

class TokenDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len):
        """
        Args:
            text (str): Raw input text
            tokenizer (callable): Function that maps text to list of token IDs
            seq_len (int): Sequence length
        """
        self.seq_len = seq_len

        # Tokenize the entire corpus at once
        self.tokens = tokenizer(text)
        self.tokens = [t for t in self.tokens if t != -1]  # filter out any unknowns if needed

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)


    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.seq_len]
        y = self.tokens[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
