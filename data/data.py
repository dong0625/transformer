import torch
from torch.utils.data import Dataset
import gzip

def load_sentences(file_path):
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        sentences = f.readlines()
    return [sentence.strip() for sentence in sentences]

class Multi30K_train(Dataset):
    def __init__(self):
        self.src_sentences = load_sentences("./data/train.en.gz")
        self.tgt_sentences = load_sentences("./data/train.de.gz")

    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, index):
        return self.src_sentences[index], self.tgt_sentences[index]
    
class Multi30K_valid(Dataset):
    def __init__(self):
        self.src_sentences = load_sentences("./data/val.en.gz")
        self.tgt_sentences = load_sentences("./data/val.de.gz")

    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, index):
        return self.src_sentences[index], self.tgt_sentences[index]