from transformers import AutoTokenizer
import torch
import torch.nn as nn
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Tokenizer:
    def __init__(self, model_name = 'distilbert-base-multilingual-cased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, texts, length):
        inputs = self.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
        return inputs
    
    def decode(self, encoded_list):
        out = []
        for ids in encoded_list:
            if 102 in ids:
                ids = ids[:ids.index(102) + 1]
            out.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        return out

class Embedding(nn.Module):
    def __init__(self, d_len, d_model, d_voc):
        super().__init__()
        self.embedding = nn.Embedding(d_voc, 512, 0)
        self.linear = nn.Linear(512, d_model)

        def val(pos, i):
            res = pos / 10000**(2 * i / d_model)
            if i & 1:
                return math.cos(res)
            else:
                return math.sin(res)
        self.position = torch.tensor([[val(pos, i) for i in range(d_model)] for pos in range(d_len)]).to(DEVICE)

    def forward(self, tokens):
        out = self.embedding(tokens)
        out = self.linear(out)
        out = out + self.position
        return out