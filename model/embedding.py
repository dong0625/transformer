from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')

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
    def __init__(self, d_model):
        super().__init__()
        self.model = AutoModel.from_pretrained('distilbert-base-multilingual-cased').to(DEVICE)
        d_embed = self.model.config.hidden_size
        self.linear = nn.Linear(d_embed, d_model)

    def forward(self, inputs):
        outputs = self.model(**inputs).last_hidden_state
        out = self.linear(outputs)
        return out