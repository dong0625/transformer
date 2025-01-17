import numpy as np
import torch
from info import *

def test(model, tokenizer, valid_loader):
    model.eval()

    with torch.no_grad():
        src, tgt = next(iter(valid_loader))
        predictions, references = generate(model, tokenizer, src[0], tgt[0])
        print(predictions)
        print(references)

def generate(model, tokenizer, src, tgt):
    model.eval()

    with torch.no_grad():
        src = tokenizer.tokenize(src, D_LEN)
        tgt = tokenizer.tokenize(tgt, D_LEN + 1)['input_ids']
        src_tokens, src_mask = map(lambda x: x.to(DEVICE), src.values())
        tgt_mask = torch.zeros_like(src_mask).to(DEVICE)
        tgt_tokens = torch.tensor([[101] + [0]* (D_LEN - 1)]).to(DEVICE)

        for i in range(D_LEN - 1):
            tgt_mask[0, i] = 1
            out = model(src_tokens, src_mask, tgt_tokens, tgt_mask)[0][i].argmax().item()
            tgt_tokens[0, i + 1] = out
        
        predictions = tokenizer.decode(tgt_tokens.tolist())
        references = [tokenizer.decode(tgt.tolist())]
        return predictions, references