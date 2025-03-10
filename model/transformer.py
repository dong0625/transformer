import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, embedding, encoder, decoder, d_model, d_voc, mask):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.out_linear = nn.Linear(d_model, d_voc)
        self.mask = mask

    def forward(self, src_tokens, src_mask, tgt_tokens, tgt_mask):
        src = self.embedding(src_tokens)
        tgt = self.embedding(tgt_tokens)

        src_tgt_mask = tgt_mask.unsqueeze(1) & src_mask.unsqueeze(2)
        src_mask = src_mask.unsqueeze(1) & src_mask.unsqueeze(2)
        tgt_mask = self.mask & tgt_mask.unsqueeze(1) & tgt_mask.unsqueeze(2)
        
        encoder_out = self.encoder(src, src_mask)
        out = self.decoder(tgt, encoder_out, tgt_mask, src_tgt_mask)
        out = self.out_linear(out)
        return out