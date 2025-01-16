import torch
import torch.nn as nn
import copy

class Decoder(nn.Module):
    def __init__(self, decoder_block, n_layer):
        super().__init__()
        self.decoder_blocks = nn.ModuleList(copy.deepcopy(decoder_block) for _ in range(n_layer))
    
    def forward(self, x, context, casual_mask, cross_mask):
        out = x
        for block in self.decoder_blocks:
            out = block(out, context, casual_mask, cross_mask)
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, attention, feed_forward, norm):
        super().__init__()
        self.self_attention = copy.deepcopy(attention)
        self.cross_attention = copy.deepcopy(attention)
        self.feed_forward = copy.deepcopy(feed_forward)
        self.norm1 = copy.deepcopy(norm)
        self.norm2 = copy.deepcopy(norm)
        self.norm3 = copy.deepcopy(norm)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, context, casual_mask, cross_mask):
        out = x

        residual = out
        out = self.self_attention(out, out, out, casual_mask)
        out = self.dropout(out)
        out = self.norm1(out + residual)

        residual = out
        out = self.cross_attention(out, context, context, cross_mask)
        out = self.dropout(out)
        out = self.norm2(out + residual)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout(out)
        out = self.norm3(out + residual)
        
        return out