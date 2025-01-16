import torch
import torch.nn as nn
import copy

class Encoder(nn.Module):
    def __init__(self, encoder_block, n_layer):
        super().__init__()
        self.encoder_blocks = nn.ModuleList(copy.deepcopy(encoder_block) for _ in range(n_layer))
    
    def forward(self, x, casual_mask):
        out = x
        for block in self.encoder_blocks:
            out = block(out, casual_mask)
        return out
    
class EncoderBlock(nn.Module):
    def __init__(self, attention, feed_forward, norm):
        super().__init__()
        self.self_attention = copy.deepcopy(attention)
        self.feed_forward = copy.deepcopy(feed_forward)
        self.norm1 = copy.deepcopy(norm)
        self.norm2 = copy.deepcopy(norm)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, casual_mask):
        out = x

        residual = out
        out = self.self_attention(out, out, out, casual_mask)
        out = self.dropout(out)
        out = self.norm1(out + residual)
        
        residual = out
        out = self.feed_forward(out)
        out = self.dropout(out)
        out = self.norm2(out + residual)

        return out