import torch
from model.transformer import Transformer
from model.embedding import Embedding
from model.encoder import Encoder, EncoderBlock
from model.decoder import Decoder, DecoderBlock
from model.layer.feed_forward import FeedForward
from model.layer.multi_head_attention import MultiHeadAttention

def build_model(d_model, d_len, d_k, d_v, d_ff, d_voc, h, N, device):
    multi_head_attention = MultiHeadAttention(d_model, d_k, d_v, h)
    feed_forward = FeedForward(d_model, d_ff)
    norm = torch.nn.LayerNorm(d_model)
    encoder_block = EncoderBlock(multi_head_attention, feed_forward, norm)
    decoder_block = DecoderBlock(multi_head_attention, feed_forward, norm)
    encoder = Encoder(encoder_block, N)
    decoder = Decoder(decoder_block, N)
    embedding = Embedding(d_len, d_model, d_voc)
    mask = torch.tril(torch.ones((d_len, d_len), dtype=torch.bool)).unsqueeze(0).to(device)
    transformer = Transformer(embedding,
                              encoder, decoder, d_model, d_voc, mask).to(device)
    
    return transformer