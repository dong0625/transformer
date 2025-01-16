import torch
import torch.nn as nn
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.Q_linear = nn.Linear(d_model, d_k * h)
        self.K_linear = nn.Linear(d_model, d_k * h)
        self.V_linear = nn.Linear(d_model, d_v * h)
        self.out_linear = nn.Linear(d_v * h, d_model)
    
    def forward(self, query, key, value, mask = None):
        batch_size = query.size(0)

        Q = self.Q_linear(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        K = self.K_linear(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = self.V_linear(value).view(batch_size, -1, self.h, self.d_v).transpose(1, 2)

        score = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k**0.5
        if (mask != None):
            mask = mask.unsqueeze(1).expand(-1, 8, -1, -1)
            score = score.masked_fill(mask == 0, -1e9)
        attention = score.softmax(dim=-1)
        out = torch.matmul(attention, V)

        out = out.transpose(1, 2).reshape(batch_size, -1, self.h * self.d_v)
        out = self.out_linear(out)
        return out

""" class Attention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.Q_linear = nn.Linear(self.d_model, d_k)
        self.K_linear = nn.Linear(self.d_model, d_k)
        self.V_linear = nn.Linear(self.d_model, d_v)

    def forward(self, query, key, value, mask = None):
        # Q, K, V 계산
        Q = self.Q_linear(query)  # Query: [batch_size, seq_len, d_k]
        K = self.K_linear(key)  # Key: [batch_size, seq_len, d_k]
        V = self.V_linear(value)  # Value: [batch_size, seq_len, d_v]

        #내적으로 가중치 구하기
        weight = torch.matmul(Q, torch.transpose(K, -2, -1)) #L * L

        #scaling
        weight = weight / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        #masking
        if (mask != None):
            weight.masked_fill_(mask == 0, -1e9)

        #weight sum
        weight = weight.softmax(dim=-1)
        out = torch.matmul(weight, V) #L * d_v

        return out """