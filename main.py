import torch
from torch.utils.data import DataLoader

from info import *
from train import train
from test import test
from build import build_model

import evaluate
from scheduler import Scheduler
from model.embedding import Tokenizer
from data.data import Multi30K_train, Multi30K_valid

if __name__ == "__main__":
    tokenizer = Tokenizer()
    model = build_model(D_MODEL, D_LEN, D_K, D_V, D_FF, tokenizer.tokenizer.vocab_size, H, N, DEVICE)
    model.load_state_dict(torch.load("model_weights.pth"))

    train_loader = DataLoader(Multi30K_train(), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(Multi30K_valid(), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = Scheduler(optimizer, 512, 4000)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    bleu = evaluate.load("bleu").compute
    
    #train(model, tokenizer, train_loader, valid_loader, optimizer, criterion, scheduler, bleu)
    test(model, tokenizer, valid_loader)