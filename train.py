import torch

from info import *

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train_per_epoch(model, tokenizer, train_loader, optimizer, criterion, scheduler):
    model.train()

    loss_sum = 0
    for idx, (src, tgt) in enumerate(tqdm(train_loader)):
        src = tokenizer.tokenize(src, D_LEN)
        tgt = tokenizer.tokenize(tgt, D_LEN + 1)
        src_tokens, src_mask = map(lambda x: x.to(DEVICE), src.values())
        tgt_tokens, tgt_mask = map(lambda x: x.to(DEVICE), tgt.values())

        optimizer.zero_grad()
        predictions = model(src_tokens, src_mask, tgt_tokens[:, :-1], tgt_mask[:, :-1]).permute(0, 2, 1)
        
        loss = criterion(predictions, tgt_tokens[:, 1:])
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_sum += loss.item()

    return loss_sum / len(train_loader)

def valid(model, tokenizer, valid_loader, bleu):
    model.eval()

    valid_bleu = 0
    with torch.no_grad():
        for idx, (src, tgt) in enumerate(valid_loader):
            src = tokenizer.tokenize(src, D_LEN)
            tgt = tokenizer.tokenize(tgt, D_LEN + 1)
            src_tokens, src_mask = map(lambda x: x.to(DEVICE), src.values())
            tgt_tokens, tgt_mask = map(lambda x: x.to(DEVICE), tgt.values())

            out = model(src_tokens, src_mask, tgt_tokens[:, :-1], tgt_mask[:, :-1])
            predictions = tokenizer.decode(out.argmax(dim=-1).tolist())
            references = [[S] for S in tokenizer.decode(tgt_tokens[:, 1:].tolist())]
            valid_bleu += bleu(predictions = predictions, references = references)['bleu']

    return valid_bleu / len(valid_loader)

def train(model, tokenizer, train_loader, valid_loader, optimizer, criterion, scheduler, bleu):
    writer = SummaryWriter()
    
    loss_min = float("inf")
    for epoch in range(EPOCH):
        loss = train_per_epoch(model, tokenizer, train_loader, optimizer, criterion, scheduler)
        writer.add_scalar("Loss/train", loss, epoch)

        valid_bleu = valid(model, tokenizer, valid_loader, bleu)
        writer.add_scalar("BLEU/valid", valid_bleu, epoch)

        if loss_min > loss:
            loss = loss_min
            torch.save(model.state_dict(), "model_weights.pth")