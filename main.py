import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from info import *
from build import build_model
from model.embedding import Tokenizer
from data.data import Multi30K_train, Multi30K_valid
from tqdm import tqdm
import evaluate

writer = SummaryWriter()

class Scheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        lr = self.d_model ** -0.5 * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        return [lr] * len(self.base_lrs)

def train(model, tokenizer, train_loader, optimizer, criterion, scheduler):
    model.train()

    loss_sum = 0
    for idx, (src, tgt) in enumerate(tqdm(train_loader)):
        src_inputs = {key: value.to(DEVICE) for key, value in tokenizer.tokenize(src, D_LEN).items()}
        tgt = {key: value.to(DEVICE) for key, value in tokenizer.tokenize(tgt, D_LEN + 1).items()}
        tgt_inputs = {key: value[:, :-1] for key, value in tgt.items()}
        labels = tgt['input_ids'][:, 1:]

        optimizer.zero_grad()
        predictions = model(src_inputs, tgt_inputs).permute(0, 2, 1)
        
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_sum += loss.item()

    return loss_sum / len(train_loader)

def test(model, tokenizer, valid_loader, bleu):
    model.eval()

    bleu_sum = 0
    with torch.no_grad():
        for idx, (src, tgt) in enumerate(valid_loader):
            src_inputs = {key: value.to(DEVICE) for key, value in tokenizer.tokenize(src, D_LEN).items()}
            tgt = {key: value.to(DEVICE) for key, value in tokenizer.tokenize(tgt, D_LEN + 1).items()}
            tgt_inputs = {key: value[:, :-1] for key, value in tgt.items()}
            labels = tgt['input_ids'][:, 1:]

            out = model(src_inputs, tgt_inputs)
            predictions = tokenizer.decode(out.argmax(dim=-1).tolist())
            references = [[S] for S in tokenizer.decode(labels.tolist())]
            bleu_score = bleu(predictions = predictions, references = references)['bleu']

            bleu_sum += bleu_score

    return bleu_sum / len(valid_loader)

def main():
    tokenizer = Tokenizer()
    model = build_model(D_MODEL, D_LEN, D_K, D_V, D_FF, tokenizer.tokenizer.vocab_size, H, N, DEVICE)

    train_loader = DataLoader(Multi30K_train(), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(Multi30K_valid(), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = Scheduler(optimizer, 512, 4000)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    bleu = evaluate.load("bleu").compute
    
    loss_min = float("inf")
    for epoch in range(EPOCH):
        loss = train(model, tokenizer, train_loader, optimizer, criterion, scheduler)
        bleu_score = test(model, tokenizer, valid_loader, bleu)

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("BLEU/valid", bleu_score, epoch)

        if loss_min > loss:
            loss = loss_min
            torch.save(model.state_dict(), "model_weights.pth")

if __name__ == "__main__":
    main()