import torch

class Scheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        lr = self.d_model ** -0.5 * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        return [lr] * len(self.base_lrs)