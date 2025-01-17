import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=32
EPOCH = 30
D_MODEL = 512
D_LEN = 64
D_K = 64
D_V = 64
D_FF = 2048
H = 8
N = 6