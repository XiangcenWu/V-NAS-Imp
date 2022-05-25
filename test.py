import torch
import time

from V_NAS import Network

gpu_idx = 2
if torch.cuda.is_available():
    device = 'cuda:' + str(gpu_idx)
else:
    device = 'cpu'


model = Network(input=(96, 96, 96), out_channel=3).to(device)

time.sleep(30)
