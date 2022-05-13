import torch


from monai.inferers import sliding_window_inference

x = torch.rand((2, 14, 32, 43, 54))

