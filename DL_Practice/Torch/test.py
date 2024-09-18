import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.mps.is_available())
device = torch.device("mps")
x = torch.randn(3,3).to(device)
print(x)