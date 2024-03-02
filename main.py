import timm
import torch
import torchvision

a = torch.randint(0, 10, (10, ))
b = [2, 5]
c = a[b]
print(a)
print(b)
print(c)
