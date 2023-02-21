import torch
import torch.nn as nn

a = torch.arange(15).reshape(1,15)
b = torch.arange(15).reshape(1,15)
c = torch.arange(15).reshape(1,15)

print(torch.cat((a,b,c), dim=0))