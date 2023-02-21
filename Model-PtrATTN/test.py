import torch
a = torch.Tensor([1, 2, 3])
b = torch.Tensor([4, 5, 6])

mask = []
mask.append(torch.where(a!=2, torch.ones_like(a), torch.zeros_like(a)))
mask.append(torch.where(b!=5, torch.ones_like(b), torch.zeros_like(b)))
pr_mask = []
pr_mask.append(torch.cat(mask, dim=0))
pr_mask = torch.stack(pr_mask, dim=0)

print(pr_mask)