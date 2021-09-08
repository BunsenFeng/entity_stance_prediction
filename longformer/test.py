import torch
a = torch.load('node_longformer.pt')
a = a.squeeze(1)
torch.save(a, 'new.pt')
b = torch.load('new.pt')
print(b.size())