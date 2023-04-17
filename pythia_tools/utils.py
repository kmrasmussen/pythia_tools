import torch.nn.functional as F

def normalize(x, dim=None):
  if len(x.shape) == 1:
    return F.normalize(x.reshape(1,-1)).reshape(-1)
  else:
    return F.normalize(x, dim=dim)