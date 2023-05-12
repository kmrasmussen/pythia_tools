import torch.nn.functional as F
import torch

def normalize(x, dim=None):
  if len(x.shape) == 1:
    return F.normalize(x.reshape(1,-1)).reshape(-1)
  else:
    return F.normalize(x, dim=dim)
  
def mem_size(tensor):
  element_size = tensor.element_size() # size of one element in bytes
  num_elements = tensor.numel() # total number of elements in the tensor
  total_size = element_size * num_elements # total size of the tensor in bytes
  return total_size / (1024 ** 2)

def get_similarity_cross(A1, A2):
    norms_A1 = torch.norm(A1, dim=1)
    norms_A2 = torch.norm(A2, dim=1)
    
    dot_products = A1 @ A2.t()
    
    B = dot_products / torch.ger(norms_A1, norms_A2)
    
    return B