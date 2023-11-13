import torch.nn.functional as F
import torch
import pickle

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

def get_kl_sim_cross(lst_A, lst_B):
  kl_loss = torch.nn.KLDivLoss(reduction="mean")
  kl_sim = torch.zeros(len(lst_A), len(lst_B))
  for i in range(len(lst_A)):
    for j in range(len(lst_B)):
      kl_sim[i,j] = kl_loss(F.log_softmax(lst_A[i],dim=0),F.softmax(lst_B[j],dim=0)).item()
  return kl_sim

def load_pkl(path):
  with open(path, 'rb') as file:
    # Load data from the file
    loaded_data = pickle.load(file)
  return loaded_data