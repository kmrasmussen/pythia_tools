import transformers
import torch
import pandas
import numpy
import matplotlib.pyplot as plt
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import transformer_lens
import transformer_lens.utils as utils
import numpy as np
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np
from scipy.stats import entropy
from os.path import exists, join
#from umap import UMAP
from sklearn.metrics import adjusted_rand_score

def is_pair_earlier_in_X(X, t):
  earlier_occurrence = np.where((X[:t-1] == X[t]) & (X[1:t] == X[t+1]))
  return earlier_occurrence[0].size != 0
def is_element_earlier_in_X(X, t):
  earlier_occurrence = np.where(X[:t] == X[t+1])
  return earlier_occurrence[0].size != 0
def dec(tok, tokenizer):
  return tokenizer.decode(tok)
def empirical_entropy(arr):
  value, counts = np.unique(arr, return_counts=True)
  probs = counts / len(arr)
  return entropy(probs, base=2)
def mem_size(tensor):
  element_size = tensor.element_size() # size of one element in bytes
  num_elements = tensor.numel() # total number of elements in the tensor
  total_size = element_size * num_elements # total size of the tensor in bytes
  return total_size / (1024 ** 2)

def get_filtered_entries(new_lownat_entries, T):
  filtered_entries = []
  for entry_id, entry in enumerate(new_lownat_entries):
    seq_id = entry[0].item()
    pos = entry[1].item()
    if pos < 200 or pos > 400:
      continue
    if is_pair_earlier_in_X(T[seq_id], pos) == False and is_element_earlier_in_X(T[seq_id], pos) == True:
      filtered_entries.append(entry)
      #print(entry_id, ' - ', seq_id, pos)
  return torch.stack(filtered_entries)

def get_behavior(T, entries, idx, tokenizer):
  seq, pos = entries[idx][0].item(), entries[idx][1].item()
  print('Seq', seq, 'pos', pos) #, 'Induction: ', check_bigram_ind(T3[seq], pos) >= 0)
  #plot_entry_nats(T, entries, idx)
  #print(f'Nats: 410: {L410[seq,pos]:.4f}, 160: {L160[seq,pos]:.4f}, 70 {L70[seq,pos]:.4f}')
  print('---')
  print(tokenizer.decode(T[seq][pos-100:pos]))
  print('---')
  print("Current token:", tokenizer.decode(T[seq][pos]))
  print("Next (low-nat) token:", tokenizer.decode(T[seq][pos+1]))
  print('---')
  print(tokenizer.decode(T[seq][pos+2:pos+100]))

def get_grad_for_entry(model, T, seq, pos):
  model.zero_grad()
  input_ids = T[seq,:pos+2].to(device).reshape(1,-1)
  #print('T', input_ids[:,:10])
  outputs = model(input_ids, labels=input_ids)
  logits = outputs.logits[0, :-1, :]
  losses = torch.nn.functional.cross_entropy(logits, input_ids[0, 1:], reduction='none')
  losses[pos].backward()
  grad = torch.clone(model.gpt_neox.layers[3].mlp.dense_4h_to_h.weight.grad[:,0])
  del outputs
  torch.cuda.empty_cache()
  return grad

def get_ag_for_entry(model, T, seq, pos):
  model.zero_grad()
  input_ids = T[seq,:pos+2].to(device).reshape(1,-1)
  #print('T', input_ids[:,:10])
  outputs = model(input_ids, labels=input_ids)
  logits = outputs.logits[0, :-1, :]
  losses = torch.nn.functional.cross_entropy(logits, input_ids[0, 1:], reduction='none')
  losses[pos].backward()
  grad = torch.stack([cache[f'blocks.{i}.mlp.hook_post'].grad[-1,:] for i in range(24)]) #torch.clone(model.gpt_neox.layers[3].mlp.dense_4h_to_h.weight.grad[:,0])
  return grad

def qdg_eric_for_entries(model, T, filtered_entries, max_entries=None):
  filtered_grads = []
  for entry in tqdm(filtered_entries):
    seq, pos = entry[0], entry[1]
    grad = get_grad_for_entry(model, T, seq, pos)
    filtered_grads.append(grad)
    if max_entries is not None:
      if len(filtered_grads) >= max_entries:
        break
  return torch.stack(filtered_grads)

def get_similarity_matrix(A):
  norms = torch.norm(A, dim=1)
  dot_products = A @ A.t()
  B = dot_products / torch.ger(norms, norms)
  return B

def plot_entry_nats(T, entries, idx, param_sizes):
  seq, pos = entries[idx][0], entries[idx][1]
  nats = [L_s[seq, pos] for L_s in Ls]
  fig, ax = plt.subplots(figsize=(6, 1))
  ax.axhline(y=0.1, color='green', alpha=0.5)
  ax.axhline(y=0.2, color='green', alpha=0.5)
  ax.set_xlabel('Parameter size')
  ax.set_ylabel('Loss (nats)')
  ax.semilogx(param_sizes, nats)
  plt.show()



def qdg_ag_for_entry(model, T, seq, pos):
  model.zero_grad()
  input_ids = T[seq,:pos+2].reshape(1,-1)
  print(input_ids.shape)
  logits, cache = model.run_with_cache(input_ids) #, remove_batch_dim=True)
  print(logits.shape)
  #logits = logits[0, :-1, :]
  #print(logits.shape)
  losses = torch.nn.functional.cross_entropy(logits[0], input_ids[0], reduction='none')
  losses[pos].backward()
  graddot_mlp_layers = torch.stack([cache[f'blocks.{i}.mlp.hook_post'].grad[-1,:] for i in range(24)]) #torch.clone(model.gpt_neox.layers[3].mlp.dense_4h_to_h.weight.grad[:,0])
  torch.cuda.empty_cache()
  return graddot_mlp_layers

def qdg_dot_for_entry(model, T, seq, pos):
  model.zero_grad()
  input_ids = T[seq,:pos+2].reshape(1,-1)
  print(input_ids.shape)
  outputs = model(input_ids, labels=input_ids)
  logits = outputs.logits[0, :-1, :]
  losses = torch.nn.functional.cross_entropy(logits, input_ids[0, 1:], reduction='none')
  losses[pos].backward()
  graddot_mlp_layers = []
  for layer_i in range(model.config.num_hidden_layers):
    graddot_mlp_layer_i = (model.gpt_neox.layers[layer_i].mlp.dense_4h_to_h.weight * model.gpt_neox.layers[layer_i].mlp.dense_4h_to_h.weight.grad).sum(dim=0).detach()
    graddot_mlp_layers.append(graddot_mlp_layer_i)
  del outputs
  torch.cuda.empty_cache()
  return torch.stack(graddot_mlp_layers)

def qdg_dot_for_entries(model, T, filtered_entries, max_entries=None, func=qdg_dot_for_entry):
  filtered_grads = []
  for entry in tqdm(filtered_entries):
    seq, pos = entry[0], entry[1]
    grad = func(model, T, seq, pos)
    filtered_grads.append(grad)
    if max_entries is not None:
      if len(filtered_grads) >= max_entries:
        break
  return torch.stack(filtered_grads)