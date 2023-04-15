import torch
from tqdm import tqdm
from .models import get_n_layers

def get_many_pres_all_layers_decorr(model, seq_ids, T):
  pres = []
  for seq_id in tqdm(seq_ids):
    logits, cache = model.run_with_cache(T[seq_id], prepend_bos=False, remove_batch_dim=True)
    pre = [torch.clone(cache[f'blocks.{layer}.mlp.hook_pre'][-1]) for layer in range(get_n_layers(model))]
    del logits
    del cache
    torch.cuda.empty_cache()
    pres.append(torch.stack(pre))
  return torch.stack(pres) # seqs, layers, neurons

def get_many_pres_all_layers(model, seq_ids, T):
  pres = []
  for seq_id in tqdm(seq_ids):
    logits, cache = model.run_with_cache(T[seq_id], prepend_bos=False, remove_batch_dim=True)
    pre = [torch.clone(cache[f'blocks.{layer}.mlp.hook_pre']) for layer in range(get_n_layers(model))]
    del logits
    del cache
    torch.cuda.empty_cache()
    pres.append(torch.stack(pre))
    del pre
  ret = torch.stack(pres)
  del pres
  return ret # seqs, layers, position, neurons