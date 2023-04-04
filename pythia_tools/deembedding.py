import torch
from tuned_lens import TunedLens
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_svd_att_head(model, layer_idx, head_idx):
  n_dim = model.config.hidden_size
  n_heads = model.config.num_attention_heads
  head_dim = n_dim // n_heads
  
  # Get WV
  qkv_weight = model.gpt_neox.layers[layer_idx].attention.query_key_value.weight.detach()
  qkv_view = qkv_weight.view(3,n_dim, n_heads, n_dim // n_heads) # qkv, d, H, h
  value_matrix_index = 2
  WV = qkv_view[value_matrix_index, :, head_idx, :]
  WV2 = qkv_weight[2*n_dim:,head_idx*head_dim:(head_idx+1)*head_dim]
  #print(WV, WV2.shape, torch.allclose(WV,WV2))

  # Get WO
  att_out_weight = model.gpt_neox.layers[layer_idx].attention.dense.weight.detach()
  att_out_weight_view = att_out_weight.view(att_out_weight.shape[0], n_heads, n_dim // n_heads)
  WO = att_out_weight_view[:,head_idx,:]

  WOV = WV @ WO.T
  #print(WV.shape, WO.T.shape)
  assert WOV.shape == torch.Size([n_dim, n_dim])
  U, S, Vh = torch.linalg.svd(WOV.T, full_matrices=False)
  #U2, S2, Vh2 = torch.linalg.svd(WOV, full_matrices=False)
  Obias = model.gpt_neox.layers[-1].attention.dense.bias
  return U, S, Vh, WV, WO #U2, S2, Vh2,

def get_svd_mlp_out(model, layer_idx):
  U, S, Vh = torch.linalg.svd(model.gpt_neox.layers[layer_idx].mlp.dense_4h_to_h.weight, full_matrices=False)
  return U, S, Vh

def get_mlp_out(model, layer_idx):
  return model.gpt_neox.layers[layer_idx].mlp.dense_4h_to_h.weight

def decode_vocab_pmf(pmf, tokenizer, top_k=20):
  token_ids = torch.argsort(pmf,dim=-1, descending=True)[:top_k]
  words = [tokenizer.decode(token_id) for token_id in token_ids]
  #words.reverse()
  return ','.join(words)

def compute_stats(array):
  mean = torch.mean(array)
  diffs = array - mean
  var = torch.mean(torch.pow(diffs, 2.0))
  std = torch.pow(var, 0.5)
  zscores = diffs / std
  skews = torch.mean(torch.pow(zscores, 3.0))
  kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0 
  return kurtoses.item()

def display_column(col_i, layer_idx, make_plots=True):
  logit_lens_logits = tuned_lens.to_logits(col_i)
  tuned_lens_logits = tuned_lens(col_i, layer_idx)
  print(f'Mean logits, logit_lens: {logit_lens_logits.mean():.3f} tuned: {tuned_lens_logits.mean():.3f}')
  print(f'Std logits, logit_lens: {logit_lens_logits.std():.3f} tuned: {tuned_lens_logits.std():.3f}')
  print(f'Median logits, logit_lens: {logit_lens_logits.median():.3f} tuned: {tuned_lens_logits.median():.3f}')
  print(f'Kurtoses {compute_stats(logit_lens_logits):.3f} tuned: {compute_stats(tuned_lens_logits):.3f}')
  diff_means = (logit_lens_logits.mean() - tuned_lens_logits.mean()) ** 2.
  print(f'Means Diff: {diff_means:.3f}, dot: {(logit_lens_logits @ tuned_lens_logits):.3f}')
  if make_plots:
    plt.hist(logit_lens_logits.detach().cpu(),bins=50)
    plt.title('logit lens hist of logits')
    plt.figure(figsize=(6, 4))
    plt.show()
    plt.hist(tuned_lens_logits.detach().cpu(),bins=50)
    plt.title('tuned lens hist of logits')
    plt.show()
  print('LOGITLENS', decode_vocab_pmf(logit_lens_logits))
  print('TUNEDLENS', decode_vocab_pmf(tuned_lens_logits))
  return compute_stats(logit_lens_logits)

def display_columns(U, n_cols=30, make_plots=True):
  return_list = []
  for i in range(min(n_cols,U.shape[1])):
    col_i = U[:,i]
    print('---')
    print('column', i, 'of U ')
    display_column_out = display_column(col_i, i, make_plots=make_plots)
    print('out', display_column_out)
    return_list.append(display_column_out)
  return torch.tensor(return_list)