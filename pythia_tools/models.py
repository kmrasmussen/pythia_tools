import transformers
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
import transformer_lens
import transformer_lens.utils as utils
import numpy as np
from transformer_lens.hook_points import HookedRootModule, HookPoint

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_names = ['70m', '160m', '410m', '1b']

def get_param_sizes_list():
    param_sizes = [7e7, 1.6e8, 4.1e8, 1e9]
    return param_sizes

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    return tokenizer

def get_model(size, rev, lens=False):
    if lens:
        # transformerlens has rev 0 indexed, so we subtract 1
        model = HookedTransformer.from_pretrained(f"pythia-{size}-deduped", checkpoint_index=rev-1)
    else:
        model = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/pythia-{size}-deduped", revision=f"step{rev}000", cache_dir=f"./pythia-{size}-deduped/step{rev}000").to(device)
    return model

def yield_models(lens=False):
    for model_name in model_names:
        model = get_model(model_name, 143, lens)
        yield (model_name, model)

def get_n_layers(model):
    if isinstance(model, GPTNeoXForCausalLM):
        return model.config.num_hidden_layers
    else:
        return len(model.blocks)