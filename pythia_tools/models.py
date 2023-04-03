import transformers
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
import transformer_lens
import transformer_lens.utils as utils
import numpy as np
from transformer_lens.hook_points import HookedRootModule, HookPoint

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_param_sizes_list():
    param_sizes = [7e7, 1.6e8, 4.1e8, 1e9]
    return param_sizes

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
    return tokenizer

def get_model(size, rev, lens=False):
    print('getmodel6')
    if lens:
        model = HookedTransformer.from_pretrained(f"pythia-{size}-deduped", checkpoint_index=rev)
    else:
        print('good')
        model = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/pythia-{size}-deduped", revision=f"step{rev}000", cache_dir=f"./pythia-{size}-deduped/step{rev}000")
    return model


        