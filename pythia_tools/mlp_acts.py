import torch
import umap
from .qh import get_similarity_matrix
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from tqdm import tqdm

def get_sims_from_acts(mlp_acts):
    H = mlp_acts
    h_corr = torch.corrcoef(H)
    h_corr_p = torch.corrcoef(H.T)
    H_sim_n = get_similarity_matrix(H)
    H_sim_p = get_similarity_matrix(H.T)
    return H_sim_n, H_sim_p

def plot_scatter(H_umap, labels=None):
    plt.scatter(H_umap[:,0], H_umap[:,1], c=labels)
    plt.show()

def get_umap_from_sim(H_sim, plot=True):
    H_umap = UMAP(n_components=2).fit_transform((1 - H_sim).detach().cpu())
    if plot:
        plot_scatter(H_umap)
    return H_umap

def cluster_spectral_from_sim(H_sim, K):
    spectral = SpectralClustering(n_clusters=K)
    spectral.fit(H_sim.cpu())
    labels = spectral.labels_
    return labels

def get_mlp_act_entry(model, T, seq, pos, layer):
    input_ids = T[seq,:pos+2].reshape(1,-1)
    with torch.no_grad():
        logits, cache = model.run_with_cache(T[0,:])
    acts = torch.clone(cache[f'blocks.{layer}.mlp.hook_post'][0,pos,:])
    del logits
    del cache
    torch.cuda.empty_cache()
    return acts

def get_mlp_act_entry_all_layers(model, T, seq, pos):
    n_layers = len(model.blocks)
    input_ids = T[seq,:pos+2].reshape(1,-1)
    with torch.no_grad():
        logits, cache = model.run_with_cache(T[0,:])
    acts = torch.stack([torch.clone(cache[f'blocks.{layer}.mlp.hook_post'][0,pos,:]) for layer in range(n_layers)])
    del logits
    del cache
    torch.cuda.empty_cache()
    return acts.to('cpu')

def get_mlp_act_entries_all_layers(model, T, entries):
    return torch.stack([get_mlp_act_entry_all_layers(model, T, entry[0], entry[1]) for entry in tqdm(entries, position=0, leave=True)])

def get_mlp_act_entries(model, T, entries, layer):
    return torch.stack([get_mlp_act_entry(model, T, entry[0], entry[1], layer) for entry in tqdm(entries, position=0, leave=True)])