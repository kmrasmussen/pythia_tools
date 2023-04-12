import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np

def hist_and_box(data, title=None, save_filename=None):
    fig, ax = plt.subplots()
    ax.hist(data, bins='auto', edgecolor='black', alpha=0.7)
    ax.set_title(title)
    return fig

model_names = ['70m', '160m', '410m', '1b']

norms_dict = torch.load('/Users/kasperrasmussen/Documents/GitHub/pythia_tools/pythia_tools/apps/mlp_norms.pt')

st.title("Visualize Norms")

norm_type = st.selectbox("Select norm type:", ("rows", "columns"))
model_size = st.selectbox("Select model size:", model_names)
layer = st.slider("Select layer:", min_value=0, max_value=n_layers-1, step=1)

norms_data = norms_dict[norm_type][model_size][layer].cpu().numpy()

st.pyplot(hist_and_box(norms_data, title=f'L2 norms of {norm_type} in {model_size}-MLP{layer}'))
