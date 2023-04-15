import os
import zipfile
from .models import n_layers_dict

def create_zip_archive(input_dir, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, input_dir))

def make_md(section_name):
    with open("plots.md", "w") as f:
        for model_size, n_layers in n_layers_dict.items():
            for layer_id in range(n_layers):
                png_path = f'plots/{section_name}_{model_size}-{layer_id}.png'
                f.write(f"![Layer {layer_id} Plots]({png_path})\n\n")

