import zstandard as zstd
import json
import io
import pandas as pd
from os.path import join, exists
import torch

def extract_lines(input_path, output_path, num_lines, line_cap=None):
    # Create a zstandard decompression context
    dctx = zstd.ZstdDecompressor()

    data = []

    # Open the compressed input file
    with open(input_path, "rb") as input_file:
        # Create a decompression stream reader
        with dctx.stream_reader(input_file) as reader:
            # Wrap the reader with a buffered reader to read lines
            buffered_reader = io.BufferedReader(reader)

            # Open the output file for writing
            with open(output_path, "w") as output_file:
                # Read and process the specified number of lines
                for _ in range(num_lines):
                    line = buffered_reader.readline()
                    if not line:
                        break  # Stop reading if the end of the file is reached

                    # Load the JSON object from the line
                    json_object = json.loads(line)
                    if line_cap is not None: # and len(json_object) > line_cap:
                      print('capping')
                      json_object['text'] = json_object['text'][:line_cap]
                    else:
                      print('no line cap')
                    print(len(json_object['text']), json_object['meta'])

                    # Write the JSON object to the output file as a JSON line
                    output_file.write(json.dumps(json_object) + "\n")

def jsonl_to_dataframe(input_path):
    # Read the input .jsonl file
    data = []
    with open(input_path, "r") as input_file:
        for line in input_file:
            # Load the JSON object from the line
            json_object = json.loads(line)

            # Extract the text and meta fields
            text = json_object["text"]
            meta = json_object["meta"]['pile_set_name']

            # Append the data to the list
            data.append({"text": text, "meta": meta})

    # Create a pandas DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    return df

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_loss_mats(data_dir = 'pythia_tools/pythia_tools/data_files'):
    print(data_dir)
    hi = join(data_dir, 'losses_rev_140_model_pythia-70m-deduped.pt')
    print(hi)
    print(exists(hi))
    L70 = torch.load(join(data_dir, 'losses_rev_140_model_pythia-70m-deduped.pt')).float().to(device)
    L160 = torch.load(join(data_dir, 'losses_rev_140_model_pythia-160m-deduped.pt')).float().to(device)
    L410 = torch.load(join(data_dir, 'losses_rev_140_model_pythia-410m-deduped.pt')).float().to(device)
    L1B = torch.load(join(data_dir, 'losses_rev_140_model_pythia-1b-deduped.pt')).float().to(device)
    return [L70, L160, L410, L1B]

def get_token_mat(data_dir = 'pythia_tools/pythia_tools/data_files'):
    T = torch.load(join(data_dir, 'tokens_10691x600.pt')).to(device)
    print(T.shape)
    return T

def get_bigrams_dict(
        dict_dir = '/content/drive/MyDrive/thesis/data/bigrams/bigrams_dict_10k.pt',
        qset_dir = '/content/drive/MyDrive/thesis/data/bigrams/consequent_sets_bigrams_10k.pt'):
    bigrams_dict = torch.load(dict_dir)
    qset_dict = torch.load(qset_dir)
    return bigrams_dict, qset_dict

