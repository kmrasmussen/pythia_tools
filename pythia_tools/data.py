import zstandard as zstd
import json
import io
import pandas as pd

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
