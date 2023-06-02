from itertools import combinations
from tqdm import tqdm

def generate_wildcards(trigram):
    # This function generates all possible wildcards for a given trigram
    n = len(trigram)
    wildcards = []
    for r in range(1, n+1):
        # Generate combinations of r elements
        for indices in combinations(range(n), r):
            wildcard = [None]*n
            for index in indices:
                wildcard[index] = trigram[index]
            wildcards.append(tuple(wildcard))
    return wildcards

def get_occurences_combi(seqs_tensor, trigram, min_j=20):
    # Get the length of trigram
    n = len(trigram)

    # Generate all possible wildcards for the trigram
    wildcard_list = generate_wildcards(trigram)

    # Sort wildcards by number of non-None elements (i.e., number of specified elements in the wildcard)
    wildcard_list.sort(key=lambda wildcard: wildcard.count(None))

    # Prepare an empty dictionary for storing the occurrences
    occurences = {wildcard: [] for wildcard in wildcard_list}
    occurences[trigram] = []

    # Now iterate over the seqs_tensor
    for i in tqdm(range(seqs_tensor.shape[0])):
        for j in range(min_j, 596): #seqs_tensor.shape[1] - n + 1):
            # Extract trigram from seqs_tensor
            seq_trigram = tuple(seqs_tensor[i, j:j+n].tolist())
            
            if seq_trigram == trigram:
                # If there's an exact match, append the location to the corresponding list in occurences
                occurences[trigram].append((i, j))
            else:
                # Compare this trigram with all possible wildcards, in increasing order of number of wildcards
                for wildcard in wildcard_list[1:]:
                    if all(w is None or w == n for w, n in zip(wildcard, seq_trigram)):
                        # If there's a match, append the location to the corresponding list in occurences and break the loop
                        occurences[wildcard].append((i, j))
                        break

    return occurences

def occ_to_seq(occ, T, tokenizer before=20, after=5, decode=False):
  occ_seq, occ_pos = occ
  seq = T[occ_seq][occ_pos-before:occ_pos+after]
  if decode:
    return tokenizer.decode(seq)
  else: return seq