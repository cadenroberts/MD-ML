#!/usr/bin/env python3
import tqdm
import glob
import os
import numpy as np
import itertools
from moleculekit.molecule import Molecule
import argparse

from transformers import T5EncoderModel, T5Tokenizer #pyright: ignore[reportPrivateImportUsage]
import torch
import tqdm

def get_T5_model(device):
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    return model, tokenizer


def gen_T5_embeddings(molecule, model, tokenizer, device):
    # Generate the FASTA sequence for each chain (assumes 1 bead per residue)
    fasta = "".join([i[-1] for i in molecule.atomtype])
    fasta_list = []
    for i in [len(list(i[1])) for i in itertools.groupby(molecule.segid)]:
        fasta_list.append(fasta[:i])
        fasta = fasta[i:]

    embedding_list = []
    for fasta in fasta_list:
        fasta = " ".join(fasta)

        token_encoding = tokenizer(fasta, add_special_tokens=True)
        input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
        attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

        with torch.no_grad():
            # The model expects a input of shape [batch, max_len]
            embedding_repr = model(input_ids[None,:], attention_mask=attention_mask[None,:])
            # Output has shape [batch, max_len, embedding_len]
            # We also need to trim off the termination token, in the original script this was done with [:s_len] to
            # also trim off the batch padding
            embedding_list.append(embedding_repr.last_hidden_state[0][:-1])

    embedding = torch.cat(embedding_list).cpu().numpy() 
    return embedding



def main():
    parser = argparse.ArgumentParser(description="Generate T5 embeddings and put them into preprocessed directory")
    parser.add_argument("dir", help="Preprocessed data directory to add embeddings into")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using {}".format(device))
    
    model, tokenizer = get_T5_model(device)
    
    out_dir = args.dir
    for i in tqdm.tqdm(glob.glob(os.path.join(out_dir, "*"))):
        pdbid = os.path.basename(i)
        if not os.path.exists(os.path.join(out_dir, f"{pdbid}/raw/")):
            tqdm.tqdm.write(f"<skip> {pdbid}")
            continue
        # This assumes 1 the mapping uses bead per residue
        molecule = Molecule(os.path.join(out_dir, f"{pdbid}/processed/{pdbid}_processed.psf"))
        embedding = gen_T5_embeddings(molecule, model, tokenizer, device)

        outpath = os.path.join(out_dir, f"{pdbid}/raw/protT5_embedding.npy")
        np.save(outpath, embedding)

if __name__ == "__main__":
    main()
