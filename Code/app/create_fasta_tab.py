import re, random
from pathlib import Path
import json
import pandas as pd
import numpy as np
from pandas import DataFrame
import os, sys
sys.path.append(os.path.abspath('.'))
from peptide_utils.pdb_id_chain_to_fasta_seqs import get_pdb_fasta_seq_pair_from_dict


input_file = Path('Data/Source_Data/polyA_fig3c.csv')
input_df = pd.read_csv(input_file)
pdb_ids = input_df['pdb_id'].tolist()
prot_seqs = []
pep_seqs = []

for pdb_id in pdb_ids:
    prot_seq, pep_seq = get_pdb_fasta_seq_pair_from_dict(pdb_id)
    prot_seqs.append(prot_seq)
    pep_seqs.append(pep_seq)

df = DataFrame({
    'pdb_id': pdb_ids,
    'peptide_fasta': pep_seqs,
    'protein_fasta': prot_seqs,
})
out_file = Path('Data/Source_Data/fasta_tab.csv')
df.to_csv(out_file, index=False, sep=',')
