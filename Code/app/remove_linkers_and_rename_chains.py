#!/usr/bin/env python
# coding: utf-8

## script to remove the linkers from af2 outputs that were modeled with a 30GLY linker. also renames the chains to be A for receptor and B for peptide

from biopandas.pdb import PandasPdb
import pandas as pd
import os
import os.path
import pandas as pd
import Bio
from Bio import SeqIO
import pickle
import re
import sys
from Bio import pairwise2
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=lambda _: str(_))
from pathlib import Path
import colabfold.colabfold as cf


fasta_tab_file = 'Data/Source_Data/fasta_tab.csv'
fasta_df = pd.read_csv(fasta_tab_file)


def rename_one_pdb_file_chain_AB(pdb_file, pdb_id_chains, pep_seq, prot_seq):
    """  
    Args:
        pdb_id_chains: 1awr_CI
    """
    ppdb = PandasPdb()
    ppdb.read_pdb(str(pdb_file))

    ic(pdb_id_chains)
    ic(prot_seq)
    ic(pep_seq)
    full_fasta = prot_seq + pep_seq
    ic(full_fasta)

    if len(pep_seq) > len(prot_seq):
        pep_seq, prot_seq = prot_seq, pep_seq #replace the fastas if for some reason peptide is first
    linker_lst=[]

    if 'X' in pep_seq:
        print('Xs in pep sequence')
        pep_seq = pep_seq.strip('X')

    if 'X' in prot_seq:
        print('Xs in prot sequence')
        prot_seq = prot_seq.strip('X')
        prot_seq = prot_seq.replace('X','')

    # if int(polyA)!=0:
    #     pep_seq = len(pep_seq)*'A'

    gap = 200
    # find the exact positions in the sequence where receptor, linker and peptide begin and end
    for match in re.finditer(prot_seq,full_fasta): # first and last positions of the receptor
        print('prot match start :', match.start())
        linker_lst.append(match.start())
        print('prot match end :', match.end())
        linker_lst.append(match.end())
        print('linker_lst after prot: ',linker_lst)

    for match in re.finditer(pep_seq,full_fasta): # first and last positions of the peptide
        pep_match_start = match.start() + gap
        pep_match_end = match.end() + gap
        print('pep match start :', pep_match_start)
        print('pep match end :',pep_match_end)
        if pep_match_start >= linker_lst[0] and pep_match_start <= linker_lst[1]:
            print('match embedded in receptor seq; this is not the peptide')
        else:
            linker_lst.append(pep_match_start)
            linker_lst.append(pep_match_end)

        print('linker_lst after prot and pep: ',linker_lst)

    # make a list of all these positions
    print('linker_lst includes: ', linker_lst)
    df=ppdb.df['ATOM']
    ic(df['residue_number'])
    # linker_lst.append(df[(df['residue_number']<linker_lst[2]) & (df['residue_number']>linker_lst[1])].index[0]) # first linker position, between rec and pep
    # linker_lst.append(df[(df['residue_number']<linker_lst[2]) & (df['residue_number']>linker_lst[1])].index[-1]) # second linker position, betweem rec and pep

    # linker_lst should look like - 
        # receptor_start, receptor_end, pep_start, pep_end, linker_start (index), linker_end (index)

    # change the receptor to chain A
    receptor_start = df[(df['residue_number']>=linker_lst[0]) & (df['residue_number']<=linker_lst[1])].index[0]
    receptor_end = df[(df['residue_number']>=linker_lst[0]) & (df['residue_number']<=linker_lst[1])].index[-1] + 1
    ic(receptor_start)
    ic(receptor_end)
    df.loc[receptor_start: receptor_end,'chain_id']='A'

    # change the peptide to chain B
    pep_start = df[(df['residue_number']>=linker_lst[2]) & (df['residue_number']<=linker_lst[3])].index[0]
    # include TER at the end
    pep_end = df[(df['residue_number']>=linker_lst[2]) & (df['residue_number']<=linker_lst[3])].index[-1] + 1
    ic(pep_start)
    ic(pep_end)
    df.loc[pep_start: pep_end,'chain_id']='B'

    # remove the linker
    # df.drop(df[(df['residue_number']<=linker_lst[2]) & (df['residue_number']>linker_lst[1])].index,axis=0,inplace=True)

    # save modified pdb out
    ppdb.df['ATOM']=df

    ic(ppdb.df.keys())
    others = ppdb.df['OTHERS']
    # ic(others)  # DataFrame
    others.at[1, 'entry'] = others.loc[1]['entry'].replace(' A ', ' B ')
    ic(others)

    out_file = pdb_file.with_stem(f'_{pdb_file.stem}')
    ppdb.to_pdb(out_file)


def batch_rename_chains():
    """  """
    for i, row in fasta_df.iterrows():
        pdb_id_chains, peptide_fasta, protein_fasta = row.values.tolist()
        # if pdb_id_chains != '1awr_CI': continue
        full_seq = protein_fasta+':'+peptide_fasta
        hash_full_seq = cf.get_hash(full_seq)
        out_pdb_dir =  Path('output') / f'{pdb_id_chains}-{hash_full_seq}'
        if out_pdb_dir.exists():
            for pdb_file in out_pdb_dir.glob('*.pdb'):
                if pdb_file.stem.startswith('rename'): continue
                rename_one_pdb_file_chain_AB(pdb_file, pdb_id_chains, peptide_fasta, protein_fasta)

def rename_out_dir():
    for i, row in fasta_df.iterrows():
        pdb_id_chains, peptide_fasta, protein_fasta = row.values.tolist()
        # if pdb_id_chains != '1awr_CI': continue
        full_seq = protein_fasta+':'+peptide_fasta
        hash_full_seq = cf.get_hash(full_seq)
        orig_out_pdb_dir =  Path('output') / hash_full_seq
        out_pdb_dir =  Path('output') / f'{pdb_id_chains}-{hash_full_seq}'
        if orig_out_pdb_dir.exists():
            os.rename(str(orig_out_pdb_dir), str(out_pdb_dir))


if __name__ == "__main__":
    batch_rename_chains()
    # rename_out_dir()
    ic('end')