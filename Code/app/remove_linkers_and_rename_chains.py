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


ic(sys.argv)
# pdb to process
pdb_file_to_read = sys.argv[1]

# table with fastas of the modeled pdbs
fasta_tab = sys.argv[2]

# pdb id of the model
i_pdb_id = sys.argv[3]

# is it a polyA run?
# polyA = str(sys.argv[4])

ppdb = PandasPdb()
ppdb.read_pdb(pdb_file_to_read)

ic(i_pdb_id)
merged_tab = pd.read_csv(fasta_tab)
ic(merged_tab.head())
pdb_id, pep_seq, prot_seq = merged_tab[merged_tab['pdb_id'] == i_pdb_id].values.tolist()[0]

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
receptor_end = df[(df['residue_number']>=linker_lst[0]) & (df['residue_number']<=linker_lst[1])].index[-1] + 2
ic(receptor_start)
ic(receptor_end)
df.loc[receptor_start: receptor_end,'chain_id']='A'

# change the peptide to chain B
pep_start = df[(df['residue_number']>=linker_lst[2]) & (df['residue_number']<=linker_lst[3])].index[0]
# include TER at the end
pep_end = df[(df['residue_number']>=linker_lst[2]) & (df['residue_number']<=linker_lst[3])].index[-1] + 2
ic(pep_start)
ic(pep_end)
df.loc[pep_start: pep_end,'chain_id']='B'

# remove the linker
# df.drop(df[(df['residue_number']<=linker_lst[2]) & (df['residue_number']>linker_lst[1])].index,axis=0,inplace=True)

# save modified pdb out
ppdb.df['ATOM']=df

## change TER chain id to B
def change_ter(row):
    """  """
    entry = row['entry']
    if row['record_name'] == 'TER':
        entry = entry.replace(' A ', ' B ')
    return entry

ic(ppdb.df.keys())
others = ppdb.df['OTHERS']
ic(others)  # DataFrame
# ic(others[others['record_name'] == 'TER'])
# others.iloc[1]['entry'] = ' 1297      ALA B 370'
# ic(others.iloc[1]['entry'])
# ic(type(others.iloc[1]['entry']))
# ic(len(others.iloc[1]['entry']))
# others['entry'] = others.apply(change_ter, axis=1)
others.at[1, 'entry'] = others.loc[1]['entry'].replace(' A ', ' B ')
ic(others)

out_file = Path(pdb_file_to_read).with_stem(f'rename_AB_chains_{Path(pdb_file_to_read).stem}')
ppdb.to_pdb(out_file)
ic('end')