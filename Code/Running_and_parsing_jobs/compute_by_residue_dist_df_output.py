#!/usr/bin/python3

# script to compute the rmsd values residue-by-residue compared to the native complex. 
# this script also computes the by residue LDDT, and outputs rmsd table, LDDT table and a comparison table between both. 
# script assums running in the directory with the models and native
import colabfold.colabfold as cf
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
import copy 
import glob
import numpy as np
sys.path.append(os.path.abspath('.'))
from utils.log_util import logger
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=lambda _: str(_))
from pathlib import Path


pdb_file = sys.argv[1]
logger.info('%s', pdb_file)
# 1awr_CI
input_pdb_id_chains = sys.argv[2]
fasta_tab = pd.read_csv('Data/Source_Data/fasta_tab.csv')
out_root = Path('output')


def prepare_native(native_pdb_file, pdb_id_chains):
    native_dict = {}
    npdb = PandasPdb()
    npdb.read_pdb(native_pdb_file)
    native_df=npdb.df['ATOM']
    native_dict['native_df'] = native_df.copy(deep=True)
    native_chains=native_df.chain_id.unique()
    ic(native_chains)
    rec_chain_native, pep_chain_native = list(pdb_id_chains.split('_')[1])
    ic(pdb_id_chains)
    ic(rec_chain_native)
    ic(pep_chain_native)
    native_dict['rec_chain_native'] = rec_chain_native
    native_dict['pep_chain_native'] = pep_chain_native
    
    rec_seq_native=npdb.amino3to1().query('chain_id == @rec_chain_native')
    pep_seq_native=npdb.amino3to1().query('chain_id == @pep_chain_native')
    rec_seq_native="".join(list(rec_seq_native.residue_name))
    pep_seq_native="".join(list(pep_seq_native.residue_name))
    ic(rec_seq_native)
    ic(pep_seq_native)
    
    native_pep_df = native_df.query('chain_id == @pep_chain_native')
    native_first_pos = native_pep_df.residue_number.min()
    native_dict['native_pep_df'] = native_pep_df.copy(deep=True)
    native_dict['native_first_pos'] = native_first_pos
    
    native_dict['rec_seq_native'] = rec_seq_native
    native_dict['pep_seq_native'] = pep_seq_native
    
    return native_dict 

# function to process and prepare each model into a dictionary of chains ids, sequences (rec and pep), lengths etc. 

def prepare_models_and_compute_distance(model_pdb, dict_of_native):
    model_pdb = str(model_pdb)
    mpdb = PandasPdb()
    mpdb.read_pdb(model_pdb)
    model_df=mpdb.df['ATOM']
    
    rec_seq_model=mpdb.amino3to1().query('chain_id == "A"')
    pep_seq_model=mpdb.amino3to1().query('chain_id == "B"')
    rec_seq_model="".join(list(rec_seq_model.residue_name))
    pep_seq_model="".join(list(pep_seq_model.residue_name))

    pep_alignment = pairwise2.align.globalxs(dict_of_native['pep_seq_native'], pep_seq_model, -3, -1)
    
    model_pep_df = model_df.query('chain_id == "B"')
    model_first_pos = model_pep_df.residue_number.min()
    
    native_offset = 0
    model_offset = 0
    rms_list = []
    pos_list = []
    for i in range(len(pep_alignment[0][1])): # the string of the model peptide fasta 
        if pep_alignment[0][0][i]=="-":
            model_offset+=1
            rms_list.append(-1) # not computed
        elif pep_alignment[0][1][i]=="-":
            native_offset+=1
            rms_list.append(-1)
        else:
            rms_val, pos_pair = by_residue_pair_dist(native_offset, model_offset, dict_of_native['native_pep_df'],
                                 dict_of_native['native_first_pos'], model_pep_df, model_first_pos)
            rms_list.append(rms_val)
            pos_list.append(pos_pair)

#            rms_list.append(by_residue_pair_dist(native_offset, model_offset, dict_of_native['native_pep_df'],
#                                 dict_of_native['native_first_pos'], model_pep_df, model_first_pos))
            model_offset+=1
            native_offset+=1
    pos_name = analysis_out_dir / "position_list.txt"
    pd.Series(pos_list).to_csv(pos_name)
    lddt_series = model_df.query('chain_id == "B"').drop_duplicates(subset='residue_number').b_factor
    return pd.Series(rms_list), lddt_series, pep_seq_model


def by_residue_pair_dist(native_offset, model_offset, native_pep_df, native_first_pos, model_pep_df, model_first_pos):
    """ compute the offset of residue numbering between model and native structures """
    native_i_pos = native_first_pos + native_offset
    model_i_pos = model_first_pos + model_offset
    filt_native_pep_df, filt_model_pep_df = intersect_res_dfs(
        native_pep_df.query('residue_number == @native_i_pos'),
        model_pep_df.query('residue_number == @model_i_pos')
    )
    i_rms = PandasPdb.rmsd(
    filt_native_pep_df,
    filt_model_pep_df
    )
    one_based_model_offset = model_offset+1
    return i_rms, (native_i_pos, model_i_pos, one_based_model_offset)


# only keep the atoms that overlap in each compared residue (e.g. model has OXT and native doesnt)
def intersect_res_dfs(df1, df2):
    filt_df1 = df1[df1.atom_name.isin(df2.atom_name)].copy()
    filt_df2 = df2[df2.atom_name.isin(df1.atom_name)].copy()
    
    filt_df1.sort_values('atom_name',inplace=True)
    filt_df2.sort_values('atom_name',inplace=True)
    return filt_df1, filt_df2


def make_rms_lddt_tab(columns_lst, lddt_series, rms_series, i_model):
    col_series = pd.Series(columns_lst)
    col_series = col_series.reset_index()
    col_series.rename(columns={0:"seq"}, inplace=True)
    lddt_series = lddt_series.reset_index()[i_model]

    rms_series = rms_series.reset_index()[i_model]
    lddt_series.name = "lddt"
    rms_series.name = "rms"


    full=pd.concat([col_series,rms_series,lddt_series],axis=1)
    full = full[['seq','rms','lddt']]
    full['model'] = i_model
    return full


def get_output_dir(pdb_id_chains):
    """  """
    pdb_id, pep_seq, prot_seq = fasta_tab[fasta_tab['pdb_id'] == pdb_id_chains].values.tolist()[0]
    full_seq = prot_seq + ':' + pep_seq
    hash_full_seq = cf.get_hash(full_seq)
    out_dir = out_root / hash_full_seq
    return out_dir


# Main - run the whole thing
native_dict = prepare_native(pdb_file, input_pdb_id_chains)
dict_copy = copy.deepcopy(native_dict)

series_lst =[]
ld_vs_rms_lst = []
lddt_lst = []
out_dir = get_output_dir(input_pdb_id_chains)
analysis_out_dir = out_dir / 'analysis'
analysis_out_dir.mkdir(exist_ok=1)
model_lst = list(out_dir.glob("rename*.pdb")) # replace the prefix with relevant model name
ic(model_lst)
for i in range(len(model_lst)):
    model_stem = model_lst[i].stem
    first_model_occur = model_stem.find("model_")
    row_name = model_stem[first_model_occur:(first_model_occur+7)] #+7 e.g. gets the "model_5" part of the output name
    ic(row_name)
    dict_copy = copy.deepcopy(native_dict)
    i_series, lddt_series, pep_seq_model = prepare_models_and_compute_distance(model_lst[i], dict_copy)
    columns_lst = [pep_seq_model[x] for x in range(len(pep_seq_model))]

    i_series.name=row_name
    series_lst.append(i_series)
    
    lddt_series.name=row_name
    lddt_lst.append(lddt_series)
    
    ld_vs_rms_lst.append(make_rms_lddt_tab(columns_lst,lddt_series,i_series,row_name))

    del dict_copy

# save outputs as csv
by_resi_df = pd.concat(series_lst,axis=1).transpose()
by_resi_df.columns = columns_lst
by_resi_df_file = analysis_out_dir / 'by_residue_rms.csv'
by_resi_df.to_csv(by_resi_df_file,sep=",", index=False)

lddt_df = pd.concat(lddt_lst, axis=1).transpose()
lddt_df.columns = columns_lst
lddt_df_file = analysis_out_dir / 'lddt_by_residue.csv'
lddt_df.to_csv(lddt_df_file, sep=",", index=False)

rms_lddt_tab = pd.concat(ld_vs_rms_lst)
rms_lddt_tab['pdb'] = input_pdb_id_chains
rms_lddt_tab.to_csv((analysis_out_dir / 'rms_vs_lddt_comparison_by_residue.csv'), sep=",", index=False)
logger.info('end')