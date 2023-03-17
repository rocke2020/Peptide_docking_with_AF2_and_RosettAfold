""" wrong codes to calculate rmsd, because not align adance! """

# script to compute the rmsd values residue-by-residue compared to the native complex.
# this script also computes the by residue LDDT, and outputs rmsd table, LDDT table and a comparison table between both.
# script assums running in the directory with the models and native
import os, sys
sys.path.append(os.path.abspath('.'))
from utils.log_util import logger
import colabfold.colabfold as cf
from biopandas.pdb import PandasPdb
import pandas as pd
import os.path
import pandas as pd
import Bio
from Bio import SeqIO
import pickle
import re
from Bio import pairwise2
import copy
import glob
import numpy as np
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=lambda _: str(_))
from pathlib import Path
from pandas import DataFrame
import dataclasses
from peptide_utils.cf_utils import check_seq


fasta_tab_file = 'Data/Source_Data/fasta_tab.csv'
fasta_df = pd.read_csv(fasta_tab_file)
out_root = Path('output')


def prepare_native(native_pdb_file, pdb_id_chains):
    native_dict = {}
    npdb = PandasPdb()
    npdb.read_pdb(native_pdb_file)
    native_df=npdb.df['ATOM']
    native_dict['native_df'] = native_df.copy(deep=True)
    native_chains=native_df.chain_id.unique()
    # ic(native_chains)
    rec_chain_native, pep_chain_native = list(pdb_id_chains.split('_')[1])
    ic(pdb_id_chains)
    native_dict['rec_chain_native'] = rec_chain_native
    native_dict['pep_chain_native'] = pep_chain_native

    rec_seq_native=npdb.amino3to1().query('chain_id == @rec_chain_native')
    pep_seq_native=npdb.amino3to1().query('chain_id == @pep_chain_native')
    rec_seq_native="".join(list(rec_seq_native.residue_name))
    pep_seq_native="".join(list(pep_seq_native.residue_name))
    # PDB pdb seq, not pdb fasta seq
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

def prepare_models_and_compute_distance(model_pdb, dict_of_native, analysis_out_dir, model_name):
    model_pdb = str(model_pdb)
    mpdb = PandasPdb()
    mpdb.read_pdb(model_pdb)
    model_df=mpdb.df['ATOM']

    rec_seq_model=mpdb.amino3to1().query('chain_id == "A"')
    pep_seq_model=mpdb.amino3to1().query('chain_id == "B"')
    # the seq is the input pdb fasta seq
    rec_seq_model="".join(list(rec_seq_model.residue_name))
    pep_seq_model="".join(list(pep_seq_model.residue_name))
    pep_alignment = pairwise2.align.globalxs(dict_of_native['pep_seq_native'], pep_seq_model, -3, -1)

    model_pep_df = model_df.query('chain_id == "B"')
    model_first_pos = model_pep_df.residue_number.min()

    native_offset = 0
    model_offset = 0
    rmsd_list = []
    locs_list = []
    pep_alignment_orig, pep_alignment_pred = pep_alignment[0][0], pep_alignment[0][1]
    logger.info('%s', pep_alignment_orig)
    logger.info('%s', pep_alignment_pred)
    logger.info('%s', dict_of_native['native_first_pos'])
    logger.info('%s', model_first_pos)
    for i in range(len(pep_alignment_pred)): # the string of the model peptide fasta
        if pep_alignment_orig[i]=="-":
            model_offset+=1
            rmsd_list.append(-1) # not computed
        elif pep_alignment_pred[i]=="-":
            native_offset+=1
            rmsd_list.append(-1)
        else:
            try:
                rmsd_val, locs_pair = by_residue_pair_dist(
                    native_offset, model_offset,
                    dict_of_native['native_pep_df'], dict_of_native['native_first_pos'],
                    model_pep_df, model_first_pos)
            except Exception as identifier:
                ic(model_name, native_offset, model_offset, pep_alignment_orig, pep_alignment_pred, i, pep_alignment_orig[i], pep_alignment_pred[i])
                raise identifier
            # rms_val, pos_pair = by_residue_pair_dist(native_offset, model_offset, dict_of_native['native_pep_df'],
                                #  dict_of_native['native_first_pos'], model_pep_df, model_first_pos)
            rmsd_list.append(rmsd_val)
            locs_list.append(locs_pair)

            model_offset+=1
            native_offset+=1
    pos_name = analysis_out_dir / f"{model_name}_position_list.txt"
    pd.Series(locs_list).to_csv(pos_name)
    lddt_series = model_df.query('chain_id == "B"').drop_duplicates(subset='residue_number').b_factor
    return pd.Series(rmsd_list), lddt_series, pep_seq_model


def by_residue_pair_dist(native_offset, model_offset, native_pep_df, native_first_pos, model_pep_df, model_first_pos):
    """ compute the offset of residue numbering between model and native structures """
    native_i_pos = native_first_pos + native_offset
    model_i_pos = model_first_pos + model_offset
    ic(native_i_pos, model_i_pos)
    filt_native_pep_df, filt_model_pep_df = intersect_res_dfs(
        native_pep_df.query('residue_number == @native_i_pos'),
        model_pep_df.query('residue_number == @model_i_pos')
    )
    # ic(filt_native_pep_df.columns)
    # ic(filt_native_pep_df.iloc[0])
    # ic(filt_native_pep_df.iloc[:10]['atom_name'])
    # ic(filt_model_pep_df.iloc[:10]['atom_name'])
    
    # ic(filt_model_pep_df.shape)
    # ic(filt_native_pep_df.shape)
    PandasPdb
    i_rmsd = PandasPdb.rmsd(
        filt_native_pep_df,
        filt_model_pep_df
    )
    one_based_model_offset = model_offset+1
    return i_rmsd, (native_i_pos, model_i_pos, one_based_model_offset)


# only keep the atoms that overlap in each compared residue (e.g. model has OXT and native doesnt)
def intersect_res_dfs(df1, df2):
    # ic(df1.shape)
    # ic(df2.shape)
    # ic(df1.atom_name)
    # ic(df2.atom_name)
    filt_df1 = df1[df1.atom_name.isin(df2.atom_name)].copy()
    filt_df2 = df2[df2.atom_name.isin(df1.atom_name)].copy()

    filt_df1.sort_values('atom_name',inplace=True)
    filt_df2.sort_values('atom_name',inplace=True)
    if filt_df1.shape[0] < filt_df2.shape[0]:
        filt_df1, filt_df2 = filt_df2, filt_df1
    if filt_df1.shape[0] > filt_df2.shape[0]:
        df1_atom_name = filt_df1['atom_name'].tolist()
        df2_atom_name = filt_df2['atom_name'].tolist()
        index_lst = []
        start_index = 0
        for atom_name in df2_atom_name:
            index = df1_atom_name.index(atom_name, start_index)
            index_lst.append(index)
            start_index = index + 1
        filt_df1 = filt_df1.iloc[index_lst]
        ic(filt_df1.iloc[:10]['atom_name'])
        ic(filt_df2.iloc[:10]['atom_name'])
    return filt_df1, filt_df2


def make_rmsd_lddt_tab(columns_lst, lddt_series, rms_series, model_name):
    col_series = pd.Series(columns_lst)
    col_series = col_series.reset_index()
    col_series.rename(columns={0:"seq"}, inplace=True)
    lddt_series = lddt_series.reset_index(drop=True)
    rms_series = rms_series.reset_index(drop=True)
    lddt_series.name = "lddt"
    rms_series.name = "rms"

    full=pd.concat([col_series,rms_series,lddt_series],axis=1)
    full = full[['seq','rms','lddt']]
    full['model'] = model_name
    return full


def get_output_dir(pdb_id_chains, full_seq):
    """  """
    hash_full_seq = cf.get_hash(full_seq)
    out_dir = out_root / f'{pdb_id_chains}-{hash_full_seq}'
    return out_dir


def calc_rmsd_lddt_for_peptide(pdb_file, input_pdb_id_chains, full_seq):
    native_dict = prepare_native(pdb_file, input_pdb_id_chains)

    rmsd_lst =[]
    ld_vs_rmsd_lst = []
    lddt_lst = []
    out_dir = get_output_dir(input_pdb_id_chains, full_seq)
    analysis_out_dir = out_dir / 'analysis'
    analysis_out_dir.mkdir(exist_ok=1)
    model_lst = list(out_dir.glob("rename*.pdb")) # replace the prefix with relevant model name
    ic(model_lst)

    model_names = []
    for i in range(len(model_lst)):
        model_stem = model_lst[i].stem
        first_model_occur = model_stem.find("model_")
        model_name = model_stem[first_model_occur:(first_model_occur+7)] #+7 e.g. the "model_5" part of the output name
        model_names.append(model_name)
        dict_copy = copy.deepcopy(native_dict)
        rmsd_series, lddt_series, pep_seq_model = prepare_models_and_compute_distance(
            model_lst[i], dict_copy, analysis_out_dir, model_name)
        columns_lst = [pep_seq_model[x] for x in range(len(pep_seq_model))]

        rmsd_series.name=model_name
        rmsd_lst.append(rmsd_series)
        # ic(rmsd_series)
        lddt_series.name=model_name
        lddt_lst.append(lddt_series)

        ld_vs_rmsd_lst.append(make_rmsd_lddt_tab(columns_lst, lddt_series, rmsd_series, model_name))

        del dict_copy
        # break
    ic(model_names)
    # pandas.core.frame.DataFrame
    by_resi_df = pd.concat(rmsd_lst,axis=1).transpose()
    by_resi_df.columns = columns_lst
    by_resi_df_file = analysis_out_dir / 'by_residue_rmsd.csv'
    by_resi_df.to_csv(by_resi_df_file,sep=",", index=False)

    # pandas.core.frame.DataFrame
    lddt_df = pd.concat(lddt_lst, axis=1).transpose()
    lddt_df.columns = columns_lst
    lddt_df_file = analysis_out_dir / 'lddt_by_residue.csv'
    lddt_df.to_csv(lddt_df_file, sep=",", index=False)

    rmsd_lddt_tab = pd.concat(ld_vs_rmsd_lst)
    rmsd_lddt_tab['pdb'] = input_pdb_id_chains
    rmsd_lddt_tab.to_csv((analysis_out_dir / 'rmsd_vs_lddt_comparison_by_residue.csv'), sep=",", index=False)
    min_values = select_best_model_by_least_rms(rmsd_lddt_tab)
    return min_values


def select_best_model_by_least_rms(rms_lddt_tab: DataFrame):
    """  """
    if isinstance(rms_lddt_tab, str):
        rms_lddt_tab = pd.read_csv(rms_lddt_tab)
    group = rms_lddt_tab.groupby(by=['model']).mean(numeric_only=True)
    df = group.reset_index()
    min_rms = min(df['rms'])
    #  model_name, min_avg_rms, plddt = min_values
    min_values = df[df['rms'] == min_rms].values.tolist()[0]
    return min_values


def main():
    """  """
    out_sum_file = f'Code/app/data/calc_sum.csv'
    good_rms_threshold = 2.5
    results = []
    for i, row in fasta_df.iterrows():
        pdb_id_chains, peptide_fasta, protein_fasta = row.values.tolist()
        # 2fmf_AB 1awr_CI 2h9m_CD
        if pdb_id_chains != '2fmf_AB': continue
        pdb_id = pdb_id_chains.split('_')[0]
        pdb_file = f'data/pdb/pdb_files/{pdb_id}.pdb'
        full_seq = protein_fasta + ':' + peptide_fasta
        logger.info('%s', protein_fasta)
        logger.info('%s', peptide_fasta)
        merged_seq = protein_fasta + peptide_fasta
        if not check_seq(merged_seq):
            logger.info('pdb_id_chains %s is invalid in seq', pdb_id_chains)
            continue
        try:
            min_values = calc_rmsd_lddt_for_peptide(pdb_file, pdb_id_chains, full_seq)
            min_values.insert(0, pdb_id_chains)
            results.append(min_values)
        except Exception as identifier:
            logger.exception(
                'pdb_id_chains %s has sth wrong on calc_rms_lddt_for_peptide:\n %s', pdb_id_chains, identifier)

    df = DataFrame(
        results, columns=['pdb_id_chains', 'model_name', 'min_avg_rmsd', 'plddt']
    )
    logger.info('%s', df)
    logger.info('\n%s', df.describe())
    df.to_csv(out_sum_file, index=False, sep=',')

if __name__ == "__main__":
    main()
    # ic(select_best_model_by_least_rms('output/0c31d4ef287c460d0518f6d4c3d8753e5cb0eece/analysis/rms_vs_lddt_comparison_by_residue.csv'))
    logger.info('end')
    pass