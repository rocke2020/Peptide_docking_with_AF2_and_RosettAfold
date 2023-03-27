#!/bin/python3

# This script superimposes the models by their receptor and calculates the receptor and
# peptide RMSD both by superimposing and aligning by sequence. It also creates a PyMol
# session with the interface residues colored. Input id a director with a native and
# models that start with 'linker_removed'.
import os, sys
sys.path.append(os.path.abspath('.'))
from utils.log_util import logger
import pandas as pd
import os.path
import re
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=lambda _: str(_))
from pathlib import Path
from pandas import DataFrame
from peptide_utils.aa_utils import is_natural_only_supper
from peptide_utils.af_utils import get_hash
from peptide_utils.pymol_utils import calc_rmsd_by_pymol_v2
import pandas as pd
import os


# load native and select receptor and peptide
fasta_tab_file = 'Data/Source_Data/fasta_tab.csv'
fasta_df = pd.read_csv(fasta_tab_file)
out_root = Path('output')


def get_af_output_dir(pdb_id_chains, full_seq, version='2'):
    """  """
    hash_full_seq = get_hash(full_seq)
    out_dir = out_root / f'{pdb_id_chains}-{hash_full_seq}'
    if version == '':
        analysis_out_dir = out_dir / f'analysis_pymol'
    else:
        analysis_out_dir = out_dir / f'analysis_pymol_v{version}'
    analysis_out_dir.mkdir(exist_ok=1)
    return out_dir, analysis_out_dir


def check(pdb_id_chains, full_seq):
    """  """
    input_dir, analysis_out_dir = get_af_output_dir(pdb_id_chains, full_seq)
    metrics_file = analysis_out_dir / (pdb_id_chains + '.csv')
    metrics_df = pd.read_csv(metrics_file)

    # create column names
    align_types = [
        '_super_pep', '_align_seq_pep', '_super_pep_bb', '_align_seq_pep_bb',
        '_super_rec', '_align_seq_rec', '_super_rec_bb', '_align_seq_rec_bb',
        '_complex', '_complex_bb']
    rmsd_after_ref = 'rmsd_after_ref'

    # num_aln_residues = 'num_aln_residues'
    # num_aln_residues_df = metrics_df[['model_name', 'num_aln_residues_complex', 'num_aln_residues_complex_bb']]
    # num_aln_residues_df.to_csv('1.csv', index=False)

    for i, row in metrics_df.iterrows():
        rmsds = []
        super_vs_align_types = align_types[:8]
        for align_type in super_vs_align_types:
            column = f'{rmsd_after_ref}{align_type}'
            value = row[column]
            rmsds.append(value)
        for i in range(0, 8, 2):
            super_value = rmsds[i]
            align_value = rmsds[i+1]
            if super_value != align_value:
                logger.info(
                    'pdb_id_chains %s super_vs_align_types %s, super_value %s != align_value %s',
                    pdb_id_chains, super_vs_align_types[i: i+2], super_value, align_value)


def main(enable_check=0):
    """  """
    out_sum_file = f'Code/app/data/calc_sum.csv'
    good_rmsd_threshold = 2.5
    results = []
    for i, row in fasta_df.iterrows():
        pdb_id_chains, peptide_fasta, protein_fasta = row.values.tolist()
        # 2fmf_AB 1awr_CI 2h9m_CD 1jwg_BD
        # if pdb_id_chains != '1jwg_BD': continue

        pdb_id = pdb_id_chains.split('_')[0]
        pdb_file = f'data/pdb/pdb_files/{pdb_id}.pdb'
        full_seq = protein_fasta + ':' + peptide_fasta
        logger.info('%s', pdb_id_chains)
        merged_seq = protein_fasta + peptide_fasta
        if not is_natural_only_supper(merged_seq):
            logger.info('pdb_id_chains %s is invalid in seq', pdb_id_chains)
            # logger.info('%s', protein_fasta)
            # logger.info('%s', peptide_fasta)
            continue
        try:
            if not enable_check:
                pdb_input_dir, analysis_out_dir = get_af_output_dir(pdb_id_chains, full_seq, version='2')
                calc_rmsd_by_pymol_v2(pdb_file, pdb_id_chains, pdb_input_dir, analysis_out_dir, process_native_only=1)
            else:
                check(pdb_id_chains, full_seq)
        except Exception as identifier:
            logger.exception(
                'pdb_id_chains %s has sth wrong on calc_rmsd:\n %s', pdb_id_chains, identifier)

    if not results: return
    df = DataFrame(results, columns=['pdb_id_chains', 'model_name', 'min_avg_rmsd'])
    logger.info('%s', df)
    logger.info('\n%s', df.describe())
    df.to_csv(out_sum_file, index=False, sep=',')


if __name__ == "__main__":
    main(enable_check=0)
    logger.info('end')
    pass