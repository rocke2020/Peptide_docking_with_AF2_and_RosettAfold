#!/bin/python3

# This script superimposes the models by their receptor and calculates the receptor and
# peptide RMSD both by superimposing and aligning by sequence. It also creates a PyMol
# session with the interface residues colored. Input id a director with a native and
# models that start with 'linker_removed'.
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
from Code.Running_and_parsing_jobs.alphafold2_utils import check_seq
from pymol import cmd, stored
# import interfaceResidues
import pandas as pd
import os
# import Focus_alignment


# load native and select receptor and peptide
fasta_tab_file = 'Data/Source_Data/fasta_tab.csv'
fasta_df = pd.read_csv(fasta_tab_file)
out_root = Path('output')


def get_af_output_dir(pdb_id_chains, full_seq):
    """  """
    hash_full_seq = cf.get_hash(full_seq)
    out_dir = out_root / f'{pdb_id_chains}-{hash_full_seq}'
    return out_dir


def process_native_pdb(pdb_file, pdb_id_chains, analysis_out_dir):
    """ pdb_id_chains: 2fmf_AB """
    chains = pdb_id_chains.split('_', 1)[1]
    native_rec_chain, native_pep_chain = list(chains)
    cmd.load(pdb_file, 'native')

    # remove not receptor and peptide chains
    cmd.select('to_remove', f'native and not chain {native_rec_chain} and not chain {native_pep_chain}')
    cmd.remove('to_remove')
    cmd.remove('resn HOH')
    cmd.remove('hydrogens')

    # select peptide and receptor chain
    cmd.select("native_rec", f'native and chain {native_rec_chain}')
    cmd.select("native_pep", f'native and chain {native_pep_chain}')

    # select interface by first selecting receptor residues within 4A of peptide, then selecting peptide residues within 4A of receptor interface
    # merge the 2 selections by "+"
    cmd.select('interface_rec', 'byres native_rec within 4 of native_pep')
    cmd.select('interface_native', 'interface_rec + byres native_pep within 4 of interface_rec')

    # color receptor interface of native in yellow
    # cmd.select('interface_rec', f'interface_native and chain {native_rec_chain}')
    cmd.color('orange', 'interface_rec')
    # ic(cmd.count_atoms('interface_rec'))

    cmd.color('red', 'native_pep')
    # color peptide interface of native in cyan
    cmd.select('interface_pep', f'interface_native and chain {native_pep_chain}')
    cmd.color('blue', 'interface_pep')

    cmd.show(representation='sticks', selection='native_pep')
    cmd.save(f'{analysis_out_dir}/{pdb_id_chains}_native.pse', format='pse')


def calc_rmsd_by_pymol(pdb_file:str, pdb_id_chains:str, full_seq:str, process_native_only=False):
    """  
    Args:
        pdb_id_chains: 2fmf_AB
    
    align returns a list with 7 items:
        RMSD after refinement
        Number of aligned atoms after refinement
        Number of refinement cycles
        RMSD before refinement
        Number of aligned atoms before refinement
        Raw alignment score
        Number of residues aligned
    Notes:
    The molecules you want to align need to be in two different objects. Else, PyMOL will answer with: ExecutiveAlign: invalid selections for alignment. You can skirt this problem by making a temporary object and aligning your original to the copy.
    By defaults, all states (like in NMR structures or trajectories) are considered, this might yield a bad or suboptimal alignment for a single state. Use the mobile_state and target_state argument to be explicit in such cases.
    """
    input_dir = get_af_output_dir(pdb_id_chains, full_seq)
    analysis_out_dir = input_dir / 'analysis_pymol'
    analysis_out_dir.mkdir(exist_ok=1)    

    process_native_pdb(pdb_file, pdb_id_chains, analysis_out_dir)
    if process_native_only:
        return

    metrics = []
    rank_re = re.compile("rank_[0-9]")
    model_re = re.compile("model_[0-9]")

    # load af models, select receptor and peptide
    rec_chain = 'A'
    pep_chain = 'B'
    model_lst = list(input_dir.glob("_*.pdb"))

    for model_file in model_lst:
        model_name = model_file.stem
        # ic(model_name)
        rank = int(str(rank_re.search(model_name).group(0)).replace('rank_', ''))
        model_no = int(str(model_re.search(model_name).group(0)).replace('model_', ''))
        metrics_for_a_model = [model_name, pdb_id_chains, rec_chain, pep_chain, rank, model_no]

        cmd.load(str(model_file), model_name)
        cmd.select("afold_rec", f'{model_name} and chain A')
        cmd.select("afold_pep", f'{model_name} and chain B')

        # align peptide chains
        super_alignment_pep = cmd.super('afold_pep', 'native_pep')
        super_alignments = tuple([float("{0:.2f}".format(n)) for n in super_alignment_pep])
        # ic(super_alignments)
        metrics_for_a_model += super_alignments

        seq_alignment_pep = cmd.align('afold_pep', 'native_pep')
        alignments = tuple([float("{0:.2f}".format(n)) for n in seq_alignment_pep])
        # ic(alignments)
        metrics_for_a_model += alignments

        # align peptide chains backbone
        super_alignment_pep = cmd.super('afold_pep and backbone', 'native_pep and backbone')
        super_alignments = tuple([float("{0:.2f}".format(n)) for n in super_alignment_pep])
        # ic(super_alignments)
        metrics_for_a_model += super_alignments

        seq_alignment_pep = cmd.align('afold_pep and backbone', 'native_pep and backbone')
        alignments = tuple([float("{0:.2f}".format(n)) for n in seq_alignment_pep])
        # ic(alignments)
        metrics_for_a_model += alignments

        # super receptor chains
        super_alignment_rec = cmd.super('afold_rec', 'native_rec')
        alignments = tuple([float("{0:.2f}".format(n)) for n in super_alignment_rec])
        # ic(alignments)
        metrics_for_a_model += alignments

        # save the superimposed structure
        cmd.select('model_to_save', model_name)
        super_filename=f'{model_name}_superimposed.pdb'
        # ic(super_filename)
        save_to_file = os.path.join(str(analysis_out_dir), super_filename)
        cmd.save(save_to_file, model_name, format='pdb')

        # super receptor chain backbones
        super_alignment_rec = cmd.super('afold_rec and backbone', 'native_rec and backbone')
        metrics_for_a_model += tuple([float("{0:.2f}".format(n)) for n in super_alignment_rec])

        # calculate rmsd-s
        seq_alignment_rec = cmd.align('afold_rec', 'native_rec')
        metrics_for_a_model += tuple([float("{0:.2f}".format(n)) for n in seq_alignment_rec])

        # calculate rmsd by backbone
        seq_alignment_rec = cmd.align('afold_rec and backbone', 'native_rec and backbone')
        metrics_for_a_model += tuple([float("{0:.2f}".format(n)) for n in seq_alignment_rec])

        super_complex = cmd.super(model_name, 'native')
        metrics_for_a_model += tuple([float("{0:.2f}".format(n)) for n in super_complex])

        super_complex = cmd.super(f'{model_name} and backbone', 'native and backbone')
        metrics_for_a_model += tuple([float("{0:.2f}".format(n)) for n in super_complex])

        cmd.color('brown', model_name)
        # color receptor interface of model in yellow
        cmd.select('interface_rec_afold', 'byres afold_rec within 4 of afold_pep')
        cmd.color('yellow', 'interface_rec_afold')

        # color peptide of model in cyan
        cmd.color('cyan', 'afold_pep')
        cmd.show(representation='sticks', selection='afold_pep')

        metrics.append(metrics_for_a_model)
        # break
    cmd.set_name('native', pdb_id_chains)
    cmd.save(f'{analysis_out_dir}/{pdb_id_chains}.pse', format='pse')

    # create column names
    colnames = ['model_name', 'pdb_id', 'rec_chain', 'pep_chain', 'rank', 'model_no']
    colnames_for_aln = [
        'rmsd_after_ref', 'num_aln_atoms_after_ref', 'ref_cycles', 'rmsd_before_ref', 'num_aln_atoms_before_ref', 
        'raw_score', 'num_aln_residues']

    align_types = [
        '_super_pep', '_align_seq_pep', '_super_pep_bb', '_align_seq_pep_bb', 
        '_super_rec', '_super_rec_bb', '_align_seq_rec', '_align_seq_rec_bb', '_complex', '_complex_bb']
    for type in align_types:
        new_colnames = [s + type for s in colnames_for_aln]
        colnames = colnames + new_colnames

    # saving calculated metrics
    metrics_df = pd.DataFrame(metrics, columns = colnames)
    metrics_df.to_csv(analysis_out_dir / (pdb_id_chains + '.csv'), index=False)


def main():
    """  """
    out_sum_file = f'Code/app/data/calc_sum.csv'
    good_rmsd_threshold = 2.5
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
            calc_rmsd_by_pymol(pdb_file, pdb_id_chains, full_seq, process_native_only=0)
            # min_values = calc_rmsd_by_pymol(pdb_file, pdb_id_chains, full_seq)
            # min_values.insert(0, pdb_id_chains)
            # results.append(min_values)
        except Exception as identifier:
            logger.exception(
                'pdb_id_chains %s has sth wrong on calc_rmsd_lddt_for_peptide:\n %s', pdb_id_chains, identifier)

    if not results: return
    df = DataFrame(
        results, columns=['pdb_id_chains', 'model_name', 'min_avg_rmsd', 'plddt']
    )
    logger.info('%s', df)
    logger.info('\n%s', df.describe())
    df.to_csv(out_sum_file, index=False, sep=',')


if __name__ == "__main__":
    main()
    # ic(select_best_model_by_least_rms('output/0c31d4ef287c460d0518f6d4c3d8753e5cb0eece/analysis/rmsd_vs_lddt_comparison_by_residue.csv'))
    logger.info('end')
    pass