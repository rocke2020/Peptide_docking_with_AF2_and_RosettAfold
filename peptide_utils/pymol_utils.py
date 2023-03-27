import os, sys, re
sys.path.append(os.path.abspath('.'))
from utils.log_util import logger
import pandas as pd
from pymol import cmd
from peptide_utils.af_utils import parse_fasta, parse_pdb_id_chains
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=lambda _: str(_))


def process_native_pdb(pdb_file, pdb_id_chains, analysis_out_dir, around_dist=10):
    """ 
    pdb_id_chains: 2fmf_AB 
    """
    pdb_id, native_rec_chain, native_pep_chain = parse_pdb_id_chains(pdb_id_chains)
    cmd.reinitialize()
    cmd.load(pdb_file, 'native')

    # remove not receptor and peptide chains
    cmd.select('to_remove', f'native and not chain {native_rec_chain} and not chain {native_pep_chain}')
    cmd.remove('to_remove')
    cmd.remove('resn HOH')
    cmd.remove('hydrogens')

    # select peptide and receptor chain
    cmd.select("native_rec", f'native and chain {native_rec_chain}')
    cmd.select("native_pep", f'native and chain {native_pep_chain}')

    # select interface by first selecting receptor residues within 10A of peptide, then selecting peptide residues within 10A of receptor interface
    # merge the 2 selections by "+"
    cmd.select('interface_rec', f'byres native_rec within {around_dist} of native_pep')
    cmd.select('interface_native', f'interface_rec + byres native_pep within {around_dist} of interface_rec')

    # color receptor and their interface of native
    cmd.color('green', 'native_rec')
    cmd.color('orange', 'interface_rec')
    # ic(cmd.count_atoms('interface_rec'))

    cmd.color('red', 'native_pep')
    cmd.select('interface_pep', f'interface_native and chain {native_pep_chain}')
    cmd.color('blue', 'interface_pep')

    # cmd.show(representation='sticks', selection='native_pep')
    cmd.save(f'{analysis_out_dir}/{pdb_id_chains}_native.pse', format='pse')


def calc_rmsd_by_pymol_v2(
    pdb_file:str, pdb_id_chains:str, pdb_input_dir, analysis_out_dir, 
    process_native_only=False, out_model_name=False):
    """ ligand RMSD and interface RMSD, both only use backbone bRMSD
    Use pymol align to calculate rmsd

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
    1. The molecules you want to align need to be in two different objects. Else, PyMOL will answer with: ExecutiveAlign: invalid selections for alignment. You can skirt this problem by making a temporary object and aligning your original to the copy.
    2. By defaults, all states (like in NMR structures or trajectories) are considered, this might yield a bad or suboptimal alignment for a single state. Use the mobile_state and target_state argument to be explicit in such cases.
    3. Number of residues aligned is as max as the total shared residues, and maybe a little less.

    For peptide, align is a better metric than super to reflect the difference between short peptide structures

    For AlphaFold ppi output, the rec_chain is always A, and pep_chain is B.
    """
    process_native_pdb(pdb_file, pdb_id_chains, analysis_out_dir)
    if process_native_only:
        return

    rank_re = re.compile("rank_[0-9]")
    model_re = re.compile("model_[0-9]")
    metrics = []
    model_lst = sorted(pdb_input_dir.glob("_*.pdb"))
    for model_file in model_lst:
        orig_model_name = model_file.stem
        logger.debug('%s', orig_model_name)
        rank = int(str(rank_re.search(orig_model_name).group(0)).replace('rank_', ''))
        model_num = int(str(model_re.search(orig_model_name).group(0)).replace('model_', ''))
        model_name = f'{pdb_id_chains}_r{rank}_m{model_num}'

        cmd.load(str(model_file), model_name)
        cmd.select("afold_rec", f'{model_name} and chain A')
        cmd.select("afold_pep", f'{model_name} and chain B')
        metrics_per_model = [pdb_id_chains, rank, model_num]

        # ligand RMSD
        alignment_pep = cmd.align('afold_pep and backbone', 'native_pep and backbone')
        metrics_per_model.append(float("{0:.2f}".format(alignment_pep[0])))

        # select interface by first selecting receptor residues within 10A of peptide, then select peptide residues
        # within 10A of receptor interface, merge the 2 selections by "+"
        cmd.select('interface_rec_af', 'byres afold_rec within 10 of afold_pep')
        cmd.select('interface_af', 'interface_rec_af + byres afold_pep within 10 of interface_rec_af')
        # interface RMSD
        try:
            alignment_interface = cmd.align('interface_native and backbone', 'interface_af and backbone')
        except Exception as identifier:
            logger.exception(f'{model_name} has error in alignment_interface', exc_info=identifier)
            alignment_interface = [-1]
        
        metrics_per_model.append(float("{0:.2f}".format(alignment_interface[0])))

        cmd.color('brown', model_name)
        cmd.color('cyan', 'afold_pep')
        cmd.color('yellow', 'interface_af')
        metrics.append(metrics_per_model)

    cmd.set_name('native', pdb_id_chains)
    cmd.save(f'{analysis_out_dir}/{pdb_id_chains}.pse', format='pse')

    col_names = ['pdb_id_chains', 'rank', 'model_num', 'l_rmsd_bb', 'i_rmsd_bb']
    # to firstly view crucial info, put model name at the last when it is out
    if out_model_name:
        col_names.append(model_name)
    metrics_df = pd.DataFrame(metrics, columns=col_names)
    metrics_file = analysis_out_dir / (pdb_id_chains + '.csv')
    metrics_df.to_csv(metrics_file, index=False)


def get_pdb_pdb_seq(selection):
    """ 
    selection, e.g.
        f'native and chain {native_pep_chain}'
     """
    fasta = cmd.get_fastastr(selection=selection)
    seq_str = parse_fasta(fasta)[0][0]
    return seq_str


def calc_rmsd_by_pymol_v1(pdb_file:str, pdb_id_chains:str, pdb_input_dir, analysis_out_dir, process_native_only=False):
    """ old, use both pymol super and align. After comparison, align is better.

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

    For peptide, align is a better metric than super to reflect the difference between short peptide structures
    """
    process_native_pdb(pdb_file, pdb_id_chains, analysis_out_dir)
    if process_native_only:
        return

    metrics = []
    # load af models, select receptor and peptide
    model_lst = sorted(pdb_input_dir.glob("_*.pdb"))
    rank_re = re.compile("rank_[0-9]")
    model_re = re.compile("model_[0-9]")
    for model_file in model_lst:
        model_name = model_file.stem
        ic(model_name)
        rank = int(str(rank_re.search(model_name).group(0)).replace('rank_', ''))
        model_num = int(str(model_re.search(model_name).group(0)).replace('model_', ''))
        metrics_per_model = [model_name, pdb_id_chains, rank, model_num]

        model_name = f'{pdb_id_chains}_r{rank}_m{model_num}'
        cmd.load(str(model_file), model_name)
        cmd.select("afold_rec", f'{model_name} and chain A')
        cmd.select("afold_pep", f'{model_name} and chain B')

        # align peptide chains
        super_alignment_pep = cmd.super('afold_pep', 'native_pep')
        super_alignments = tuple([float("{0:.2f}".format(n)) for n in super_alignment_pep])
        # ic(super_alignments)
        metrics_per_model += super_alignments

        seq_alignment_pep = cmd.align('afold_pep', 'native_pep')
        alignments = tuple([float("{0:.2f}".format(n)) for n in seq_alignment_pep])
        # ic(alignments)
        metrics_per_model += alignments

        # align peptide chains backbone
        super_alignment_pep = cmd.super('afold_pep and backbone', 'native_pep and backbone')
        super_alignments = tuple([float("{0:.2f}".format(n)) for n in super_alignment_pep])
        # ic(super_alignments)
        metrics_per_model += super_alignments

        seq_alignment_pep = cmd.align('afold_pep and backbone', 'native_pep and backbone')
        alignments = tuple([float("{0:.2f}".format(n)) for n in seq_alignment_pep])
        # ic(alignments)
        metrics_per_model += alignments

        # super receptor chains
        super_alignment_rec = cmd.super('afold_rec', 'native_rec')
        alignments = tuple([float("{0:.2f}".format(n)) for n in super_alignment_rec])
        # ic(alignments)
        metrics_per_model += alignments

        # # save the superimposed structure
        # cmd.select('model_to_save', model_name)
        # super_filename=f'{model_name}_superimposed.pdb'
        # # ic(super_filename)
        # save_to_file = os.path.join(str(analysis_out_dir), super_filename)
        # cmd.save(save_to_file, model_name, format='pdb')

        # calculate rmsd-s
        seq_alignment_rec = cmd.align('afold_rec', 'native_rec')
        metrics_per_model += tuple([float("{0:.2f}".format(n)) for n in seq_alignment_rec])

        # super receptor chain backbones
        super_alignment_rec = cmd.super('afold_rec and backbone', 'native_rec and backbone')
        metrics_per_model += tuple([float("{0:.2f}".format(n)) for n in super_alignment_rec])

        # calculate rmsd by backbone
        seq_alignment_rec = cmd.align('afold_rec and backbone', 'native_rec and backbone')
        metrics_per_model += tuple([float("{0:.2f}".format(n)) for n in seq_alignment_rec])

        super_complex = cmd.super(model_name, 'native')
        metrics_per_model += tuple([float("{0:.2f}".format(n)) for n in super_complex])

        super_complex = cmd.super(f'{model_name} and backbone', 'native and backbone')
        metrics_per_model += tuple([float("{0:.2f}".format(n)) for n in super_complex])

        cmd.color('brown', model_name)
        # color receptor interface of model in yellow
        cmd.select('interface_rec_afold', 'byres afold_rec within 4 of afold_pep')
        cmd.color('yellow', 'interface_rec_afold')

        # color peptide of model in cyan
        cmd.color('cyan', 'afold_pep')
        # cmd.show(representation='sticks', selection='afold_pep')

        metrics.append(metrics_per_model)
        # break
    cmd.set_name('native', pdb_id_chains)
    cmd.save(f'{analysis_out_dir}/{pdb_id_chains}.pse', format='pse')

    col_names = create_columns_v1(full=True)
    # saving calculated metrics
    metrics_full_df = pd.DataFrame(metrics, columns=col_names)
    # metrics_full_file = analysis_out_dir / (pdb_id_chains + '_full.csv')
    # metrics_full_df.to_csv(metrics_full_file, index=False)

    metrics_df = metrics_full_df[create_columns_v1(full=False)]
    metrics_file = analysis_out_dir / (pdb_id_chains + '.csv')
    metrics_df.to_csv(metrics_file, index=False)


def create_columns_v1(full=False):
    """  """
    align_types = [
        '_super_pep', '_align_seq_pep', '_super_pep_bb', '_align_seq_pep_bb',
        '_super_rec', '_align_seq_rec', '_super_rec_bb', '_align_seq_rec_bb',
        '_complex', '_complex_bb']
    colnames_for_aln = [
        'rmsd_after_ref', 'num_aln_atoms_after_ref', 'ref_cycles', 'rmsd_before_ref', 'num_aln_atoms_before_ref',
        'raw_score', 'num_aln_residues'
        ]
    if full:
        col_names = ['model_name', 'pdb_id_chains', 'rank', 'model_num']
        for type in align_types:
            new_colnames = [s + type for s in colnames_for_aln]
            col_names = col_names + new_colnames
    else:
        col_names = ['pdb_id_chains', 'rank', 'model_num']
        rmsd_after_ref = 'rmsd_after_ref'
        for type in align_types:
            new_colname = rmsd_after_ref + type
            col_names.append(new_colname)
    return col_names