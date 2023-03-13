from pathlib import Path
import pandas as pd
from pandas import DataFrame
import os, sys, re, json
sys.path.append(os.path.abspath('.'))
from utils.log_util import logger
from tqdm import tqdm
from peptide_utils.aa_utils import basic_aa_3chars_to_1chars, is_natural_only_supper, aminoacids
from biopandas.pdb import PandasPdb


pdb_files_dir = Path('data/pdb/pdb_files')
pdb_fasta_seqres_file = Path('data/pdb/pdb_seqres.txt')
organism_prefixes = ('ORGANISM_SCIENTIFIC', 'ORGANISM_COMMON')
human_organism_names = ('HOMO SAPIENS',)
HUMAN_ORGANISM = 'homo sapiens'
PDB_ID = 'pdb_id'
PEP_CHAIN = 'pep_chain'
PROT_CHAIN = 'prot_chain'
PEP_SEQ = 'pep_seq'
PROT_SEQ = 'prot_seq'
UNIPROT_ID = 'Uniprot_id'
PROTEIN_FAMILIES = 'protein_families'
SEQUENCE = 'Sequence'


def natural_aa_ratio(seq):
    """  """
    total_len = len(seq)
    count = 0
    for aa in seq:
        if aa in aminoacids:
            count += 1
    return count / total_len


def create_biopandas_pdb_seqs(pdb_id, chain):
    """ get pdb pdb seqs, filter not 3d residues. """
    native_file = str(pdb_files_dir / f'{pdb_id}.pdb')
    pdb = PandasPdb().read_pdb(native_file)
    sequence = pdb.amino3to1()
    sequence_list = list(sequence.loc[sequence['chain_id'] == chain, 'residue_name'])
    seq = ''.join(sequence_list)
    return seq


def get_index_with_seq_chain(pdb_id, seq_chain, items, chain):
    if chain == seq_chain:
        index = items[5]
    elif chain[0] == seq_chain:
        index = chain[1:]
    else:
        logger.info(f'{pdb_id} has abnormal chain at line {seq_chain}, items: {items}')
        raise Exception()
    return index


def get_all_downloaded_pdb_files(pdb_files_dir=pdb_files_dir):
    """  """
    all_downloaded_pdb_ids = []
    for file in pdb_files_dir.glob('*.pdb'):
        all_downloaded_pdb_ids.append(file.stem.lower())
    return all_downloaded_pdb_ids


def write_undownloaded_ids_to_file(un_downloaded_pdb_ids, out_file='data_process/batch_download_pdb/list_file.txt'):
    """  """
    out_str = ','.join(un_downloaded_pdb_ids) + '\n'
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(out_str)


def is_hunman_type_by_checking_pdb(pdb_id, pdb_files_dir=pdb_files_dir):
    """  """
    pdb_file = pdb_files_dir / f'{pdb_id}.pdb'
    organisms = []
    with open(pdb_file, 'r', encoding='utf-8') as f:
        for line in f:
            for organism_prefix in organism_prefixes:
                if organism_prefix in line:
                    items = line.strip().split(':')
                    organism_type = items[-1]
                    organisms.append(organism_type)
                    break
    for organism in organisms:
        for human_organism_name in human_organism_names:
            if human_organism_name in organism:
                return True
    return False


if __name__ == "__main__":
    pass