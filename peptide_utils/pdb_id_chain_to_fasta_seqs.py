from pathlib import Path
import os, sys, re, json
sys.path.append(os.path.abspath('.'))
from utils.log_util import logger
from peptide_utils.pdb_utils import PDB_ID, PEP_CHAIN, PROT_CHAIN, PROT_SEQ, PEP_SEQ, pdb_fasta_seqres_file


pdb_fasta_seqres_dict_file = Path('data/pdb/pdb_seqres.json')
with open(pdb_fasta_seqres_dict_file, 'r', encoding='utf-8') as f:
    pdb_fasta_seqres_dict = json.load(f)


def get_pdb_fasta_seq_from_row_with_pep_chain(row):
    """  """
    pdb_id = row[PDB_ID]
    chain = row[PEP_CHAIN]
    return get_pdb_fasta_seq_from_dict(pdb_id, chain)


def get_pdb_fasta_seq_from_row_with_prot_chain(row):
    """  """
    pdb_id = row[PDB_ID]
    chain = row[PROT_CHAIN]
    return get_pdb_fasta_seq_from_dict(pdb_id, chain)


def get_pdb_fasta_seq_from_dict(pdb_id, chain):
    """  """
    pdb_id_chain = pdb_id.lower() + '_' + chain
    seq  = pdb_fasta_seqres_dict.get(pdb_id_chain, '')
    if seq == '':
        logger.info(f'pdb_id_chain {pdb_id_chain} has not fasta seq in dict')
    return seq


def get_pdb_fasta_seq_pair_from_dict(pdb_id_chains):
    """ pdb_id_chains: 1awr_CI """
    pdb_id, chains = pdb_id_chains.split('_')
    seq_a = get_pdb_fasta_seq_from_dict(pdb_id, chains[0])
    seq_b = get_pdb_fasta_seq_from_dict(pdb_id, chains[1])
    return seq_a, seq_b


def write_pdb_fasta_seqs_to_dict(check_id_chain='7kme_H'):
    """  """
    pdb_id_chain_seqs = {}
    pdb_id_line_head = re.compile(r'^>\d[\da-zA-Z]{3}_[\da-zA-Z]+')
    seq_count = 0
    with open(pdb_fasta_seqres_file,'r') as f:
        for line in f.readlines():
            line = line.strip()
            if pdb_id_line_head.search(line):
                pdb_id = line[1:5]
                items = line.split()
                assert items[1].startswith('mol:')
                chain = items[0][6:]
                seq_count = 0
                pdb_id_chain = pdb_id + '_' + chain
            else:
                seq_count += 1
                if seq_count > 1:
                    logger.error(f'Error line for continous {line}')
                    return
                pdb_id_chain_seqs[pdb_id_chain] = line
    seq = pdb_id_chain_seqs.get(check_id_chain)
    logger.info(f'{check_id_chain} seq is {seq}')
    with open(pdb_fasta_seqres_dict_file, 'w', encoding='utf-8') as f:
        json.dump(pdb_id_chain_seqs, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    write_pdb_fasta_seqs_to_dict()
