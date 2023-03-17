""" general alpha fold utils """
import hashlib
from typing import Tuple, List, Sequence


def get_hash(x):
  return hashlib.sha1(x.encode()).hexdigest()


def parse_fasta(fasta_string: str) -> Tuple[Sequence[str], Sequence[str]]:
  """Parses FASTA string and returns list of strings with amino-acid sequences.

  Arguments:
    fasta_string: The string contents of a FASTA file.

  Returns:
    A tuple of two lists:
    * A list of sequences.
    * A list of sequence descriptions taken from the comment lines. In the
      same order as the sequences.
  """
  sequences = []
  descriptions = []
  index = -1
  for line in fasta_string.splitlines():
    line = line.strip()
    if line.startswith('>'):
      index += 1
      descriptions.append(line[1:])  # Remove the '>' at the beginning.
      sequences.append('')
      continue
    elif not line:
      continue  # Skip blank lines.
    sequences[index] += line

  return sequences, descriptions


def parse_pdb_id_chains(pdb_id_chains):
    """  
    pdb_id_chains is '5qtu_A_H' or '5qtu_AH'
    """
    items = pdb_id_chains.split('_')
    pdb_id = items[0]
    if len(items) == 3:
        prot_chain, pep_chain = items[1], items[2]
    elif len(items) == 2:
        prot_chain, pep_chain = list(items[1])
    return pdb_id, prot_chain, pep_chain
    