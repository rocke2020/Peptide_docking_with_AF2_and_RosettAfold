import re, random
from pathlib import Path
import json
import pandas as pd
import numpy as np
from pandas import DataFrame
import os, sys
from utils.log_util import logger
import jax
from alphafold.common import protein


MIN_SEQUENCE_LENGTH = 16
MAX_SEQUENCE_LENGTH = 2500
aatypes = set('ACDEFGHIKLMNPQRSTVWY')  # 20 standard aatypes


def check_seq(full_sequence):
  """  """
  if not set(full_sequence).issubset(aatypes):
    logger.exception(f'Input sequence contains non-amino acid letters: {set(full_sequence) - aatypes}. AlphaFold only supports 20 standard amino acids as inputs.')
    return 0
  if len(full_sequence) < MIN_SEQUENCE_LENGTH:
    logger.exception(f'Input sequence is too short: {len(full_sequence)} amino acids, while the minimum is {MIN_SEQUENCE_LENGTH}')
    return 0
  if len(full_sequence) > MAX_SEQUENCE_LENGTH:
    logger.exception(f'Input sequence is too long: {len(full_sequence)} amino acids, while the maximum is {MAX_SEQUENCE_LENGTH}. Please use the full AlphaFold system for long sequences.')
    return 0
  if len(full_sequence) > 1400:
    logger.exception(f"WARNING: For a typical Google-Colab-GPU (16G) session, the max total length is ~1300 residues. You are at {len(full_sequence)}! Run Alphafold may crash.")
  return 1

#############################
# define input features
#############################
def _placeholder_template_feats(num_templates_, num_res_):
  return {
      'template_aatype': np.zeros([num_templates_, num_res_, 22], np.float32),
      'template_all_atom_masks': np.zeros([num_templates_, num_res_, 37, 3], np.float32),
      'template_all_atom_positions': np.zeros([num_templates_, num_res_, 37], np.float32),
      'template_domain_names': np.zeros([num_templates_], np.float32),
      'template_sum_probs': np.zeros([num_templates_], np.float32),
  }


def subsample_msa(F, N=10000, random_seed=0):
  '''subsample msa to avoid running out of memory'''
  M = len(F["msa"])
  if N is not None and M > N:
    np.random.seed(random_seed)
    idx = np.append(0,np.random.permutation(np.arange(1,M)))[:N]
    F_ = {}
    F_["msa"] = F["msa"][idx]
    F_["deletion_matrix_int"] = F["deletion_matrix_int"][idx]
    F_["num_alignments"] = np.full_like(F["num_alignments"],N)
    for k in ['aatype', 'between_segment_residues',
              'domain_name', 'residue_index',
              'seq_length', 'sequence']:
              F_[k] = F[k]
    return F_
  else:
    return F


###########################
# run alphafold
###########################
def parse_results(prediction_result, processed_feature_dict):
  b_factors = prediction_result['plddt'][:,None] * prediction_result['structure_module']['final_atom_mask']
  dist_bins = jax.numpy.append(0,prediction_result["distogram"]["bin_edges"])
  dist_logits = prediction_result["distogram"]["logits"]
  out = {"unrelaxed_protein": protein.from_prediction(processed_feature_dict, prediction_result, b_factors=b_factors),
         "plddt": prediction_result['plddt'],
         "pLDDT": prediction_result['plddt'].mean(),
         "dists": dist_bins[dist_logits.argmax(-1)],
         "adj": jax.nn.softmax(dist_logits)[:,:,dist_bins < 8].sum(-1)}
  if "ptm" in prediction_result:
    out.update({"pae": prediction_result['predicted_aligned_error'],
                "pTMscore": prediction_result['ptm']})
  return out


def read_pdb_id_chains_lst():
    """  """
    fasta_tab = pd.read_csv('Data/Source_Data/fasta_tab.csv')
    return fasta_tab['pdb_id'].tolist(),  fasta_tab['protein_fasta'],  fasta_tab['peptide_fasta']