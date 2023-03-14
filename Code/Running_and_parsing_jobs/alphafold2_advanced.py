# -*- coding: utf-8 -*-
"""AlphaFold2 running script based on AlphaFold2_advanced.ipynb from ColabFold repository

Original file is located at
    https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/beta/AlphaFold2_advanced.ipynb
See [ColabFold](https://github.com/sokrypton/ColabFold/) for other related notebooks by Sergey Ovchinnikov

"""
import os, sys
sys.path.append(os.path.abspath('.'))
import argparse
from utils.log_util import logger
os.environ['TF_FORCE_UNIFIED_MEMORY']='1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='2.0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH ']='true'
from peptide_utils.pdb_id_chain_to_fasta_seqs import get_pdb_fasta_seq_pair_from_dict
from pathlib import Path
import tensorflow as tf
import jax
from IPython.utils import io
import subprocess
logger.info(os.getcwd())
import colabfold.colabfold as cf
import sys
import pickle
from urllib import request
from concurrent import futures
import json
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np

from alphafold.model import model
from alphafold.model import config
from alphafold.model import data
from alphafold.data import parsers
from alphafold.data import pipeline
from alphafold.common import protein


#### Read arguments ####
parser = argparse.ArgumentParser(description='Call AlphaFold with advanced parameters')
group_input = parser.add_argument_group('Input')
group_output = parser.add_argument_group('Output')
group_msa = parser.add_argument_group('Control MSA')
group_model = parser.add_argument_group('Control model parameters')
group_relax = parser.add_argument_group('Relax parameters')

group_input.add_argument('-s', dest='sequence', action='store', help='The sequence(s) used for prediction. Use / to specify intra-protein chainbreaks (for trimming regions within protein). Use : to specify inter-protein chainbreaks (for modeling protein-protein hetero-complexes).', default=None)
group_input.add_argument('--fasta', dest='fasta', action='store', help='The sequences for prediction, one per', default=None)
group_input.add_argument('--pdb_id_chains', dest='pdb_id_chains', help='example 1awr_CI, used to create 2 sequences, if available, overwrite args.sequence', default='')
group_input.add_argument('--num_models', dest='num_models', type=int, default=5, help='Number of models.')
group_input.add_argument('--homooligomer', dest='homooligomer', type=str, default='1', help='Number of times to repeat the CONCATENATED sequence')

group_output.add_argument('--prefix', dest='prefix', help='Prefix for every file, e.g. complex name')
group_output.add_argument('--working_dir', dest='working_dir', default='.', help='Working directory')
group_output.add_argument('--dpi', dest='dpi', type=int, default=100, help='DPI of output figures')
group_output.add_argument('--save_pae_json', dest='save_pae_json', action='store_true', default=False, help='')
group_output.add_argument('--delete_files', dest='delete_files', action='store_true', default=False, help='Delete old files before prediction')

group_model.add_argument('--max_recycles', dest='max_recycles', type=int, default=9, help='Controls the maximum number of times the structure is fed back into the neural network for refinement.')
group_model.add_argument('--use_ptm', dest='use_ptm', action='store_true', default=False,
          help="Uses Deepmind's `ptm` finetuned model parameters to get PAE per structure. Disable to use the original model params.")
group_model.add_argument('--is_training', dest='is_training', action='store_true', default=False,
          help="enables the stochastic part of the model (dropout), when coupled with `num_samples` can be used to sample a diverse set of structures")
group_model.add_argument('--tol', dest='tol', type=int, default=0, help='Tolerance for deciding when to stop (CA-RMS between recycles)')
group_model.add_argument('--num_ensemble', dest='num_ensemble', type=int, default=1,
          help='The trunk of the network is run multiple times with different random choices for the MSA cluster centres. (`1`=`default`, `8`=`casp14 setting`)')
group_model.add_argument('--num_samples', dest='num_samples', type=int, default=1,
          help='Sets number of random_seeds to iterate through for each model.')
group_model.add_argument('--use_turbo', dest='use_turbo', action='store_true', default=True,
          help='Introduces a few modifications (compile once, swap params, adjust max_msa) to speedup and reduce memory requirements. Disable for default behavior.')

group_msa.add_argument('--enable_subsample_msa', dest='enable_subsample_msa', action='store_true', help='subsample msa to avoid OOM error, only select max 10000 msa sequences')
group_msa.add_argument('--just_msa', dest='just_msa', action='store_true', help='create MSA and exit')
group_msa.add_argument('--no_use_env', dest='no_use_env', action='store_true', help='use environmental sequences?')
group_msa.add_argument('--pair_msa', dest='pair_msa', action='store_true', help='pair msa for prokaryotic sequences')
group_msa.add_argument('--msa_method', dest='msa_method', type=str, default="mmseqs2", help='MSA method. [mmseqs2, single_sequence, custom_a3m, precomputed]')
group_msa.add_argument('--custom_a3m', dest='custom_a3m', type=str, help='In case msa_method=custom_a3m, this option is required')
group_msa.add_argument('--rank_by', dest='rank_by', type=str, default='pLDDT', help='specify metric to use for ranking models (For protein-protein complexes, we recommend pTMscore). [pLDDT, pTMscore]')
group_msa.add_argument('--max_msa', dest='max_msa', type=str, default='512:1024', help='defines: `max_msa_clusters:max_extra_msa` number of sequences to use. (Lowering will reduce GPU requirements, but may result in poor model quality.')
group_msa.add_argument('--precomputed', dest='precomputed', type=str,
          help='In case msa_method=precomputed, this option is required. If you have previously run this notebook and saved the results, you can skip MSA generating step by providing the previously generated  `prediction/msa.npz')
group_msa.add_argument('--cov', dest='cov', type=int, default=0, help="Filter to remove sequences that don't cover at least `cov` %% of query. (Set to `0` to disable all filtering.)")

group_relax.add_argument('--use_amber_relax', dest='use_amber_relax', action='store_true', default=False, help='Use amber relaxation - NOT YET SUPPORTED')
group_relax.add_argument('--relax_all', dest='relax_all', action='store_true', default=False,
          help='Amber-relax all models. Disable to only relax the top ranked model. (Note: no models are relaxed if `use_amber_relax` is disabled. - NOT YET SUPPORTED')
args = parser.parse_args()
logger.info(args)

if(args.msa_method=='custom_a3m' and args.custom_a3m is None):
  logger.info('If custom_a3m is set, a file for option --custom_a3m must be provided. See -h for more information. Exiting.')
  exit()
if(args.fasta==None and args.sequence is None):
  logger.info('Provide sequences for prediction either as sequences or in FASTA format. Exiting.')
  exit()
if(not args.fasta==None and not args.sequence is None):
  logger.info('Provide sequences for prediction EITHER as sequences or in FASTA format, but not both. Exiting.')
  exit()
if(args.msa_method=='precomputed' and args.precomputed is None):
  logger.info('If precomputed is set, a file for option --precomputed must be provided. See -h for more information. Exiting.')
  exit()
if(args.msa_method=='custom_a3m' and args.custom_a3m is None):
  logger.info('If custom_a3m is set, a file for option --custom_a3m must be provided. See -h for more information. Exiting.')
  exit()

rank_by = args.rank_by
max_msa = args.max_msa
max_msa_clusters, max_extra_msa = [int(x) for x in max_msa.split(":")]
just_msa = args.just_msa

use_amber_relax = args.use_amber_relax # NOT SUPPORTED, need more installation
use_ptm = args.use_ptm
max_recycles = args.max_recycles
num_models = args.num_models
tol = args.tol
num_ensemble = args.num_ensemble
num_samples = args.num_samples
pair_msa = args.pair_msa
enable_subsample_msa = args.enable_subsample_msa

use_turbo = args.use_turbo
relax_all = args.relax_all
save_pae_json = args.save_pae_json
dpi = args.dpi
cov = args.cov
msa_method = args.msa_method
homooligomer = args.homooligomer
is_training = args.is_training

if args.no_use_env == True:
	use_env = False
else:
	use_env = True

MIN_SEQUENCE_LENGTH = 16
MAX_SEQUENCE_LENGTH = 2500

##############################################################x
# prepare sequence
import re

if args.pdb_id_chains:
  sequences = get_pdb_fasta_seq_pair_from_dict(args.pdb_id_chains)
  sequence = ':'.join(sequences)
else:
  sequence = args.sequence
logger.info('sequence: %s', sequence)
sequence = re.sub("[^A-Z:/]", "", sequence.upper())
sequence = re.sub(":+", ":", sequence)
sequence = re.sub("/+", "/", sequence)

# define number of copies
homooligomer =  "1" #@param {type:"string"}
if len(homooligomer) == 0: homooligomer = "1"
homooligomer = re.sub("[^0-9:]", "", homooligomer)
homooligomers = [int(h) for h in homooligomer.split(":")]

ori_sequence = sequence
sequence = sequence.replace("/","").replace(":","")
seqs = ori_sequence.replace("/","").split(":")

# prepare homo-oligomeric sequence
if len(seqs) != len(homooligomers):
  if len(homooligomers) == 1:
    homooligomers = [homooligomers[0]] * len(seqs)
    homooligomer = ":".join([str(h) for h in homooligomers])
  else:
    while len(seqs) > len(homooligomers):
      homooligomers.append(1)
    homooligomers = homooligomers[:len(seqs)]
    homooligomer = ":".join([str(h) for h in homooligomers])
    logger.info("WARNING: Mismatch between number of breaks ':' in 'sequence' and 'homooligomer' definition")

full_sequence = "".join([s*h for s,h in zip(seqs,homooligomers)])

# prediction directory
ori_seq_hash = cf.get_hash(ori_sequence)
output_dir = Path(args.working_dir) / 'output' / ori_seq_hash
output_dir.mkdir(exist_ok=1, parents=1)

if args.prefix is None and args.pdb_id_chains is not None:
  args.prefix  = args.pdb_id_chains
ori_sequence_file = output_dir / f'{args.prefix}-ori_sequence.txt'
with open(ori_sequence_file, 'w', encoding='utf-8') as f:
    f.write(ori_sequence)
logger.info(f"working directory: {output_dir}")

# print out params
logger.info(f"homooligomer: '{homooligomer}'")
logger.info(f"total_length: '{len(full_sequence)}'")
logger.info(f"working_directory: '{output_dir}'")

#############
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      logger.info(gpu)
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    logger.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    logger.info(e)
############

if use_amber_relax:
  from alphafold.relax import relax
  from alphafold.relax import utils


aatypes = set('ACDEFGHIKLMNPQRSTVWY')  # 20 standard aatypes
if not set(full_sequence).issubset(aatypes):
  raise Exception(f'Input sequence contains non-amino acid letters: {set(sequence) - aatypes}. AlphaFold only supports 20 standard amino acids as inputs.')
if len(full_sequence) < MIN_SEQUENCE_LENGTH:
  raise Exception(f'Input sequence is too short: {len(full_sequence)} amino acids, while the minimum is {MIN_SEQUENCE_LENGTH}')
if len(full_sequence) > MAX_SEQUENCE_LENGTH:
  raise Exception(f'Input sequence is too long: {len(full_sequence)} amino acids, while the maximum is {MAX_SEQUENCE_LENGTH}. Please use the full AlphaFold system for long sequences.')

if len(full_sequence) > 1400:
  logger.info(f"WARNING: For a typical Google-Colab-GPU (16G) session, the max total length is ~1300 residues. You are at {len(full_sequence)}! Run Alphafold may crash.")

# tmp directory
os.makedirs('tmp', exist_ok=True)

# --- Search against genetic databases ---
out_seq_file = output_dir / 'target.fasta'
with open(out_seq_file, 'w') as f:
  f.write(f'>query\n{sequence}')
output_dir = str(output_dir)

# Run the search against chunks of genetic databases (since the genetic
# databases don't fit in Colab ramdisk).

if msa_method == "precomputed":
  logger.info("use precomputed pickled msa from previous run")
  with open(args.precomputed, "rb") as msa_pickle:
    msas_dict = pickle.load(msa_pickle)
  msas, deletion_matrices = (msas_dict[k] for k in ['msas', 'deletion_matrices'])

elif msa_method == "single_sequence":
  msas = [[sequence]]
  deletion_matrices = [[[0] * len(sequence)]]

elif msa_method == "custom_a3m":
  logger.info("use custom a3m")
  a3m_file = open(args.custom_a3m, "r")
  a3m_content = f.read()
  lines = a3m_content.splitlines()
  a3m_lines = []
  for line in lines:
    line = line.replace("\x00", "")
    if len(line) > 0 and not line.startswith('#'):
      a3m_lines.append(line)
  msa_obj = parsers.parse_a3m("\n".join(a3m_lines))
  msa, deletion_matrix = msa_obj.sequences, msa_obj.deletion_matrix  
  msas, deletion_matrices = [msa], [deletion_matrix]

  if len(msas[0][0]) != len(sequence):
    logger.info("ERROR: the length of msa does not match input sequence")

else:
  seqs = ori_sequence.replace('/', '').split(':')

  _blank_seq = ["-" * len(seq) for seq in seqs]
  _blank_mtx = [[0] * len(seq) for seq in seqs]


  def _pad(ns, vals, mode):
    if mode == "seq": _blank = _blank_seq.copy()
    if mode == "mtx": _blank = _blank_mtx.copy()
    if isinstance(ns, list):
      for n, val in zip(ns, vals): _blank[n] = val
    else:
      _blank[ns] = vals
    if mode == "seq": return "".join(_blank)
    if mode == "mtx": return sum(_blank, [])


  # gather msas
  msas, deletion_matrices = [], []
  if msa_method == "mmseqs2":
    features_prefix = os.path.join(output_dir, 'features')
    logger.info(f"running mmseqs2")
    A3M_LINES = cf.run_mmseqs2(seqs, features_prefix, filter=True)

  for n, seq in enumerate(seqs):
    # tmp directory
    prefix = cf.get_hash(seq)
    prefix = os.path.join('tmp', prefix)

    if msa_method == "mmseqs2":
      # run mmseqs2
      a3m_lines = A3M_LINES[n]
      msa_obj = parsers.parse_a3m(a3m_lines)
      msa, mtx = msa_obj.sequences, msa_obj.deletion_matrix
      deduped_msa = list(dict.fromkeys(msa))
      if n == 1:
        logger.info('peptide deduped msa, len %s, first 3:  %s', len(deduped_msa), deduped_msa[:3])
      else:
        logger.info('protein deduped msa, len %s, first 3:  %s', len(deduped_msa), deduped_msa[:3])
      msas_, mtxs_ = [msa], [mtx]

    # pad sequences
    for msa_, mtx_ in zip(msas_, mtxs_):
      msa, mtx = [sequence], [[0] * len(sequence)]
      for s, m in zip(msa_, mtx_):
        msa.append(_pad(n, s, "seq"))
        mtx.append(_pad(n, m, "mtx"))

      msas.append(msa)
      deletion_matrices.append(mtx)

# save MSA as pickle
with open(os.path.join(output_dir,"msa.pickle"), "wb") as output_file:
    pickle.dump({"msas":msas,"deletion_matrices":deletion_matrices}, output_file)


if just_msa:
    logger.info('MSA created, exiting...')
    exit()

if msa_method != "single_sequence" and cov > 0:
  # filter sequences that don't cover at least %
  msas, deletion_matrices = cf.cov_filter(msas, deletion_matrices, cov)

full_msa = []
for msa in msas: full_msa += msa

seq_lenghs = set([len(seq) for seq in full_msa])
assert len(seq_lenghs) == 1, logger.info('%s', seq_lenghs)

# deduplicate
deduped_full_msa = list(dict.fromkeys(full_msa))
total_msa_size = len(deduped_full_msa)
if msa_method == "mmseqs2":
  logger.info(f'\n{total_msa_size} de-duplicated Sequences Found in Total (after filtering)')
else:
  logger.info(f'\n{total_msa_size} Sequences Found in Total')

msa_arr = np.array([list(seq) for seq in deduped_full_msa])
num_alignments, num_res = msa_arr.shape
logger.info('num_alignments: %s, num_res: %s', num_alignments, num_res)

if num_alignments > 1:
  plt.figure(figsize=(8,5),dpi=100)
  plt.title("Sequence coverage")
  seqid = (np.array(list(sequence)) == msa_arr).mean(-1)
  seqid_sort = seqid.argsort() #[::-1]
  non_gaps = (msa_arr != "-").astype(float)
  non_gaps[non_gaps == 0] = np.nan
  plt.imshow(non_gaps[seqid_sort]*seqid[seqid_sort,None],
            interpolation='nearest', aspect='auto',
            cmap="rainbow_r", vmin=0, vmax=1, origin='lower',
            extent=(0, msa_arr.shape[1], 0, msa_arr.shape[0]))
  plt.plot((msa_arr != "-").sum(0), color='black')
  plt.xlim(0,msa_arr.shape[1])
  plt.ylim(0,msa_arr.shape[0])
  plt.colorbar(label="Sequence identity to query",)
  plt.xlabel("Positions")
  plt.ylabel("Sequences")
  plt.savefig(os.path.join(output_dir, "msa_coverage.png"), bbox_inches = 'tight', dpi=200)
  plt.show()

from string import ascii_uppercase

if use_ptm == False and rank_by == "pTMscore":
  logger.info("WARNING: models will be ranked by pLDDT, 'use_ptm' is needed to compute pTMscore")
  rank_by = "pLDDT"

#############################
# delete old files
#############################
if args.delete_files:
	for f in os.listdir(output_dir):
	  if "rank_" in f:
	    os.remove(os.path.join(output_dir, f))

#############################
# homooligomerize
#############################
lengths = [len(seq) for seq in seqs]
msas_mod, deletion_matrices_mod = cf.homooligomerize_heterooligomer(msas, deletion_matrices,
                                                                    lengths, homooligomers)
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

num_res = len(full_sequence)
feature_dict = {}
feature_dict.update(pipeline.make_sequence_features(full_sequence, 'test', num_res))
logger.info('len(msas_mod) %s', len(msas_mod))
logger.info('msas_mod[0] first 3: %s', msas_mod[0][:3])
logger.info('msas_mod[1] first 3: %s', msas_mod[1][:3])
msa_objects = []
for msa, deletion_matrix in zip(msas_mod, deletion_matrices_mod):
  msa_objects.append(parsers.Msa(msa, deletion_matrix, descriptions=[''] * len(msa)))
feature_dict.update(pipeline.make_msa_features(msa_objects))
if not use_turbo:
  feature_dict.update(_placeholder_template_feats(0, num_res))


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


"""
# set chain breaks, add big enough number to residue index to indicate chain breaks
03-14 10:32:11 alphafold2_advanced.py 441: [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71
  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89
  90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125
 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161
 162 163 164 165 166 167 168 169]
03-14 10:32:11 alphafold2_advanced.py 443: [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71
  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89
  90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125
 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161
 162 163 364 365 366 367 368 369]
"""

Ls = []
for seq, h in zip(ori_sequence.split(":"),homooligomers):
  Ls += [len(s) for s in seq.split("/")] * h
# logger.info('%s', Ls) # [164, 6]
Ls_plot = sum([[len(seq)]*h for seq, h in zip(seqs, homooligomers)], [])
# logger.info('%s', feature_dict['residue_index'])
feature_dict['residue_index'] = cf.chain_break(feature_dict['residue_index'], Ls)
# logger.info('%s', feature_dict['residue_index'])

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

model_names = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5'][:num_models]
total = len(model_names) * num_samples
if use_amber_relax:
  if relax_all: total += total
  else: total += 1

#######################################################################
# precompile model and recompile only if length changes
af_data_dir = '/mnt/sdc/af_data/'
if use_turbo:
  name = "model_5_ptm" if use_ptm else "model_5"
  N = len(feature_dict["msa"])
  L = len(feature_dict["residue_index"])
  compiled = (N, L, use_ptm, max_recycles, tol, num_ensemble, max_msa, is_training)
  if "COMPILED" in dir():
    if COMPILED != compiled: recompile = True
  else:
    recompile = True
  if recompile:
    cf.clear_mem("gpu")
    cfg = config.model_config(name)
    cfg.data.common.max_extra_msa = min(N, max_extra_msa)
    cfg.data.eval.max_msa_clusters = min(N, max_msa_clusters)
    cfg.data.common.num_recycle = max_recycles
    cfg.model.num_recycle = max_recycles
    cfg.model.recycle_tol = tol
    cfg.data.eval.num_ensemble = num_ensemble

    params = data.get_model_haiku_params(name, af_data_dir)
    model_runner = model.RunModel(cfg, params, is_training=is_training)
    COMPILED = compiled
    recompile = False
else:
  cf.clear_mem("gpu")
  recompile = True

# cleanup
if "outs" in dir(): del outs
outs = {}
cf.clear_mem("cpu")

#######################################################################
for num, model_name in enumerate(model_names):  # for each model
  name = model_name + "_ptm" if use_ptm else model_name

  # setup model and/or params
  params = data.get_model_haiku_params(name, af_data_dir)
  if use_turbo:
    for k in model_runner.params.keys():
      model_runner.params[k] = params[k]
  else:
    cfg = config.model_config(name)
    cfg.data.common.num_recycle = cfg.model.num_recycle = max_recycles
    cfg.model.recycle_tol = tol
    cfg.data.eval.num_ensemble = num_ensemble
    model_runner = model.RunModel(cfg, params, is_training=is_training)

  for seed in range(num_samples):  # for each seed -
    # predict
    key = f"{name}_seed_{seed}"

    if enable_subsample_msa:
      logger.info('Subsampling MSA')
      sampled_feats_dict = subsample_msa(feature_dict, random_seed=seed)
      processed_feature_dict = model_runner.process_features(sampled_feats_dict, random_seed=seed)
    else:
      processed_feature_dict = model_runner.process_features(feature_dict, random_seed=seed)

    # processed_feature_dict = model_runner.process_features(feature_dict, random_seed=seed)
    predicted, recycles = model_runner.predict(processed_feature_dict, random_seed=seed)
    # logger.info('%s', list(predicted.keys()))
    # predicted.keys: 
    # ['distogram', 'experimentally_resolved', 'masked_msa', 'mean_plddt', 'plddt', 'predicted_lddt', 
    # 'ranking_confidence', 'structure_module', 'tol']
    predicted_cpu = cf.to(predicted, "cpu")
    logger.info('%s', list(predicted_cpu.keys()))
    tolerance = predicted_cpu['tol']
    outs[key] = parse_results(predicted_cpu, processed_feature_dict)

    # report
    line = f"{key} recycles:{recycles} tol:{tolerance:.2f} pLDDT:{outs[key]['pLDDT']:.2f}"
    if use_ptm: line += f" pTMscore:{outs[key]['pTMscore']:.2f}"
    logger.info(line)

    # cleanup
    del processed_feature_dict, predicted_cpu

  if use_turbo:
    del params
  else:
    del params, model_runner, cfg
    cf.clear_mem("gpu")

# delete old files
    for f in os.listdir(output_dir):
      if "rank_" in f:
        os.remove(os.path.join(output_dir, f))

# Find the best model according to the mean rank_by
model_rank = list(outs.keys())
model_rank = [model_rank[i] for i in np.argsort([outs[x][rank_by] for x in model_rank])[::-1]]

# Write out the prediction
for n,key in enumerate(model_rank):
  prefix = f"{args.prefix}_rank_{n+1}_{key}_recycle_{max_recycles}"
  pred_output_path = os.path.join(output_dir,f'{prefix}_unrelaxed.pdb')

  pdb_lines = protein.to_pdb(outs[key]["unrelaxed_protein"])
  with open(pred_output_path, 'w') as f:
    f.write(pdb_lines)
  if use_amber_relax:
    logger.info(f'AMBER relaxation')
    if relax_all or n == 0:
      amber_relaxer = relax.AmberRelaxation(
          max_iterations=0,
          tolerance=2.39,
          stiffness=10.0,
          exclude_residues=[],
          max_outer_iterations=20)
      relaxed_pdb_lines, _, _ = amber_relaxer.process(prot=outs[key]["unrelaxed_protein"])
      pred_output_path = os.path.join(output_dir,f'{args.prefix}_{prefix}_relaxed.pdb')
      with open(pred_output_path, 'w') as f:
        f.write(relaxed_pdb_lines)

############################################################
logger.info(f"model rank based on {rank_by}")
for n, key in enumerate(model_rank):
  logger.info(f"rank_{n+1}_{key} {rank_by}:{outs[key][rank_by]:.2f}")
  if use_ptm and save_pae_json:
    pae = outs[key]["pae"]
    max_pae = pae.max()
    # Save pLDDT and predicted aligned error (if it exists)
    pae_output_path = os.path.join(output_dir,f'rank_{n+1}_{key}_pae.json')
    # Save predicted aligned error in the same format as the AF EMBL DB
    rounded_errors = np.round(np.asarray(pae), decimals=1)
    indices = np.indices((len(rounded_errors), len(rounded_errors))) + 1
    indices_1 = indices[0].flatten().tolist()
    indices_2 = indices[1].flatten().tolist()
    pae_data = {
        'residue1': indices_1,
        'residue2': indices_2,
        'distance': rounded_errors.flatten().tolist(),
        'max_predicted_aligned_error': max_pae.item()
    }
    with open(pae_output_path, 'w') as f:
      json.dump(pae_data, f, ensure_ascii=False, indent=4)

#@title Extra plots
if use_ptm:
  logger.info("predicted alignment error")
  cf.plot_paes([outs[k]["pae"] for k in model_rank],dpi=dpi)
  plt.savefig(os.path.join(output_dir,f'predicted_alignment_error.png'), bbox_inches = 'tight', dpi=np.maximum(200,dpi))
  plt.show()

logger.info("predicted contacts")
cf.plot_adjs([outs[k]["adj"] for k in model_rank],dpi=dpi)
plt.savefig(os.path.join(output_dir,f'predicted_contacts.png'), bbox_inches = 'tight', dpi=np.maximum(200,dpi))
plt.show()

logger.info("predicted distogram")
cf.plot_dists([outs[k]["dists"] for k in model_rank],dpi=dpi)
plt.savefig(os.path.join(output_dir,f'predicted_distogram.png'), bbox_inches = 'tight', dpi=np.maximum(200,dpi))
plt.show()

logger.info("predicted LDDT")
cf.plot_plddts([outs[k]["plddt"] for k in model_rank], Ls=Ls, dpi=dpi)
plt.savefig(os.path.join(output_dir,f'predicted_LDDT.png'), bbox_inches = 'tight', dpi=np.maximum(200,dpi))
plt.show()

# add settings file
settings_path = os.path.join(output_dir, "settings.txt")
with open(settings_path, "w") as text_file:
  text_file.write(f"sequence={ori_sequence}\n")
  text_file.write(f"msa_method={msa_method}\n")
  text_file.write(f"homooligomer={homooligomer}\n")
  text_file.write(f"pair_msa={pair_msa}\n")
  text_file.write(f"max_msa={max_msa}\n")
  text_file.write(f"cov={cov}\n")
  text_file.write(f"use_amber_relax={use_amber_relax}\n")
  text_file.write(f"use_turbo={use_turbo}\n")
  text_file.write(f"use_ptm={use_ptm}\n")
  text_file.write(f"rank_by={rank_by}\n")
  text_file.write(f"num_models={num_models}\n")
  text_file.write(f"subsample_msa={subsample_msa}\n")
  text_file.write(f"num_samples={num_samples}\n")
  text_file.write(f"num_ensemble={num_ensemble}\n")
  text_file.write(f"max_recycles={max_recycles}\n")
  text_file.write(f"tol={tol}\n")
  text_file.write(f"is_training={is_training}\n")
  text_file.write(f"use_templates=False\n")
  text_file.write(f"-------------------------------------------------\n")

  for n,key in enumerate(model_rank):
    line = f"rank_{n+1}_{key} pLDDT:{outs[key]['pLDDT']:.2f}" + f" pTMscore:{outs[key]['pTMscore']:.4f}" if use_ptm else ""
    text_file.write(line+"\n")
