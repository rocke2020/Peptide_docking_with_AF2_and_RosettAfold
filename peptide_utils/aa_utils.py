from utils.log_util import logger


aminoacids = ["A", "C", "D", "E", "F", "G", "H", "I", "L", "M", "N", "P", "K", "Q", "R", "S", "T", "V", "W", "Y"]

basic_aa_1chars_to_3chars_lower = {
    "A": "Ala",
    "C": "Cys",
    "D": "Asp",
    "E": "Glu",
    "F": "Phe",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "L": "Leu",
    "M": "Met",
    "N": "Asn",
    "P": "Pro",
    "K": "Lys",
    "Q": "Gln",
    "R": "Arg",
    "S": "Ser",
    "T": "Thr",
    "V": "Val",
    "W": "Trp",
    "Y": "Tyr"
}

basic_aa_3chars_lower_to_1chars = {
    "Ala": "A",
    "Cys": "C",
    "Asp": "D",
    "Glu": "E",
    "Phe": "F",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Met": "M",
    "Asn": "N",
    "Pro": "P",
    "Lys": "K",
    "Gln": "Q",
    "Arg": "R",
    "Ser": "S",
    "Thr": "T",
    "Val": "V",
    "Trp": "W",
    "Tyr": "Y"
}

basic_aa_1chars_to_3chars = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "K": "LYS",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR"
}

basic_aa_3chars_to_1chars = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "LYS": "K",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y"
}

multi_chars_to_single_char_dict = {'Arg': 'R', 'His': 'H', 'Lys': 'K', 'Asp': 'D', 'Glu': 'E', 'Ser': 'S', 'Thr': 'T', 'Asn': 'N',
                    'Gln': 'Q', 'Cys': 'C', 'Sec': 'U', 'Gly': 'G', 'Pro': 'P', 'Ala': 'A', 'Ile': 'I', 'Leu': 'L',
                    'Met': 'M', 'Phe': 'F', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V', 'Dap': '1', 'Dab': '2',
                    'BOrn': '3', 'BLys': '4', 'Hyp': 'Z', 'Orn': 'O', 'bAla': '!', 'Gaba': '?', 'dDap': '5',
                    'dDab': '6',
                    'dBOrn': '7', 'dBLys': '8', 'dArg': 'r', 'dHis': 'h', 'dLys': 'k', 'dAsp': 'd', 'dGlu': 'e',
                    'dSer': 's',
                    'dThr': 't', 'dAsn': 'n', 'dGln': 'q', 'dCys': 'c', 'dSec': 'u', 'dGly': 'g', 'dPro': 'p',
                    'dAla': 'a',
                    'dIle': 'i', 'dLeu': 'l', 'dMet': 'm', 'dPhe': 'f', 'dTrp': 'w', 'dTyr': 'y', 'dVal': 'v',
                    'dHyp': 'z', 'dOrn': 'o', 'a5a': '=', 'a6a': '%', 'a7a': '$', 'a8a': '@', 'a9a': '#',
                    'Cys1': 'Ä', 'Cys2': 'Ö', 'Cys3': 'Ü', 'dCys1': 'ä', 'dCys2': 'ö', 'dCys3': 'ü',
                    'Ac': '&', 'NH2': '+', 'met': '-', 'cy': 'X'}

single_char_to_multi_chars_dict = {v: k for k, v in multi_chars_to_single_char_dict.items()}


def convert_3letters_to_1letters(seq):
    """translates from 3letters code to one symbol

    Arguments:
        seq {string} -- 3 letters code seq (e.g. Ala-Gly-Leu)

    Returns:
        string -- one letter symbol seq (e.g. AGL)
    """

    items = []
    seq = seq.split('-')
    for bb in seq:
        items.append(multi_chars_to_single_char_dict[bb])
    seq = ''.join(items)
    return seq


def convert_1letters_to_3letters(seq):
    """translates one symbol to three letters code

    Arguments:
        seq {string} -- one letter symbol seq (e.g. AGL)

    Returns:
        string -- 3 letters code seq (e.g. Ala-Gly-Leu)
    """

    items = []
    for bb in seq:
        items.append(single_char_to_multi_chars_dict[bb])
    seq = '-'.join(items)

    return seq


def is_natural_only_supper(seq):
    """ If the char is not in upper case, treat as not natural """
    if isinstance(seq, str) and seq:
        for aa in seq:
            if aa not in aminoacids:
                logger.info('aa %s is not natural', aa)
                return False
        return True
    logger.info('seq %s is not valid', seq)
    return False