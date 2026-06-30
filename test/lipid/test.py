from misc.pipeline import build_aa_topology
from misc.logger import logger
logger.setLevel('INFO')
logger.propagate = True
from misc.io.sdf import write_mols_to_sdf

NC3 = 'C[N+](C)(C)C'
TAP = 'C[N+](C)C'
PO4 = 'OP(=O)([O-])OC'
GL1 = 'FC[C@@H](OC(=O)CC)CO'
GL2 = 'C(=O)CC'
GL0 = 'OCCO'
C1A = 'CCCF'
C2A = 'CCCF'
C3A = 'CCCF'
C4A = 'CCCF'
C5A = 'CCCF'
C1B = 'CCCF'
C2B = 'CCCF'
C3B = 'CCCF'
C4B = 'CCCF'
C5B = 'CCCF'
reaction_template = {
    'NC3-PO4': {
        'cg_reactant_list': [('NC3', 'PO4')],
        'smarts': '[C:1].[C:2]>>[C:1][C:2]',
        'prod_idx': [0]
    },
    'GL0-PO4': {
        'cg_reactant_list': [('GL0', 'PO4')],
        'smarts': '[C:1].[C:2]>>[C:1][C:2]',
        'prod_idx': [0]
    },
    'PO4-GL1': {
        'cg_reactant_list': [('PO4', 'GL1')],
        'smarts': '[O:1].[C@@H:2][C:3][F:4]>>[O:1][C:3][C@@H:2].[F:4]',
        'prod_idx': [0]
    },
    'GL1-GL2': {
        'cg_reactant_list': [('GL1', 'GL2')],
        'smarts': '[OH1:1].[C:2]=[O:3]>>[O:1][C:2]=[O:3]',
        'prod_idx': [0]
    },
    'GL1-C1A': {
        'cg_reactant_list': [('GL1', 'C1A')],
        'smarts': '[CH3:1].[C:2][F:3]>>[CH2:1][C:2].[F:3]',
        'prod_idx': [0]
    },
    'GL2-C1B': {
        'cg_reactant_list': [('GL2', 'C1B')],
        'smarts': '[CH3:1].[C:2][F:3]>>[CH2:1][C:2].[F:3]',
        'prod_idx': [0]
    },
    'C1A-C2A': {
        'cg_reactant_list': [('C1A', 'C2A')],
        'smarts': '[CH3:1].[F:2][C:3]>>[CH2:1][CH2:3].[F:2]',
        'prod_idx': [0]
    },
    'C2A-C3A': {
        'cg_reactant_list': [('C2A', 'C3A')],
        'smarts': '[CH3:1].[F:2][C:3]>>[CH2:1][CH2:3].[F:2]',
        'prod_idx': [0]
    },
    'C3A-C4A': {
        'cg_reactant_list': [('C3A', 'C4A')],
        'smarts': '[CH3:1].[F:2][C:3]>>[CH2:1][CH2:3].[F:2]',
        'prod_idx': [0]
    },
    'C4A-C5A': {
        'cg_reactant_list': [('C4A', 'C5A')],
        'smarts': '[CH3:1].[F:2][C:3]>>[CH2:1][CH2:3].[F:2]',
        'prod_idx': [0]
    },
    'C1B-C2B': {
        'cg_reactant_list': [('C1B', 'C2B')],
        'smarts': '[CH3:1].[F:2][C:3]>>[C:1][C:3].[F:2]',
        'prod_idx': [0]
    },
    'C2B-C3B': {
        'cg_reactant_list': [('C2B', 'C3B')],
        'smarts': '[CH3:1].[F:2][C:3]>>[CH2:1][CH2:3].[F:2]',
        'prod_idx': [0]
    },
    'C3B-C4B': {
        'cg_reactant_list': [('C3B', 'C4B')],
        'smarts': '[CH3:1].[F:2][C:3]>>[CH2:1][CH2:3].[F:2]',
        'prod_idx': [0]
    },
    'C4B-C5B': {
        'cg_reactant_list': [('C4B', 'C5B')],
        'smarts': '[CH3:1].[F:2][C:3]>>[CH2:1][CH2:3].[F:2]',
        'prod_idx': [0]
    },
}
mols = {
      'NC3': {'smiles': NC3, 'file': None, 'is_rigid':False},
      'PO4': {'smiles': PO4, 'file': None, 'is_rigid':False},
      'GL0': {'smiles': GL0, 'file': None, 'is_rigid':False},
      'GL1': {'smiles': GL1, 'file': None, 'is_rigid':False},
      'GL2': {'smiles': GL2, 'file': None, 'is_rigid':False},
      'C1A': {'smiles': C1A, 'file': None, 'is_rigid':False},
      'C2A': {'smiles': C2A, 'file': None, 'is_rigid':False},
      'C3A': {'smiles': C3A, 'file': None, 'is_rigid':False},
      'C4A': {'smiles': C4A, 'file': None, 'is_rigid':False},
      'C5A': {'smiles': C5A, 'file': None, 'is_rigid':False},
      'C1B': {'smiles': C1B, 'file': None, 'is_rigid':False},
      'C2B': {'smiles': C2B, 'file': None, 'is_rigid':False},
      'C3B': {'smiles': C3B, 'file': None, 'is_rigid':False},
      'C4B': {'smiles': C4B, 'file': None, 'is_rigid':False},
      'C5B': {'smiles': C5B, 'file': None, 'is_rigid':False},
}


xmlfile = 'cg.xml'

rdmols = build_aa_topology(mols, reaction_template, xmlfile, reactions=None, large=20)
write_mols_to_sdf(rdmols, 'lipid.sdf')
