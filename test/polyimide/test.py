from misc.pipeline import build_aa_topology
from misc.logger import logger
logger.setLevel('INFO')
logger.propagate = True
from misc.io.sdf import write_mols_to_sdf

smiA = 'Nc1ccc(Oc2ccc(N)cc2)cc1'
smiB = 'O1C(=O)c2cc(Oc3cc4C(=O)OC(=O)c4cc3)ccc2C1=O'

reaction_template = {
        'B-A': {
            'cg_reactant_list': [('A', 'B')],
            'smarts': '[#7H2:1].[#6:3](=[#8:4])[#8:2][#6:5]=[#8:6]>>[#6:3](=[#8:4])[#7:1][#6:5]=[#8:6].[#8:2]',
            'prod_idx': [0]
        }
    }
mols = {
    'A': {'smiles': smiA, 'file': None},
    'B': {'smiles': smiB, 'file': None},
}

xmlfile = 'cg.xml'

rdmols = build_aa_topology(mols, reaction_template, xmlfile, reactions=None, large=100)
write_mols_to_sdf(rdmols, 'polyimide.sdf')
