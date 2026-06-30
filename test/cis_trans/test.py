from misc.pipeline import build_aa_topology
from misc.logger import logger
logger.setLevel('INFO')
logger.propagate = True

smiC = 'C=CC(=C)C'

reaction_template = {
        'C-C': {
            'cg_reactant_list': [('C', 'C')],
            'smarts': '[C:1]=[C:2][C:3](C)=[C:4].[C:5]=[CH1:6]>>[C:1]/[C:2]=[C:3](C)/[C:4][C:5]=[C:6]',
            'prod_idx': [0]
        },
}
mols = {
    'C': {'smiles': smiC, 'file': None},
    }

xmlfile = 'cg.xml'

rdmols = build_aa_topology(mols, reaction_template, xmlfile, reactions=None)
