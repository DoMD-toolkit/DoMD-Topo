import json

from embed_molecule import embed_molecules
from misc.io.sdf import write_mols_to_sdf
from misc.parser import parse_config
from pipeline import topology_builder

user_config = json.load(open('config.json', 'r'))
config = parse_config(user_config)
rdmols, aa_graphs = [], []

for cg_graph, reactions in zip(config.cg_graphs, config.reaction_list):
    rdmol, aa_graph = topology_builder(config.reactant_config, config.reaction_template,
                                       config.rigid_config, cg_graph, reactions)
    rdmols.append(rdmol)
    aa_graphs.append(aa_graph)

rdmols = embed_molecules(rdmols, aa_graphs, config, chunk_per_d=1)
write_mols_to_sdf(rdmols, 'spe_network.sdf', force_v3000=True)