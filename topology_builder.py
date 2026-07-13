import os

from domd_topo.lib import reactions_search
from rdkit import Chem

from .domd_topo.reactor import Reactor
from .misc.parser import mols_to_nxgraphs


def topology_builder(reactants_config, reaction_template, rigid_configs=None, cg_graph=None, reactions=None):
    """Builds the all-atom topology from coarse-grained input using the Reactor class.

        Args:
            reactants_config (dict): Configuration for reactant molecules, including SMILES and file paths.
            reaction_template (dict): Reaction SMARTS patterns and topology rules.
            rigid_configs (dict, optional): Configuration for rigid molecules, including file paths and mappings.
            cg_graph (networkx.Graph, optional): Coarse-grained graph representation of the system.
            reactions (list/tuple, optional): Explicit sequence of reactions. If None, inferred from cg_graph.

        Returns:
            tuple: A tuple containing:
                - list[Chem.Mol]: List of reconstructed all-atom molecules.
                - dict: Metadata associated with the reconstructed molecules.
    """
    if rigid_configs is None:
        rigid_configs = {}
    if cg_graph.graph['rigidity'] == 'RIGID':
        body_ids = cg_graph.graph['body_id']
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body_id ({body_ids}) for a rigid molecule, but found {len(body_ids)}."
                f"Please ensure that the whole rigid coarse-grained molecule with same body id.")
        body_id = body_ids[0]
        mol_type = f'R_{body_id}'
        aa_file = rigid_configs[body_id]['file']
        if os.path.exists(aa_file):
            aa_mol_h = Chem.MolFromPDBFile(aa_file, removeHs=False)
            for a in aa_mol_h.GetAtoms():
                a.SetIntProp("res_id", -1)
                a.SetIntProp("global_res_id", -1)
                a.SetProp('res_name', mol_type)
            aa_graph = mols_to_nxgraphs([aa_mol_h])[0]
        else:
            raise FileNotFoundError(f"Rigid molecule file of rigid type {cg_graph.graph['type']} not found: {aa_file}")
    else:
        if cg_graph is None and reactions is None:
            raise ValueError("Either cg_graph or reactions must be provided for topology building.")
        if reactions is None:
            reactions = reactions_search(cg_graph)
        reactor = Reactor(reactants_config, reaction_template)
        aa_mol_h, aa_graph = reactor.process(cg_graph, reactions)
    return aa_mol_h, aa_graph
