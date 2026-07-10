from .reactor import Reactor
import os
from rdkit import Chem
from misc.parser import mols_to_nxgraphs

from collections import deque


def reactions_search(cg_graph):
    """Searches for reactions in the coarse-grained graph using a deterministic
    Breadth-First Search (BFS) traversal, strictly preserving the path discovery direction.

    Args:
        cg_graph (networkx.Graph): Coarse-grained graph representation of the system.

    Returns:
        list: A list of reactions found in the cg_graph, each represented as a tuple (bondtype, i, j).
              The direction i -> j strictly reflects the BFS exploration trajectory.
    """
    reactions = []
    visited_nodes = set()
    # Store undirected edges as a sorted tuple (min, max) to prevent processing the same edge twice,
    # while allowing us to keep the actual discovery order (curr -> nbr) in the final reactions list.
    visited_edges = set()
    # Step 1: Sort all nodes globally to ensure BFS roots start from the smallest index numbers
    global_sorted_nodes = sorted(list(cg_graph.nodes))
    for root_node in global_sorted_nodes:
        if root_node in visited_nodes:
            continue
        # Initialize BFS queue for the current connected component
        queue = deque([root_node])
        visited_nodes.add(root_node)
        while queue:
            curr_node = queue.popleft()
            # Step 2: Sort neighbors to enforce the "smaller index first" expansion priority locally
            sorted_neighbors = sorted(list(cg_graph.neighbors(curr_node)))
            for nbr_node in sorted_neighbors:
                # Generate a canonical edge key to check for duplicate undirected edge detection
                edge_key = (min(curr_node, nbr_node), max(curr_node, nbr_node))
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    # Fetch edge data from the graph
                    bond_data = cg_graph.edges[curr_node, nbr_node]
                    bondtype = bond_data['bond_type']
                    # CRITICAL: Append the reaction strictly in the direction of path discovery (curr_node -> nbr_node)
                    reactions.append((bondtype, curr_node, nbr_node))
                # If the neighbor node hasn't been discovered yet, push it to the queue
                if nbr_node not in visited_nodes:
                    visited_nodes.add(nbr_node)
                    queue.append(nbr_node)
    return reactions

def reactions_search_(cg_graph):
    """Searches for reactions in the coarse-grained graph based on the provided reaction template.

    Args:
        cg_graph (networkx.Graph): Coarse-grained graph representation of the system.
        reaction_template (dict): Reaction SMARTS patterns and topology rules.

    Returns:
        list: A list of reactions found in the cg_graph, each represented as a tuple (bondtype, i, j).
    """
    reactions = []
    for i, j, bond in cg_graph.edges(data=True):
        i, j = sorted((i, j))
        bondtype = bond['bond_type']
        reactions.append((bondtype, i, j))
    return reactions

def topology_builder(reactants_config, reaction_template, rigid_configs=None, cg_graph=None, reactions=None, mol_idx=0):
    """Builds the all-atom topology from coarse-grained input using the Reactor class.

        Args:
            reactants_config (dict): Configuration for reactant molecules, including SMILES and file paths.
            reaction_template (dict): Reaction SMARTS patterns and topology rules.
            rigid_configs (dict, optional): Configuration for rigid molecules, including file paths and mappings.
            cg_graph (networkx.Graph, optional): Coarse-grained graph representation of the system.
            reactions (list/tuple, optional): Explicit sequence of reactions. If None, inferred from cg_graph.
            mol_idx (int, optional): Index of the molecule being processed (for logging purposes).

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
            raise ValueError(f"Expected exactly one body_id ({body_ids}) for a rigid molecule, but found {len(body_ids)}."
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
        aa_mol_h, aa_graph = reactor.process(cg_graph, reactions, rigid_configs, mol_idx=mol_idx)
    return aa_mol_h, aa_graph