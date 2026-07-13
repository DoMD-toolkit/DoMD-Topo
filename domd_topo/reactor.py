from itertools import permutations
from typing import Any, Union

import networkx as nx
import tqdm
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from ..misc.logger import logger
from ..misc.parser import mols_to_nxgraphs, molecule_reader
from ._mapping import process_reactants, atom_map, bond_map
from .lib import set_molecule_id_for_h


def reaction_mol_mapping(reactions: list[tuple]) -> dict[int, set]:
    """Groups reactions by the indices of the participating reactants.

        Args:
            reactions (list[tuple]): A list of reaction tuples, where each tuple contains
                (reaction_name, reactant_index_1, reactant_index_2, ...).

        Returns:
            dict[int, set]: A dictionary mapping a reactant index to a set of all
                reactions involving that reactant.
    """
    reaction_hash = {}
    for r in reactions:
        reaction_indices = r[1:]
        for rid in reaction_indices:
            if reaction_hash.get(rid) is None:
                reaction_hash[rid] = set()
            reaction_hash[rid].add(r)
    return reaction_hash


def reaction_mol_mapping_(reactions: list[tuple], cg_molecules: list[nx.Graph]) -> dict[int, list]:
    """Maps reactions to specific Coarse-Grained (CG) molecules.

        This function associates a list of reactions with the index of the CG molecule
        in which they occur. It assumes global node indices map uniquely to specific molecules.

        Args:
            reactions (list[tuple]): List of reaction tuples.
            cg_molecules (list[nx.Graph]): List of CG molecule graphs.

        Returns:
            dict[int, list]: A mapping from CG molecule index to a list of reactions
                occurring within that molecule.
    """
    reactions_hash = {i: [] for i, cgm in enumerate(cg_molecules)}
    global_to_molId = {}
    for im, cg_mol in enumerate(cg_molecules):
        for n in cg_mol.nodes:
            global_to_molId[n] = im
    for r in reactions:
        reaction_indices = r[1:]
        mid = global_to_molId[reaction_indices[0]]
        reactions_hash[mid].append(r)
    return reactions_hash


class Reaction(object):
    """Represents a specific chemical reaction template defined by SMARTS.

        Manages the RDKit reaction object and caches mapping information (atom and bond changes)
        for specific sets of reactant molecules to speed up processing.

        Attributes:
            cg_reactant_list (list): List of reactant types (strings) involved.
            reaction_name (str): Name of the reaction.
            reaction (rdChemReactions.ChemicalReaction): RDKit reaction object.
            smarts (str): SMARTS string defining the reaction.
            prod_idx (int, optional): Index of the main product in the reaction definition.
            reaction_maps (dict): Cache of pre-calculated atom/bond mappings.
    """

    def __init__(self, name, cg_reactant_list, smarts, prod_idx=None):
        """Initializes the Reaction object.

                Args:
                    name (str): Name of the reaction.
                    cg_reactant_list (list): Types of CG beads involved.
                    smarts (str): Reaction SMARTS string.
                    prod_idx (int, optional): Index of the product to track. Defaults to None.
        """
        self.cg_reactant_list = cg_reactant_list
        self.reaction_name = name
        self.reaction = rdChemReactions.ReactionFromSmarts(smarts)
        self.smarts = smarts
        self.prod_idx = prod_idx
        self.reaction_maps = {}

    def build_reaction_maps(self, cg_reactants, molecules):
        """Pre-calculates and caches atom and bond mappings for a set of reactants.

                Runs the reaction on the provided molecules to determine how atoms map
                from reactants to products and how bonds change.

                Args:
                    cg_reactants (tuple): Tuple of reactant types.
                    molecules (list[Chem.Mol]): List of RDKit molecule objects corresponding to reactants.

                Raises:
                    ValueError: If the reaction produces no products.
        """
        if self.reaction_maps.get(cg_reactants) is None:
            self.reaction_maps[cg_reactants] = []
            reactants = process_reactants(molecules)
            products = self.reaction.RunReactants(reactants)
            if len(products) == 0:
                raise ValueError(f"Reaction {self.smarts} does not run on CG reactants {cg_reactants}")
            for product in products:
                amap, reacting_atoms = atom_map(product, self.reaction)
                bmap = bond_map(reactants, product, self.reaction, self.smarts)
                self.reaction_maps[cg_reactants].append((reacting_atoms, amap, bmap))


def allowed_p(reacted_atoms, cg_reactants, reaction):
    """Determines if a specific reaction pathway is allowed based on atom availability.

        Checks if the specific atoms required for a reaction map have already been
        involved in previous reactions (conflict checking).

        Args:
            reacted_atoms (dict): Dictionary mapping reactant index to a set of atom indices
                that have already reacted.
            cg_reactants (tuple): Tuple of reactant types.
            reaction (Reaction): The Reaction object.

        Returns:
            tuple: A tuple (reaction_map, prod_idx) if allowed, otherwise (None, None).
    """
    for reaction_map in reaction.reaction_maps.get(cg_reactants):
        allowed = True
        for ri in reaction_map[0]:
            if not set(reaction_map[0][ri]).symmetric_difference(reacted_atoms[ri]):  # not necessarily subset,e.g.
                # reaction 1 takes {0,1},{10,11} but reaction 2 takes {0,1,2,3},{10,11,12,13}
                # if reaction 1 happened with (0,1} already, reacted atoms has intersection with
                # reaction 2
                # if set.issubset(set(reaction map[0][ri]),reacted atoms[ri]): 
                # only idle function groups
                # multi-step reaction info are considered as reaction info with all reactants in one step.
                # FOR ANY ATOM, THERE IS ONLY ONE REACTION, therefore intersection is fine.
                allowed = False
        if allowed:
            return reaction_map, reaction.prod_idx
    return None, None


def post_process(aa_mol: Union[Chem.Mol, Chem.RWMol]) -> tuple[Chem.Mol, nx.Graph]:
    """Post-processes an all-atom molecule to generate its corresponding graph representation.

            Args:
                aa_mol (Union[Chem.Mol, Chem.RWMol]): The all-atom RDKit molecule.

            Returns:
                tuple: A tuple containing:
                    - aa_mol (Chem.Mol): The processed all-atom RDKit molecule.
                    - mol_graph (nx.Graph): Graph representation of the molecule with atom and bond properties.
    """
    Chem.SanitizeMol(aa_mol)
    aa_mol_h = Chem.AddHs(aa_mol)
    set_molecule_id_for_h(aa_mol_h)
    mol_graph = mols_to_nxgraphs([aa_mol_h])[0]
    return aa_mol_h, mol_graph


class Reactor(object):
    """Orchestrates the generation of All-Atom topologies from Coarse-Grained graphs.



        This class handles the initialization of all-atom molecules from SMILES/PDB based on
        CG types, applies reactions defined in the template to update connectivity, and manages
        property assignment (Residue IDs).

        Attributes:
            reactants_meta (dict): Metadata for reactants (SMILES, file paths).
            reaction_templates (dict): Dictionary of Reaction objects.
    """

    def __init__(self, reactants_meta: dict[str, dict[str, str]], reaction_templates: dict[str, dict[str, Any]]):
        self.reactants_meta = reactants_meta
        # self.cg_molecules = None
        # self.aa_molecules = []
        # self.meta = []
        self.reaction_templates = {}
        for reaction_name in reaction_templates:
            _info = reaction_templates[reaction_name]
            self.reaction_templates[reaction_name] = Reaction(
                reaction_name, _info['cg_reactant_list'], _info['smarts'], _info.get("prod_idx")
            )

    def process(self, cg_mol: nx.Graph, reactions: list) -> tuple[Chem.Mol, nx.Graph]:
        """Processes a single CG molecule to generate its All-Atom structure.

                Args:
                    cg_mol (nx.Graph): Coarse-Grained graph where nodes represent monomers/reactants
                        and edges represent connectivity.
                    reactions (list): List of reactions to apply.
                    rigid_configs (dict, optional): Configuration for rigid molecules, including file paths and mappings.

                Returns:
                    tuple: A tuple containing:
                        - aa_molecule (Chem.RWMol): The generated all-atom RDKit molecule.
                        - meta (nx.Graph): Metadata graph tracking the mapping between CG nodes and AA atoms.

                Raises:
                    ValueError: If a reaction template or reactant definition is missing, or if a reaction fails.
        """
        rigid_configs = cg_mol.graph.get('rigid_configs', {})
        rigid_nodes = set()
        non_rigid_nodes = set()
        for n in cg_mol.nodes:
            if cg_mol.nodes[n].get('body') >= 0:
                rigid_nodes.add(n)
            else:
                non_rigid_nodes.add(n)
        rigid_groups = cg_mol.graph.get('rigid_groups')
        aa_mol = Chem.RWMol()
        mol_meta = nx.Graph()
        rigid_mol_global2local = {}
        global_count = 0
        rigid_types = set()
        for body_id in rigid_groups:
            file = rigid_configs[body_id]['file']
            mapping = rigid_configs[body_id]['mapping']  # react_site cg_node to rigid_mol atom_idx
            rigid_mol = molecule_reader(file)
            positions = rigid_mol.GetConformer(0).GetPositions()

            rigid_nodes_ = rigid_groups[body_id]
            rigid_mol_local2global = {}
            for atom_id in range(rigid_mol.GetNumAtoms()):
                rigid_mol_local2global[atom_id] = atom_id + global_count
                rigid_mol_global2local[atom_id + global_count] = atom_id
                atom = rigid_mol.GetAtomWithIdx(atom_id)
                aa_mol.AddAtom(atom)
            for bond in rigid_mol.GetBonds():
                aa_mol.AddBond(
                    bond.GetBeginAtomIdx() + global_count,
                    bond.GetEndAtomIdx() + global_count,
                    bond.GetBondType()
                )
                stereo = bond.GetStereo()
                bond_created = aa_mol.GetBondBetweenAtoms(
                    bond.GetBeginAtomIdx() + global_count,
                    bond.GetEndAtomIdx() + global_count,
                )
                bond_created.SetStereo(stereo)
                bond_created.SetBondDir(bond.GetBondDir())
            global_count += rigid_mol.GetNumAtoms()
            for atom_id in range(rigid_mol.GetNumAtoms()):
                atom = aa_mol.GetAtomWithIdx(rigid_mol_local2global[atom_id])
                atom.SetIntProp('global_res_id', -body_id - 1)
                atom.SetIntProp('res_id', -body_id - 1)
                atom.SetIntProp('body_id', body_id)
                atom.SetIntProp('intra_mol_id', rigid_mol_global2local[atom.GetIdx()])
                atom.SetProp('res_name', str(cg_mol.nodes[list(rigid_nodes_)[0]].get('type')))
                atom.SetIntProp('x', int(positions[atom_id][0] * 10000))
                atom.SetIntProp('y', int(positions[atom_id][1] * 10000))
                atom.SetIntProp('z', int(positions[atom_id][2] * 10000))
            for node in rigid_nodes_:
                local_res_id = cg_mol.nodes[node].get(
                    'intra_mol_id')  # intra_mol_id is the local residue id in the rigid molecule, which is used to map to the rigid molecule atom index.
                rigid_type = cg_mol.nodes[node].get('type')
                rigid_types.add(rigid_type)
                if local_res_id in mapping:
                    rigid_frag_atom_mapping = cg_mol.nodes[node].get('frag_atom_mapping')
                    rigid_react_site_atom_id_list = sorted(mapping[local_res_id]['atom_index'])
                    atom_idx = {rigid_frag_atom_mapping[i]: rigid_mol_local2global[rigid_react_site_atom_id_list[i]] for
                                i in range(len(rigid_react_site_atom_id_list))}

                    mol_meta.add_node(node, atom_idx=atom_idx, reacting_map={}, rm_atoms=set())
                    for rigid_react_site_atom_id in rigid_react_site_atom_id_list:
                        atom = aa_mol.GetAtomWithIdx(rigid_mol_local2global[rigid_react_site_atom_id])
                        atom.SetIntProp('global_res_id', int(node))
                        atom.SetIntProp('res_id', local_res_id)
                else:
                    mol_meta.add_node(node, atom_idx={}, reacting_map={}, rm_atoms=set())

        for node in non_rigid_nodes:
            atom_idx = {}
            reactant = cg_mol.nodes[node]
            reactant_molecule = Chem.MolFromSmiles(reactant['smiles'])
            for atom_id in range(reactant_molecule.GetNumAtoms()):
                atom = reactant_molecule.GetAtomWithIdx(atom_id)
                aa_mol.AddAtom(atom)
                atom_idx[atom_id] = atom_id + global_count
            for bond in reactant_molecule.GetBonds():
                aa_mol.AddBond(
                    bond.GetBeginAtomIdx() + global_count,
                    bond.GetEndAtomIdx() + global_count,
                    bond.GetBondType()
                )
                stereo = bond.GetStereo()
                bond_created = aa_mol.GetBondBetweenAtoms(
                    bond.GetBeginAtomIdx() + global_count,
                    bond.GetEndAtomIdx() + global_count,
                )
                bond_created.SetStereo(stereo)
                bond_created.SetBondDir(bond.GetBondDir())
            for bond in reactant_molecule.GetBonds():
                stereo = bond.GetStereo()
                bond_created = aa_mol.GetBondBetweenAtoms(
                    bond.GetBeginAtomIdx() + global_count,
                    bond.GetEndAtomIdx() + global_count,
                )
                if stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE):
                    stereo_atoms = list(bond.GetStereoAtoms())
                    if len(stereo_atoms) == 2:
                        bond_created.SetStereoAtoms(stereo_atoms[0] + global_count, stereo_atoms[1] + global_count)
            global_count += reactant_molecule.GetNumAtoms()
            mol_meta.add_node(node, atom_idx=atom_idx, reacting_map={}, rm_atoms=set())
        if len(cg_mol.nodes) == 1:
            for m in mol_meta.nodes:
                molecule = mol_meta.nodes[m]
                for idx in molecule['atom_idx'].values():
                    atom = aa_mol.GetAtomWithIdx(idx)
                    atom.SetIntProp('global_res_id', int(m))
                    atom.SetProp('res_name', str(cg_mol.nodes[m].get('type')))
                    logger.debug(f"global_res_id for atom {idx} in residue {m} is {m}")
                    if cg_mol.nodes[m].get('local_res_id') is None:
                        logger.warning(f"No local_res_id found in cg_molecule!")
                        atom.SetIntProp('res_id', -1)
                    else:
                        logger.debug(f"local_res_id for res {m}: {cg_mol.nodes[m]['local_res_id']}")
                        atom.SetIntProp('res_id', cg_mol.nodes[m]['local_res_id'])
            aa_mol_h, mol_graph = post_process(aa_mol)
            return aa_mol_h, mol_graph
        for edge in cg_mol.edges:
            mol_meta.add_edge(*edge)
        r__id = 0
        for r in tqdm.tqdm(reactions, total=len(reactions), desc='reacting', disable=True):
            r__id += 1
            reaction_name = r[0]
            _reactant_idx = r[1:]
            rxn_tpls = self.reaction_templates.get(reaction_name)
            if rxn_tpls is None:
                raise ValueError(f"Reaction {r} is not defined in reaction_info!")
            _all_orders = list(permutations(_reactant_idx))
            _reactants_tuple = tuple([cg_mol.nodes[_]['type'] for _ in _reactant_idx])
            reactants_order = reactants_tuple = None
            for _order in _all_orders:
                _tuple = tuple([cg_mol.nodes[_]['type'] for _ in _order])
                if _tuple in rxn_tpls.cg_reactant_list:
                    reactants_order = _order
                    reactants_tuple = _tuple
            if not reactants_order:
                raise ValueError(f"Reaction {r} for reactants ({_reactants_tuple}) is not defined!")

            _molecules = []
            for i, t in enumerate(reactants_tuple):
                meta = cg_mol.nodes[reactants_order[i]]
                if t in rigid_types:
                    if meta.get('smarts'):
                        _molecules.append(Chem.MolFromSmarts(meta['smarts']))
                    elif meta.get('smiles'):
                        _molecules.append(Chem.MolFromSmiles(meta['smiles']))
                    else:
                        raise ValueError(
                            f"Rigid reactant type '{t}' requires either 'smarts' or 'smiles' in reactants_meta, but neither was found.")
                else:
                    if meta.get('smiles'):
                        _molecules.append(Chem.MolFromSmiles(meta['smiles']))
                    else:
                        raise ValueError(
                            f"Flexible reactant type '{t}' requires 'smiles' in reactants_meta, but it was not found.")
            rxn_tpls.build_reaction_maps(reactants_tuple, _molecules)

            if len(reactants_tuple) == 2:
                if reactants_tuple[0] == reactants_tuple[1]:
                    di_same = True
                    reactants_order = _reactant_idx  # sorted(reactants_order)
            reactants = [mol_meta.nodes[_] for _ in reactants_order]
            key = tuple(sorted(_reactant_idx))
            reacted_atoms = {}

            for ri in range(len(reactants)):
                reacted_atoms[ri] = set()

            for ri, rt in enumerate(reactants):  # keep reactant order
                for k in rt['reacting_map']:
                    for at in rt['reacting_map'][k]:
                        reacted_atoms[ri].add(at)

            reaction_map, product_idx = allowed_p(reacted_atoms, reactants_tuple, rxn_tpls)

            if not reaction_map:
                if not reaction_map:
                    raise (ValueError(
                        f"{r} with order {_reactant_idx}, {_reactants_tuple} can not react! This error happens while "
                        f"the reacted atoms in one bead have been reacted more than once."))

            amap, bmap = reaction_map[1], reaction_map[2]  # store the reacted atoms.
            for ri, rt in enumerate(reactants):
                if rt['reacting_map'].get(key) is None:
                    rt['reacting_map'][key] = set()
                for at in reaction_map[0][ri]:
                    rt['reacting_map'][key].add(at)

            for atom in amap:
                if product_idx is not None:
                    if atom.product_id not in product_idx:
                        reactant = reactants[atom.reactant_id]
                        reactant['rm_atoms'].add(reactant['atom_idx'][atom.reactant_atom_id])
            for b in bmap:
                if b.status == 'deleted':
                    reactant = reactants[b.reactants_id[0]]
                    bi = reactant['atom_idx'][b.reactant_atoms_id[0]]
                    bj = reactant['atom_idx'][b.reactant_atoms_id[1]]
                    aa_mol.RemoveBond(bi, bj)
                if b.status == 'changed':
                    reactant = reactants[b.reactants_id[0]]
                    bi = reactant['atom_idx'][b.reactant_atoms_id[0]]
                    bj = reactant['atom_idx'][b.reactant_atoms_id[1]]
                    bond = aa_mol.GetBondBetweenAtoms(bi, bj)
                    bond.SetBondType(b.bond_type)

                    bond.SetBondDir(b.bond_dir)
                    if (b.bond_stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE)) and (
                            len(b.stereo_atoms) == 2):
                        bond.SetStereoAtoms(b.stereo_atoms[0], b.stereo_atoms[1])
                        bond.SetStereo(b.bond_stereo)
                if b.status == 'new':
                    reactant0 = reactants[b.reactants_id[0]]
                    reactant1 = reactants[b.reactants_id[1]]
                    bi = reactant0['atom_idx'][b.reactant_atoms_id[0]]
                    bj = reactant1['atom_idx'][b.reactant_atoms_id[1]]
                    aa_mol.AddBond(bi, bj, b.bond_type)
                    bond = aa_mol.GetBondBetweenAtoms(bi, bj)
                    bond.SetBondDir(b.bond_dir)
                    if (b.bond_stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE)) and (
                            len(b.stereo_atoms) == 2):
                        bond.SetStereoAtoms(b.stereo_atoms[0], b.stereo_atoms[1])
                        bond.SetStereo(b.bond_stereo)

        rm_all = []
        for m in tqdm.tqdm(mol_meta.nodes,
                           total=len(mol_meta.nodes),
                           desc='set res_id and get removing atom',
                           disable=True):
            molecule = mol_meta.nodes[m]
            for idx in molecule['atom_idx'].values():
                atom = aa_mol.GetAtomWithIdx(idx)
                if not atom.HasProp('body_id'):
                    atom.SetIntProp('body_id', -1)
                atom.SetIntProp('global_res_id', int(m))
                atom.SetProp('res_name', str(cg_mol.nodes[m]['type']))
                logger.debug(f"global_res_id for atom {idx} in residue {m} is {m}")
                if cg_mol.nodes[m].get('local_res_id') is None:
                    logger.warning(f"No local_res_id found in cg_molecule!")
                    atom.SetIntProp('res_id', -1)
                else:
                    logger.debug(f"local_res_id for res {m}: {cg_mol.nodes[m]['local_res_id']}")
                    atom.SetIntProp('res_id', cg_mol.nodes[m]['local_res_id'])
            rm_all.extend(list(molecule['rm_atoms']))

        rm_all = sorted(list(set(rm_all)), reverse=True)
        for bi in tqdm.tqdm(rm_all, total=len(rm_all), desc='removing atom', disable=True):
            aa_mol.RemoveAtom(bi)
        aa_mol_h, mol_graph = post_process(aa_mol)
        return aa_mol_h, mol_graph
