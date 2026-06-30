import cgi
from itertools import permutations
from typing import Any
import pickle
import networkx as nx
import tqdm
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from misc.logger import logger
from ._mapping import process_reactants, atom_map, bond_map


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

#def allowed_p_(reacted_atoms, cg_reactants, reaction):
#    for reaction_map in reaction.reaction_maps.get(cg_reactants):
#        allowed = True
#        for ri in reaction_map[0]:
#            if set.intersection(set(reaction_map[0][ri]), reacted_atoms[ri]):
#                allowed = False
#        if allowed:
#            return reaction_map, reaction.prod_idx
#    return None, None  # if no available reaction is chosen.


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
        #print(reaction.reaction_maps)
        allowed = True
        for ri in reaction_map[0]:
            #print(set(reaction_map[0][ri]), reacted_atoms[ri], reaction_map[0], ri)
            if not set(reaction_map[0][ri]).symmetric_difference(reacted_atoms[ri]):#not necessarily subset,e.g.
                # reaction 1 takes {0,1},{10,11} but reaction 2 takes {0,1,2,3},{10,11,12,13}
                # if reaction 1 happened with (0,1} already, reacted atoms has intersection with
                # reaction 2
                # if set.issubset(set(reaction map[0][ri]),reacted atoms[ri]): 
                # only idle function groups
                # multi-step reaction info are considered as reaction info with all reactants in one step.
                # FOR ANY ATOM, THERE IS ONLY ONE REACTION, therefore intersection is fine.
                allowed =False
        #print('-')
        if allowed:
            #print(set(reaction_map[0][ri]), reacted_atoms[ri], reaction_map[0], ri)
            return reaction_map, reaction.prod_idx
    return None, None  # if no available reaction is chosen.


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

    def process(self, cg_molecules: list[nx.Graph], reactions: list) -> tuple[list[Chem.RWMol], list[nx.Graph]]:
        """Processes CG molecules to generate All-Atom structures.

                Iterates through each CG molecule:
                1.  **Initialization**: Creates an initial disconnected all-atom system by placing
                    reactants based on the CG nodes.
                2.  **Reaction Application**: Iterates through the provided `reactions` list. For each
                    reaction, it finds valid atom mappings, updates bonds (add/remove/change type),
                    and updates stereochemistry.
                3.  **Cleanup**: Removes atoms marked for deletion (e.g., small molecule byproducts)
                    and assigns residue IDs.

                Args:
                    cg_molecules (list[nx.Graph]): List of Coarse-Grained graphs where nodes represent
                        monomers/reactants and edges represent connectivity.
                    reactions (list): List of reactions to apply.

                Returns:
                    tuple: A tuple containing:
                        - aa_molecules (list[Chem.RWMol]): The generated all-atom RDKit molecules.
                        - meta (list[nx.Graph]): Metadata graphs tracking the mapping between CG nodes and AA atoms.

                Raises:
                    ValueError: If a reaction template or reactant definition is missing, or if a reaction fails.
        """
        aa_molecules = []
        meta = []
        reaction_hash = reaction_mol_mapping_(reactions, cg_molecules)
        #reaction_hash = reaction_mol_mapping(reactions)
        #print(reaction_hash)
        for _i, cg_mol in enumerate(cg_molecules):
            logger.info(f"Generating top for CG molecule {_i} with residue num of {len(cg_mol)}")
            if cg_mol.graph['is_rigid']:
                t = cg_mol.graph['type']
                if (self.reactants_meta[t].get('file') is None):
                    logger.warnning(f"Get no file for rigid molecule. Try to generate molecule from SMILES.")
                    if (self.reactants_meta[t].get('smiles') is None):
                        logger.error(f"Get no smiles for rigid molecule.")
                        raise
                    else:
                        aa_mol = Chem.MolFromSmiles(self.reactants_meta[t]['smiles'])
                        for a in aa_mol.GetAtoms():
                            a.SetIntProp('res_id',0)
                            a.SetIntProp('global_res_id',0)
                            a.SetProp('res_name',t)
                else:
                    aa_mol = Chem.MolFromPDBFile(self.reactants_meta[t]['file'], removeHs=False)
                    for a in aa_mol.GetAtoms():
                        a.SetIntProp('res_id',0)
                        a.SetIntProp('global_res_id',0)
                        a.SetProp('res_name',t)
                mol_meta = nx.Graph()
                for n in cg_mol.graph['rigid_aidxs_map']:
                    mol_meta.add_node(n, atom_idx=cg_mol.graph['rigid_aidxs_map'][n], reacting_map={},rm_atoms=set())
                aa_molecules.append(aa_mol)
                meta.append(mol_meta)
                continue
            aa_mol = Chem.RWMol()
            mol_meta = nx.Graph()
            global_count = 0
            mol_reactions = reaction_hash[_i]#set()
            #print(mol_reactions)
            for node in cg_mol.nodes:
                atom_idx = {}
                #for r in reaction_hash[node]:
                #    mol_reactions.add(r)
                reactant = cg_mol.nodes[node]
                #print(reactant)
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
                    #print(bond.GetBeginAtomIdx() + global_count,
                    #    bond.GetEndAtomIdx() + global_count,)
                    bond_created = aa_mol.GetBondBetweenAtoms(
                            bond.GetBeginAtomIdx() + global_count,
                            bond.GetEndAtomIdx() + global_count,
                            )
                    bond_created.SetStereo(stereo)
                    bond_created.SetBondDir(bond.GetBondDir())
                #print('-'*10)
                for bond in reactant_molecule.GetBonds():
                    stereo = bond.GetStereo()
                    bond_created = aa_mol.GetBondBetweenAtoms(
                            bond.GetBeginAtomIdx() + global_count,
                            bond.GetEndAtomIdx() + global_count,
                            )
                    if stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE):
                        stereo_atoms = list(bond.GetStereoAtoms())
                        #print(bond.GetBeginAtomIdx() + global_count,
                        #bond.GetEndAtomIdx() + global_count,stereo_atoms)
                        if len(stereo_atoms) == 2:
                            bond_created.SetStereoAtoms(stereo_atoms[0]+global_count, stereo_atoms[1]+global_count)
                global_count += reactant_molecule.GetNumAtoms()
                mol_meta.add_node(node, atom_idx=atom_idx, reacting_map={}, rm_atoms=set())
            #pickle.dump((reaction_hash, mol_reactions), open('test.pkl','wb'))
            #if len(cg_mol.nodes) > 1000:
            #    continue
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
                aa_molecules.append(aa_mol)
                meta.append(mol_meta)
                continue
            for edge in cg_mol.edges:
                mol_meta.add_edge(*edge)
            r__id = 0
            #print(mol_meta.nodes(data=True))
            #raise
            for r in tqdm.tqdm(mol_reactions,total=len(mol_reactions),desc='reacting',disable=True):
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
                for t in reactants_tuple:
                    _molecules.append(Chem.MolFromSmiles(self.reactants_meta[t]['smiles']))
                rxn_tpls.build_reaction_maps(reactants_tuple, _molecules)

                if len(reactants_tuple) == 2:
                    if reactants_tuple[0] == reactants_tuple[1]:
                        di_same = True
                        reactants_order = _reactant_idx #sorted(reactants_order)
                #print(_reactant_idx,reactants_order)
                #reactants_order = _reactant_idx
                reactants = [mol_meta.nodes[_] for _ in reactants_order]
                #print(reactants)
                key = tuple(sorted(_reactant_idx))
                reacted_atoms = {}

                for ri in range(len(reactants)):
                    reacted_atoms[ri] = set()

                for ri, rt in enumerate(reactants):  # keep reactant order
                    for k in rt['reacting_map']:
                        for at in rt['reacting_map'][k]:
                            reacted_atoms[ri].add(at)

                reaction_map, product_idx = allowed_p(reacted_atoms, reactants_tuple, rxn_tpls)
                #reaction_map, product_idx = allowed_p_(reacted_atoms, reactants_tuple, rxn_tpls)
                #print(reaction_map)
                #if not reaction_map:
                #    if di_same:
                #        reactants_order = [reactants_order[1], reactants_order[0]]
                #        reactants = [mol_meta.nodes[_] for _ in reactants_order]
                #        key = tuple(sorted(_reactant_idx))
                #        reacted_atoms = {}
                #        for ri in range(len(reactants)):
                #            reacted_atoms[ri] = set()
        
                #        for ri, rt in enumerate(reactants):  # keep reactant order
                #            for k in rt['reacting_map']:
                #                for at in rt['reacting_map'][k]:
                #                    reacted_atoms[ri].add(at)
                #        reaction_map, product_idx = allowed_p(reacted_atoms, reactants_tuple, rxn_tpls)

                if not reaction_map:
                    if not reaction_map:
                        raise (ValueError(
                            f"{r} with order {_reactant_idx}, {_reactants_tuple} can not react! This error happens while "\
                            f"the reacted atoms in one bead have been reacted more than once. We reconmand you use another "\
                            f"reaction template to avoid this."))

                amap, bmap = reaction_map[1], reaction_map[2]  # store the reacted atoms.
                #print(bmap)
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
                #print(bmap)
                #print(r,reactants)
                #print('-'*100)
                for b in bmap:
                    if b.status == 'deleted':
                        #print(b)
                        reactant = reactants[b.reactants_id[0]]
                        #print(reactant['atom_idx'].keys(),b)
                        bi = reactant['atom_idx'][b.reactant_atoms_id[0]]
                        bj = reactant['atom_idx'][b.reactant_atoms_id[1]]
                        aa_mol.RemoveBond(bi, bj)
                    if b.status == 'changed':
                        reactant = reactants[b.reactants_id[0]]
                        bi = reactant['atom_idx'][b.reactant_atoms_id[0]]
                        bj = reactant['atom_idx'][b.reactant_atoms_id[1]]
                        bond = aa_mol.GetBondBetweenAtoms(bi, bj)
                        bond.SetBondType(b.bond_type)
                        #bond.SetStereo(b.bond_stereo)
                        bond.SetBondDir(b.bond_dir)
                        if (b.bond_stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE)) and (len(b.stereo_atoms) == 2):
                            bond.SetStereoAtoms(b.stereo_atoms[0],b.stereo_atoms[1])
                            bond.SetStereo(b.bond_stereo)
                        #print(b.bond_stereo)
                    if b.status == 'new':
                        reactant0 = reactants[b.reactants_id[0]]
                        reactant1 = reactants[b.reactants_id[1]]
                        bi = reactant0['atom_idx'][b.reactant_atoms_id[0]]
                        bj = reactant1['atom_idx'][b.reactant_atoms_id[1]]
                        aa_mol.AddBond(bi, bj, b.bond_type)
                        bond = aa_mol.GetBondBetweenAtoms(bi, bj)
                        #bond.SetStereo(b.bond_stereo)
                        bond.SetBondDir(b.bond_dir)
                        if (b.bond_stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE)) and (len(b.stereo_atoms) == 2):
                            bond.SetStereoAtoms(b.stereo_atoms[0],b.stereo_atoms[1])
                        bond.SetStereo(b.bond_stereo)

            rm_all = []
            for m in tqdm.tqdm(mol_meta.nodes, total=len(mol_meta.nodes), desc='removing atom',disable=True):
                molecule = mol_meta.nodes[m]
                for idx in molecule['atom_idx'].values():
                    atom = aa_mol.GetAtomWithIdx(idx)
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
            for bi in tqdm.tqdm(rm_all, total=len(rm_all), desc='removing atom',disable=True):
                aa_mol.RemoveAtom(bi)
            #aid_mol2aid_mono = {}
            #for n in mol_meta.nodes:
            #    aid_mono2aid_mol = mol_meta.nodes[n]['atom_idx']
            #    for i in aid_mono2aid_mol:
            #        aid_mol2aid_mono[aid_mono2aid_mol[i]] = i
            #for i in aid_mol2aid_mono:
            #    atom = aa_mol.GetAtomWithIdx(i)
            #    atom.SetIntProp('monomer_idx',aid_mol2aid_mono[i])
            # Chem.AssignStereochemistry(aa_mol, force=True)
            aa_molecules.append(aa_mol)
            meta.append(mol_meta)
        return aa_molecules, meta
