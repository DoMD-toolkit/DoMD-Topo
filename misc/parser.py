from rdkit import Chem
import numpy as np
from misc.logger import logger, DuplicateFilter
import tqdm
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Union
from misc.io.xml import XmlParser
bondorder_to_type = {
        0: Chem.rdchem.BondType.UNSPECIFIED,
        1: Chem.rdchem.BondType.SINGLE,
        1.5:Chem.rdchem.BondType.AROMATIC,
        2:Chem.rdchem.BondType.DOUBLE,
        3:Chem.rdchem.BondType.TRIPLE
}


def mols_to_nxgraphs(molecules: List[Union[Chem.Mol, Chem.RWMol]]):
    """Converts a list of RDKit molecules into NetworkX graphs.



        Preserves atomic properties (element, mass, charge, aromaticity) and custom
        residue identifiers (`res_id`, `global_res_id`, `res_name`) as node attributes.
        Bond orders are stored as edge attributes.

        Args:
            molecules (List[Union[Chem.Mol, Chem.RWMol]]): List of RDKit molecule objects.

        Returns:
            List[nx.Graph]: A list of NetworkX graphs representing the molecules.
    """
    mols_meta = []
    for mol in tqdm.tqdm(molecules,total=len(molecules),desc='converting Chem.Mol to nx.Graph',disable=True):
        mol_meta = nx.Graph()
        nodes = set()
        tqdm_show = True
        #if mol.GetNumAtoms() > 5000:
        #    tqdm_show = True
        for ai in tqdm.tqdm(mol.GetAtoms(),total=mol.GetNumAtoms(),desc='adding atoms',disable=tqdm_show):
            i = ai.GetIdx()
            if i not in nodes:
                mol_meta.add_node(i, element =ai.GetSymbol(),
                                     atomic_num = ai.GetAtomicNum(),
                                     mass =ai.GetMass(),
                                     formal_charge =ai.GetFormalCharge(),
                                     is_aromatic=ai.GetIsAromatic(),
                                     res_name =ai.GetProp('res_name'),
                                     res_id =ai.GetIntProp('res_id'),
                                     global_res_id =ai.GetIntProp('global_res_id'),
                                     chiral_tag = ai.GetChiralTag(),
                                     hybridization = ai.GetHybridization(),
                                     radical_electrons = ai.GetNumRadicalElectrons(),
                                     isotope = ai.GetIsotope()
                                )
        for bond in tqdm.tqdm(mol.GetBonds(),total=mol.GetNumBonds(),desc='adding bonds', disable=tqdm_show):
            ai, aj = bond.GetBeginAtom(), bond.GetEndAtom()
            i, j = ai.GetIdx(), aj.GetIdx()
            mol_meta.add_edge(i, j, bond_type=bond.GetBondType(),
                                    bondorder = bond.GetBondTypeAsDouble(),
                                    bond_stereo = bond.GetStereo(),
                                    bond_dir = bond.GetBondDir(),
                                    stereo_atoms = list(bond.GetStereoAtoms()))
        mols_meta.append(mol_meta)
    return mols_meta

def nxgraphs_to_mols(mols_meta: List[nx.Graph]):
    """Reconstructs RDKit molecules from NetworkX graphs.

        Reverse operation of `mols_to_nxgraphs`. Rebuilds the molecule structure and
        restores atomic properties and bond types based on the graph attributes.

        Args:
            mols_meta (List[nx.Graph]): List of NetworkX graphs with molecular metadata.

        Returns:
            List[Chem.Mol]: A list of sanitized RDKit molecules.
    """
    molecules = []
    for g in tqdm.tqdm(mols_meta,total=len(mols_meta),desc='converting nx.Graph to Chem.Mol',disable=True):
        mol = Chem.RWMol()
        nodes = g.nodes
        n_to_aid = {}
        for n in g.nodes:
            atom = Chem.Atom(nodes[n]['atomic_num'])
            atom.SetIsAromatic(nodes[n]['is_aromatic'])
            atom.SetFormalCharge(nodes[n]['formal_charge'])
            atom.SetIntProp('res_id',nodes[n]['res_id'])
            atom.SetIntProp('global_res_id',nodes[n]['global_res_id'])
            atom.SetProp('res_name',nodes[n]['res_name'])
            aid = mol.AddAtom(atom)
            n_to_aid[n] = aid
        edges = g.edges
        for i,j in g.edges:
            mol.AddBond(
                    n_to_aid[i],
                    n_to_aid[j],
                    bondorder_to_type[edges[(i,j)]['bondorder']]
                    )
        mol = Chem.Mol(mol)
        Chem.SanitizeMol(mol)
        molecules.append(mol)
    return molecules

def compute_rg_tensor(positions, masses=None):
    """Computes the Radius of Gyration (Rg) tensor and its principal components.

    The Rg tensor is calculated relative to the center of mass.
    .. math::
        R_{cm} = \\frac{\\sum m_i r_i}{\\sum m_i}
    .. math::
        S = \\frac{1}{M} \\sum_{i} m_i (r_i - R_{cm})(r_i - R_{cm})^T
    Args:
        positions (array_like): Atomic coordinates of shape (N, 3).
        masses (array_like, optional): Atomic masses of shape (N,).
            If None, all masses are set to 1.0. Defaults to None.
    Returns:
        tuple: A tuple containing:
            - Rg (np.ndarray): The 3x3 radius of gyration tensor.
            - eigvals (np.ndarray): The eigenvalues of the tensor, sorted descending.
            - principal_rgs (np.ndarray): The square roots of the eigenvalues (principal radii).
            - eigvecs (np.ndarray): The eigenvectors (principal axes), columns correspond to eigvals.
    Raises:
        ValueError: If the length of `masses` does not match `positions`.
    """
    coords = np.asarray(positions, dtype=float)
    N = coords.shape[0]

    if masses is None:
        m = np.ones(N)
    else:
        m = np.asarray(masses, dtype=float)
        if m.shape[0] != N:
            raise ValueError("masses length must match number of positions")

    M = m.sum()
    r_cm = (coords * m[:,None]).sum(axis=0) / M

    dr = coords - r_cm  # shape (N,3)

    Rg = (m[:,None,None] * np.einsum('ki,kj->kij', dr, dr)).sum(axis=0) / M

    eigvals, eigvecs = np.linalg.eigh(Rg)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    principal_rgs = np.sqrt(eigvals)

    return Rg, eigvals, principal_rgs, eigvecs

def read_cg_topology(cg_system, residues: Dict) -> Tuple[nx.Graph, List[nx.Graph]]:
    """Converts raw CG system data into NetworkX topology graphs.

    Constructs a graph representation of the coarse-grained system where nodes correspond
    to residues/beads and edges correspond to bonds. It handles the identification of
    residue types, SMILES strings, and positions. It also separates connected components
    into individual molecules and identifies rigid bodies.

    Note:
        Positions are multiplied by 10 during import (likely converting nm to Å).

    Args:
        cg_system (object): An object containing simulation data. Expected to have a
            `.data` attribute (dict) with keys 'position', 'bond', 'type', and optionally 'body'.
        residues (Dict): Metadata dictionary mapping monomer types to info (e.g., {'A': {'smiles': '...'}}).

    Returns:
        Tuple[nx.Graph, List[nx.Graph]]: A tuple containing:
            - cg_sys (nx.Graph): The global graph containing all beads in the system.
            - cg_molecules (List[nx.Graph]): A list of subgraphs, each representing a distinct molecule
              (connected component).
    """
    box = np.array([cg_system.box.lx, cg_system.box.ly, cg_system.box.lz]).astype(float) * 10
    cg_sys = nx.Graph()
    for bond in cg_system.data['bond']:
        bond_type, i, j = str(bond[0]), int(bond[1]), int(bond[2])
        type_i, type_j = cg_system.data['type'][i], cg_system.data['type'][j]
        if cg_system.data.get('body') is None:
            cg_system.data['body'] = np.zeros(cg_system.data['type'].shape) -1
        body_i = cg_system.data['body'][i]
        body_j = cg_system.data['body'][i]
        ri = residues.get(type_i) or {}
        rj = residues.get(type_j) or {}
        if not (ri or rj):
            with DuplicateFilter(logger):
                logger.warning(f"The residues {residues} do not contain "
                               f"residue information for type {type_i} or {type_j}, "
                               f"this is usually for manually operations.")
        cg_sys.add_node(i, type=type_i,
                        smiles=ri.get('smiles'),
                        x=cg_system.data['position'][i] * 10,
                        body=body_i
                        )
        cg_sys.add_node(j, type=type_j,
                        smiles=rj.get('smiles'),
                        x=cg_system.data['position'][j] * 10,
                        body=body_j
                        )
        cg_sys.add_edge(i, j, bond_type=bond_type)
    # singe = nx.Graph()
    rigid_mols = {}
    for n in range(cg_system.data['type'].__len__()):
        if cg_sys.nodes.get(n) is None:
            t = cg_system.data['type'][n]
            body = cg_system.data['body'][n]
            if body >= 0:
                if rigid_mols.get(body) is None:
                    rigid_mols[body] = []
                rigid_mols[body].append(n)
                continue
            r = residues.get(t) or {}
            cg_sys.add_node(n, type=t, smiles=r.get('smiles'),
                            x=cg_system.data['position'][n] * 10,
                            body=body)
    cg_molecules = [cg_sys.subgraph(c).copy() for c in nx.connected_components(cg_sys)]
    for rigid_id in rigid_mols:
        cg_rigid_mol = nx.Graph()
        for n in rigid_mols[rigid_id]:
            t = cg_system.data['type'][n]
            r = residues.get(t) or {}
            cg_rigid_mol.add_node(
                    n, type=t, smiles=r.get('smiles'),
                    x=cg_system.data['position'][n] * 10,
                    body=rigid_id
                    )
            cg_sys.add_node(
                    n, type=t, smiles=r.get('smiles'),
                    x=cg_system.data['position'][n] * 10,
                    body=rigid_id
                    )
        cg_molecules.append(cg_rigid_mol)
    for cg_mol in cg_molecules:
        cg_mol.graph['box'] = box
        is_rigid = False
        for res_id, node in enumerate(cg_mol):
            if cg_mol.nodes[node]['body'] >=0:
                is_rigid=True
                mol_type = cg_mol.nodes[node]['type']
                cg_mol.graph['type'] = mol_type
            cg_mol.nodes[node]["local_res_id"] = res_id
        cg_mol.graph['is_rigid'] = is_rigid
        if is_rigid:
            rigid_pos = cg_system.data['position'][list(cg_mol.nodes)]
            rxx, ryy, rzz = (rigid_pos**2).mean(axis=0)
            rxy = (rigid_pos[:,:1]*rigid_pos[:,1:2]).mean()
            rxz = (rigid_pos[:,:1]*rigid_pos[:,2:3]).mean()
            ryz = (rigid_pos[:,2:3]*rigid_pos[:,1:2]).mean()
            rg_tensor_eigv = np.array([
                                     [rxx,rxy,rxz],
                                     [rxy,ryy,ryz],
                                     [rxz,ryz,rzz]
                                     ])
            cg_mol.graph['rg_tensor_eigv'] = rg_tensor_eigv
    return cg_sys, cg_molecules

def post_process_aa_mol(rdmol, box_tensor):
    """Post-processes a list of all-atom RDKit molecules.

    This function sanitizes each molecule, adds hydrogens, and sets the 'global_res_id'
    property for all atoms in the molecule.

    Args:
        rdmol (Chem.Mol): An RDKit molecule object.
        box_tensor (list): A list of 9 floats representing the box dimensions.
    :return:
        Chem.Mol: The processed RDKit molecule with updated properties.
    """
    box_tensor = [str(l) for l in box_tensor] + ['0']*(9 - len(box_tensor)) if len(box_tensor) < 9 else [str(l) for l in box_tensor]
    box_tensor_str = ' '.join(box_tensor)
    res_name = []
    res_id = []
    for a in rdmol.GetAtoms():
        res_id.append(a.GetIntProp("global_res_id") if a.HasProp("global_res_id") else 1)
        res_name.append(a.GetProp("res_name") if a.HasProp("res_name") else "UNL")
    res_name_str = ' '.join(res_name)
    res_num_str = ' '.join(map(str, res_id))
    # Best for native RDKit compatibility (only supports strings/ints).
    rdmol.SetProp("RES_NAMES", res_name_str)
    rdmol.SetProp("RES_NUMS", res_num_str)
    rdmol.SetProp("BOX_TENSOR", box_tensor_str)
    return rdmol



def parse_cg_xml_topology(
    xml_path: str,
    mols_config: Dict[str, Any],
    reactions: Dict = None,
    rigid_meta: Optional[Dict[str, Any]] = None
) -> Tuple[List[nx.Graph], np.ndarray, Dict[int, List[Tuple[str, int, int]]]]:
    """
    CG XML Parser Workshop: Extracts box dimensions, builds coarse-grained molecular graphs,
    maps rigid-body metadata, and infers linking reactions from the raw XML layout.

    Args:
        xml_path (str): Path to the GALAMOST CG .xml configuration file.
        mols_config (dict): Configuration mapping CG bead types to their atomic SMILES definitions.
        reactions (list, optional): Explicit ordered sequence of reactions.
            If None, infers automatically from XML bonds.
        rigid_meta (dict, optional): External tracking dictionary containing 'rigid_aidxs_map' configurations.

    Returns:
        Tuple[List[nx.Graph], np.ndarray, List[Tuple[int, int, int]]]:
            - cg_mols: Pre-processed list of NetworkX graphs representing coarse-grained molecules.
            - box_tensor: NumPy array containing 3D simulation box lengths [Lx, Ly, Lz] in Angstroms.
            - reactions: Finalized execution sequence of reaction tuples (atom_i, atom_j, bond_type).
    """
    logger.info(f"Parsing CG XML file from layout reference: '{xml_path}'...")
    xml = XmlParser(xml_path)

    # 1. Extract dimensional configurations and scale to standard Angstrom units
    box_coords = (xml.box.lx, xml.box.ly, xml.box.lz, xml.box.xy, xml.box.xz, xml.box.yz)
    box_tensor = np.array(tuple(map(float, box_coords)))[:3] * 10

    # 2. Extract baseline system entities and local molecule subgraphs
    cg_sys, cg_mols = read_cg_topology(xml, mols_config)

    # 3. Complement rigid-body topologies with structural map indexes if provided
    if rigid_meta is not None:
        for cg_mol in cg_mols:
            if cg_mol.graph.get('is_rigid', False):
                mol_type = cg_mol.graph.get('type')
                if mol_type in rigid_meta:
                    cg_mol.graph['rigid_aidxs_map'] = rigid_meta[mol_type]['rigid_aidxs_map']
                else:
                    logger.warning(
                        f"Rigid flag detected for type '{mol_type}', "
                        f"but target metadata was omitted inside rigid_meta specifications."
                    )

    # 4. Infer operational transaction logic from XML bonds tracking if explicit sequence is missing
    if not reactions:
        reactions = {}
        for i, cg_mol in enumerate(cg_mols):
            reactions[i] = []
            for bond in cg_mol.edges(data=True):
                node_i, node_j = bond[0], bond[1]
                bond_type = bond[2].get('bond_type')
                reactions[i].append((bond_type, node_i, node_j))
        logger.info(f"Inferred {len(reactions)} reaction connections directly from the XML bond matrix.")
    else:
        logger.info(f"Using {len(reactions)} explicitly provided reaction parameters.")

    return cg_mols, box_tensor, reactions