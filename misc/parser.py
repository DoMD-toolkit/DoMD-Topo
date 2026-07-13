import os
from typing import Dict, List, Tuple, Any, Union

import gsd.hoomd
import networkx as nx
import numpy as np
import tqdm
from rdkit import Chem

from .io.xml import XmlParser
from .logger import logger

bondorder_to_type = {
    0: Chem.rdchem.BondType.UNSPECIFIED,
    1: Chem.rdchem.BondType.SINGLE,
    1.5: Chem.rdchem.BondType.AROMATIC,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE
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
    for mol in tqdm.tqdm(molecules, total=len(molecules), desc='converting Chem.Mol to nx.Graph', disable=True):
        mol_meta = nx.Graph()
        nodes = set()
        tqdm_show = True
        # if mol.GetNumAtoms() > 5000:
        #    tqdm_show = True
        has_body = False
        rigid_groups = {}
        for ai in tqdm.tqdm(mol.GetAtoms(), total=mol.GetNumAtoms(), desc='adding atoms', disable=tqdm_show):
            i = ai.GetIdx()
            if i not in nodes:
                body_id = ai.GetIntProp('body_id') if ai.HasProp('body_id') else -1
                mol_meta.add_node(i, element=ai.GetSymbol(),
                                  atomic_num=ai.GetAtomicNum(),
                                  mass=ai.GetMass(),
                                  formal_charge=ai.GetFormalCharge(),
                                  is_aromatic=ai.GetIsAromatic(),
                                  res_name=ai.GetProp('res_name'),
                                  res_id=ai.GetIntProp('res_id'),
                                  global_res_id=ai.GetIntProp('global_res_id'),
                                  chiral_tag=ai.GetChiralTag(),
                                  hybridization=ai.GetHybridization(),
                                  radical_electrons=ai.GetNumRadicalElectrons(),
                                  isotope=ai.GetIsotope(),
                                  body_id=body_id
                                  )
                if body_id >= 0:
                    has_body = True
                    mol_meta.nodes[i]['intra_mol_id'] = ai.GetIntProp('intra_mol_id')
                    if body_id not in rigid_groups:
                        rigid_groups[body_id] = set()
                    rigid_groups[body_id].add(i)
                    if ai.HasProp('x') and ai.HasProp('y') and ai.HasProp('z'):
                        mol_meta.nodes[i]['pos'] = np.array(
                            [float(ai.GetProp('x')), float(ai.GetProp('y')), float(ai.GetProp('z'))]) * 0.0001
        if has_body:
            mol_meta.graph['is_rigid'] = True
        mol_meta.graph['rigid_groups'] = {k: list(v) for k, v in rigid_groups.items()}
        for bond in tqdm.tqdm(mol.GetBonds(), total=mol.GetNumBonds(), desc='adding bonds', disable=tqdm_show):
            ai, aj = bond.GetBeginAtom(), bond.GetEndAtom()
            i, j = ai.GetIdx(), aj.GetIdx()
            mol_meta.add_edge(i, j, bond_type=bond.GetBondType(),
                              bondorder=bond.GetBondTypeAsDouble(),
                              bond_stereo=bond.GetStereo(),
                              bond_dir=bond.GetBondDir(),
                              stereo_atoms=list(bond.GetStereoAtoms()))
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
    for g in tqdm.tqdm(mols_meta, total=len(mols_meta), desc='converting nx.Graph to Chem.Mol', disable=True):
        mol = Chem.RWMol()
        nodes = g.nodes
        n_to_aid = {}

        # Phase 1: Add all nodes and restore full atomic attributes
        for n in g.nodes:
            atom = Chem.Atom(nodes[n]['atomic_num'])
            atom.SetIsAromatic(nodes[n]['is_aromatic'])
            atom.SetFormalCharge(nodes[n]['formal_charge'])
            atom.SetChiralTag(nodes[n]['chiral_tag'])
            atom.SetHybridization(nodes[n]['hybridization'])
            atom.SetNumRadicalElectrons(nodes[n]['radical_electrons'])
            atom.SetIsotope(nodes[n]['isotope'])

            # Custom identifiers
            if nodes[n].get('res_name') is not None:
                atom.SetProp('res_name', nodes[n]['res_name'])
            if nodes[n].get('res_id') is not None:
                atom.SetIntProp('res_id', nodes[n]['res_id'])
            if nodes[n].get('global_res_id') is not None:
                atom.SetIntProp('global_res_id', nodes[n]['global_res_id'])

            # MD body identifiers
            if 'body_id' in nodes[n]:
                atom.SetIntProp('body_id', nodes[n]['body_id'])
            if 'intra_mol_id' in nodes[n]:
                atom.SetIntProp('intra_mol_id', nodes[n]['intra_mol_id'])

            aid = mol.AddAtom(atom)
            n_to_aid[n] = aid

        # Phase 2: Add all edges and restore full bond attributes
        edges = g.edges
        for i, j in g.edges:
            mol.AddBond(
                n_to_aid[i],
                n_to_aid[j],
                edges[(i, j)]['bond_type']
            )

            # Retrieve the newly created bond to set stereo and direction flags
            bond = mol.GetBondBetweenAtoms(n_to_aid[i], n_to_aid[j])
            if edges[(i, j)].get('bond_dir') is not None:
                bond.SetBondDir(edges[(i, j)]['bond_dir'])
            # if (b.bond_stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE)) and (
            #                len(b.stereo_atoms) == 2):
            if ((edges[(i, j)].get('bond_stereo') is not None) and (edges[(i, j)].get('stereo_atoms') is not None)
                and (len(edges[(i, j)]['stereo_atoms']) == 2)) \
                    and (edges[(i, j)]['bond_stereo'] in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE)):
                bond.SetStereo(edges[(i, j)]['bond_stereo'])

                ai, aj = edges[(i, j)]['stereo_atoms']
                bond.SetStereoAtoms(ai, aj)

        mol = Chem.Mol(mol)
        Chem.SanitizeMol(mol)
        molecules.append(mol)
    return molecules


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
    box_tensor = [str(l) for l in box_tensor] + ['0'] * (9 - len(box_tensor)) if len(box_tensor) < 9 else [str(l) for l
                                                                                                           in
                                                                                                           box_tensor]
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


def sdf_load_all_as_one(input_path):
    suppl = Chem.SDMolSupplier(input_path, removeHs=False)
    rd_combined_mol = None

    for mol in suppl:
        if mol is None:
            continue

        num_atoms = mol.GetNumAtoms()

        if rd_combined_mol is None:
            rd_combined_mol = mol
        else:
            rd_combined_mol = Chem.CombineMols(rd_combined_mol, mol)
    return rd_combined_mol


def molecule_reader(input_path):
    if input_path.endswith('.sdf'):
        rdmol = sdf_load_all_as_one(input_path)
    elif input_path.endswith('.pdb'):
        # explicit bond only
        rdmol = Chem.MolFromPDBFile(input_path, removeHs=False, proximityBonding=False)
    else:
        raise ValueError(f"Unsupported file format: {input_path}. Only .sdf and .pdb for AA are supported.")
    return rdmol


def _extract_raw_data_from_xml(xml_path: str) -> Tuple[Dict[str, Any], np.ndarray]:
    """Extracts positions, bonds, types, and bodies from XML."""
    xml = XmlParser(xml_path)
    box_coords = (xml.box.lx, xml.box.ly, xml.box.lz)
    box_tensor = np.array(tuple(map(float, box_coords))) * 10

    raw_data = {
        'position': xml.data['position'] * 10,  # nm -> Å
        'type': list(xml.data['type']),
        'bond': [(str(b[0]), int(b[1]), int(b[2])) for b in xml.data['bond']],
        'body': list(xml.data.get('body', np.zeros(len(xml.data['type'])) - 1))
    }
    return raw_data, box_tensor


def _extract_raw_data_from_gsd(gsd_path: str) -> Tuple[Dict[str, Any], np.ndarray]:
    """Extracts positions, bonds, types, and bodies from GSD snapshot."""
    with gsd.hoomd.open(gsd_path, 'r') as traj:
        frame = traj[-1]

    box_tensor = np.array(frame.configuration.box[:3], dtype=float) * 10
    num_particles = frame.particles.N

    bodies = frame.particles.body
    if bodies is None or len(bodies) == 0:
        bodies = np.zeros(num_particles, dtype=np.int32) - 1

    bonds = []
    if hasattr(frame, 'bonds') and frame.bonds is not None and frame.bonds.N > 0:
        bond_groups = frame.bonds.group
        bond_types = frame.bonds.types
        bond_typeids = frame.bonds.typeid
        for b_idx in range(frame.bonds.N):
            u, v = bond_groups[b_idx]
            b_name = bond_types[bond_typeids[b_idx]]
            bonds.append((str(b_name), int(u), int(v)))

    raw_data = {
        'position': frame.particles.position * 10,  # nm -> Å
        'type': [frame.particles.types[tid] for tid in frame.particles.typeid],
        'bond': bonds,
        'body': [int(b) for b in bodies]
    }
    return raw_data, box_tensor


def _build_global_system_graph(raw_data: Dict[str, Any], reactants_config: Dict[str, Any]) -> nx.Graph:
    """Builds a global networkx graph and applies virtual edges to unify complex rigid topologies."""
    cg_sys = nx.Graph()
    num_nodes = len(raw_data['type'])

    # 1. First Pass: Map dynamic attributes onto nodes
    rigid_groups = {}  # Track nodes belonging to the same body_id for virtual stitching

    for tag in range(num_nodes):
        body_id = int(raw_data['body'][tag])
        raw_type = raw_data['type'][tag]
        pos = raw_data['position'][tag]

        # Default fallback initialization descriptors
        bead_type = raw_type
        smiles = reactants_config.get(raw_type, {}).get('smiles', None)

        if body_id >= 0:
            if body_id not in rigid_groups:
                rigid_groups[body_id] = []
            rigid_groups[body_id].append(tag)

        cg_sys.add_node(
            tag,
            type=bead_type,
            smiles=smiles,
            x=pos,
            body=body_id,
        )

    # 2. Second Pass: Inject real chemical bonds
    for bond_type, u, v in raw_data['bond']:
        cg_sys.add_edge(u, v, bond_type=bond_type, is_virtual=False)

    # 3. Third Pass: Virtual Stitching to mathematically lock multi-rigid/grafted systems
    for body_id, tags in rigid_groups.items():
        if len(tags) > 1:
            for i in range(len(tags) - 1):
                # Draw invisible topological bridges to consolidate the macro component
                cg_sys.add_edge(tags[i], tags[i + 1], is_virtual=True, bond_type="VIRTUAL")

    return cg_sys


def _final_all_rigid_id(cg_mol: nx.Graph) -> List[int]:
    """Returns a list of unique rigid body IDs present in the molecular graph."""
    return list(set(node_data['body'] for _, node_data in cg_mol.nodes(data=True) if node_data['body'] >= 0))


def _extract_chem_complete_submol(orig_mol: Chem.Mol, seed_atom_indices: list) -> Chem.Mol:
    keep_set = set(seed_atom_indices)
    for idx in seed_atom_indices:
        atom = orig_mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() == 'H':
                keep_set.add(nbr.GetIdx())
    rw_mol = Chem.RWMol(orig_mol)
    for idx in reversed(range(rw_mol.GetNumAtoms())):
        if idx not in keep_set:
            rw_mol.RemoveAtom(idx)
    sub_mol = rw_mol.GetMol()
    return sub_mol


def parse_body_configs(body_configs: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Converts body_configs from string keys to integer body_id keys."""
    parsed_configs = {}
    for body_type, config in body_configs.items():
        body_ids = config.get('body_idx', [])
        file = config.get('file')
        mapping = config.get('mapping', {})
        for body_id in body_ids:
            parsed_configs[body_id] = {
                'file': file,
                'mapping': mapping,
                'type': body_type
            }

    return parsed_configs


def _segment_and_purify_molecules(cg_sys: nx.Graph, box_tensor: np.ndarray, body_configs, use_extract_submol=False) -> \
List[nx.Graph]:
    """Segments the unified system graph into isolated molecule graphs and cleans metadata tags."""
    cg_mols = []
    # Extract independent components (shadow edges ensure rigid networks and grafts cluster perfectly)
    for component in nx.connected_components(cg_sys):
        # Create an isolated editable deep copy
        subgraph = cg_sys.subgraph(component).copy()

        # Unlink the temporary virtual edges to preserve immaculate chemical bond profiles
        virtual_edges = [(u, v) for u, v, d in subgraph.edges(data=True) if d.get('is_virtual', False)]
        subgraph.remove_edges_from(virtual_edges)

        # Inject Graph-level tracking parameters
        subgraph.graph['box'] = box_tensor
        total_nodes_count = len(subgraph.nodes)
        rigid_nodes_count = 0
        primary_type = None

        # Standardize local sequence indexes and inspect particle constraint states
        flexible_nodes = []
        rigid_groups = {}  # {body_id: [global_node_tags]}

        for node_idx in subgraph.nodes:
            body_id = subgraph.nodes[node_idx]['body']
            if body_id == -1:
                flexible_nodes.append(node_idx)
            else:
                if body_id not in rigid_groups:
                    rigid_groups[body_id] = []
                rigid_groups[body_id].append(node_idx)
        flexible_nodes.sort()
        for local_idx, node_idx in enumerate(flexible_nodes):
            subgraph.nodes[node_idx]['intra_mol_id'] = local_idx
        for body_id, nodes_list in rigid_groups.items():
            nodes_list.sort()
            for local_idx, node_idx in enumerate(nodes_list):
                subgraph.nodes[node_idx]['intra_mol_id'] = local_idx

        # -------------------------------------------------------------------------
        subgraph_rigid_configs = {body_id: body_configs[body_id] for body_id in rigid_groups if body_id in body_configs}
        subgraph.graph['rigid_configs'] = subgraph_rigid_configs
        for body_id in rigid_groups:

            file_path = body_configs[body_id].get('file')
            sites_config = body_configs[body_id].get('mapping', {})
            for node in subgraph.nodes:
                smarts = None
                smiles = None
                atom_index = None
                bead_type = subgraph.nodes[node]['type']
                local_cg_idx = subgraph.nodes[node]['intra_mol_id']
                if local_cg_idx in sites_config:
                    bead_type = sites_config[local_cg_idx].get('type', bead_type)
                    atom_index = sites_config[local_cg_idx].get('atom_index')
                    smiles = sites_config[local_cg_idx].get('smiles', smiles)
                    smarts = sites_config[local_cg_idx].get('smarts', smarts)

                    rigid_frag_mol = None
                    if smarts:
                        rigid_frag_mol = Chem.MolFromSmarts(smarts)
                    elif smiles:
                        rigid_frag_mol = Chem.MolFromSmiles(smiles)
                    elif atom_index and len(atom_index) == 1:
                        rigid_mol = molecule_reader(file_path)
                        atom_idx = atom_index[0]
                        atom = rigid_mol.GetAtomWithIdx(atom_idx)
                        smarts = f'[{atom.GetSymbol()}]'
                        rigid_frag_mol = Chem.MolFromSmarts(smarts)
                    else:
                        logger.error(
                            f"No valid SMARTS or SMILES provided for reaction bead {local_cg_idx} in body_id {body_id}. Please check the mapping configuration.")

                    if rigid_frag_mol is not None:
                        if use_extract_submol:
                            rigid_mol = molecule_reader(file_path)
                            represent_rigid_frag_mol = _extract_chem_complete_submol(rigid_mol, atom_index)
                            match = represent_rigid_frag_mol.GetSubstructMatch(rigid_frag_mol, useChirality=True)
                            rigid_frag_mol = Chem.RenumberAtoms(rigid_frag_mol, match)
                        rigid_frag_atom_mapping_ = {}
                        all_num_maps = [atom.GetAtomMapNum() for atom in rigid_frag_mol.GetAtoms()]
                        if len(all_num_maps) != len(set(all_num_maps)):
                            explicit_num_map = False
                        else:
                            explicit_num_map = True

                        for frag_atom_i in range(rigid_frag_mol.GetNumAtoms()):
                            atom = rigid_frag_mol.GetAtomWithIdx(frag_atom_i)
                            map_num = atom.GetAtomMapNum()
                            if explicit_num_map:
                                rigid_frag_atom_mapping_[map_num] = frag_atom_i
                            else:
                                rigid_frag_atom_mapping_[frag_atom_i] = frag_atom_i

                        sorted_num_maps = sorted(rigid_frag_atom_mapping_.keys())
                        frag_atom_mapping = {}
                        for i, map_num in enumerate(sorted_num_maps):
                            frag_atom_mapping[i] = rigid_frag_atom_mapping_[map_num]
                        subgraph.nodes[node]['frag_atom_mapping'] = frag_atom_mapping
                        subgraph.nodes[node]['smarts'] = smarts

        # -------------------------------------------------------------------------

        sorted_all_nodes = sorted(list(subgraph.nodes))
        for intra_id, node_idx in enumerate(sorted_all_nodes):
            subgraph.nodes[node_idx]['local_res_id'] = intra_id

        # 统计刚柔节点配比，锁死宏观刚性架构模式
        total_nodes_count = len(sorted_all_nodes)
        rigid_nodes_count = sum(len(tags) for tags in rigid_groups.values())

        # 2. 三路分流：精准确立分子的宏观刚性架构模式 (Rigidity Mode)
        if rigid_nodes_count == 0:
            # 状态 0：纯柔性分子
            subgraph.graph['is_rigid'] = False
            subgraph.graph['rigidity'] = 'FLEXIBLE'
            subgraph.graph['body_id'] = [-1]
            subgraph.graph['rigid_groups'] = {}
        elif rigid_nodes_count == total_nodes_count:
            # 状态 1：纯刚体结构
            subgraph.graph['is_rigid'] = True
            subgraph.graph['rigidity'] = 'RIGID'
            subgraph.graph['body_id'] = _final_all_rigid_id(subgraph)
            subgraph.graph['rigid_groups'] = rigid_groups
        else:
            # 状态 2：半刚半柔杂化体系（接枝分子）
            subgraph.graph['is_rigid'] = True
            subgraph.graph['rigidity'] = 'HYBRID'
            subgraph.graph['body_id'] = _final_all_rigid_id(subgraph)
            subgraph.graph['rigid_groups'] = rigid_groups

        # 3. 记录刚体的主导化学类型
        if primary_type:
            subgraph.graph['type'] = primary_type
        cg_mols.append(subgraph)

    return cg_mols


def _set_rigid_files(cg_mols, rigid_configs):
    """Sets the file paths for rigid molecules in the molecular graphs."""
    for cg_mol in cg_mols:
        if cg_mol.graph['is_rigid']:
            rigid_files = {}
            for body_id in cg_mol.graph['rigid_groups']:
                if body_id in rigid_configs:
                    file_path = rigid_configs[body_id].get('file')
                    if file_path:
                        rigid_files[body_id] = file_path
                    else:
                        logger.error(f"No file path specified for rigid body_id {body_id}.")
                        raise ValueError(
                            f"Missing file path for rigid body_id {body_id}. Please provide a valid file in rigid_configs.")
                else:
                    logger.error(f"Rigid body_id {body_id} not found in rigid_configs.")
                    raise ValueError(
                        f"Rigid body_id {body_id} is missing in rigid_configs. Please ensure all rigid bodies are defined.")
            cg_mol.graph['rigid_files'] = rigid_files
    return cg_mols


def parse_cg_topology(
        cg_configuration_file: str,
        reactants_config: Dict[str, Any],
        rigid_configs: Dict[int, Any] = None,
) -> Tuple[List[nx.Graph], np.ndarray]:
    """
    Unified Coarse-Grained Topology Parser Workshop.
    Standardizes inputs from either XML or GSD formats and builds metadata-synchronized
    molecular component networks optimized for hybrid rigid-grafted architectures.

    Args:
        cg_configuration_file (str): Path to the target simulation configuration (.xml or .gsd).
        reactants_config (dict): Global mapping dictionary of flexible bead types to SMILES info.
        rigid_configs (dict): Standard body_configs map tracking PDB links and localized reaction sites.

    Returns:
        Tuple[List[nx.Graph], np.ndarray]:
            - cg_mols: List of distinct molecular graphs packaged with complete fine-grained mapping vectors.
            - box_tensor: Scaled 3D bounding length matrix [Lx, Ly, Lz] in Angstrom units.
    """
    ext = os.path.splitext(cg_configuration_file)[-1].lower()
    rigid_configs = parse_body_configs(rigid_configs) if rigid_configs else {}
    # 1. Branch to localized raw file readers (Module 1)
    if ext == '.xml':
        raw_data, box_tensor = _extract_raw_data_from_xml(cg_configuration_file)
    elif ext == '.gsd':
        raw_data, box_tensor = _extract_raw_data_from_gsd(cg_configuration_file)
    else:
        raise ValueError(f"Unsupported structural layout extension '{ext}'. Route matching failed.")

    # 2. Build the unified network containing shadow link bridges (Module 2)
    if rigid_configs is None:
        rigid_configs = {}
    global_system_graph = _build_global_system_graph(raw_data, reactants_config)

    # 3. Segregate molecules and strip validation tracking edges (Module 3)
    cg_mols = _segment_and_purify_molecules(global_system_graph, box_tensor, rigid_configs)

    cg_mols = _set_rigid_files(cg_mols, rigid_configs)

    logger.info(f"Successfully compiled system topology: Harvester collected "
                f"{len(cg_mols)} isolated molecular networks.")
    return cg_mols, box_tensor
