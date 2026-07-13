from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .embed_with_cg_xyz import (
    generate_local_fragment_coords,
    analyze_topology,
    get_best_alignment,
    rotate_confs,
    pbc
)
from .optimize_orientation import Meta, optimize_res_orientation
from ..misc.logger import logger
from ..misc.parser import nxgraphs_to_mols


def assemble_and_stitch_system(
        molecule: Chem.Mol,
        molecule_graph: nx.Graph,
        cg_graph: nx.Graph,
        local2atoms: Dict[int, List[int]],
        all_local_coords: Dict[int, np.ndarray],
        box: np.ndarray,
        chunks_per_d: int
) -> Chem.Conformer:
    """
    Workshop E: Global Stitching Aligner. Solves orientation optimization matrices,
    executes global 3D rotations/translations, and commits coordinates to a new conformer.
    """
    n_residues = len(local2atoms)
    bonds = []
    local_frame_idx = []

    # 1. Parse inter-residue connecting bonds to construct topological linkers metadata
    for u, v in molecule_graph.edges:
        res_u = molecule_graph.nodes[u]['res_id']
        res_v = molecule_graph.nodes[v]['res_id']

        # Identify cross-boundary linkages spanning between different residues
        if res_u != res_v:
            bonds.append((res_u, res_v))
            local_frame_idx.append((u, v))

    # 2. Extract target mapping translation vectors from normalized CG template positions
    trans = np.zeros((n_residues, 3))
    for node in cg_graph.nodes:
        local_res_id = cg_graph.nodes[node]['local_res_id']
        trans[local_res_id] = cg_graph.nodes[node].get("x")

    # 3. Flatten compiled atomic properties into arrays for Numba/SciPy performance loops
    atom_pos_initial = np.zeros((molecule_graph.number_of_nodes(), 3))
    atom_res_ids = np.zeros(molecule_graph.number_of_nodes(), dtype=np.int64)

    for g_id, coord in all_local_coords.items():
        atom_pos_initial[g_id] = coord
        atom_res_ids[g_id] = molecule_graph.nodes[g_id]['res_id']

    # 4. Initialize structural meta descriptors block
    meta = Meta(
        np.array(bonds, dtype=np.int64),
        trans,
        np.array(local_frame_idx, dtype=np.int64),
        atom_pos_initial,
        atom_res_ids,
        box
    )

    # 5. Invoke numerical solver to optimize rotational orientation matrices
    logger.info("Optimize cross-cg-bead rotation matrix...")
    rot = optimize_res_orientation(n_residues, meta, chunk_per_d=chunks_per_d)

    # 6. Stitching Pass: Map final rigid transformation [Coord * Rot.T + Trans] under PBC limits
    final_conformer = Chem.Conformer()
    for local_res_id, atom_ids in local2atoms.items():
        for g_id in atom_ids:
            # Perform rigid matrix rotation based on optimized orientation directions
            rotated_coord = rot[local_res_id].dot(all_local_coords[g_id])
            # Translate directly to the matched Coarse-Grained space target coordinate
            final_global_coord = rotated_coord + trans[local_res_id]

            # Commit wrapped PBC position straight into the RDKit conformer storage
            final_conformer.SetAtomPosition(g_id, pbc(final_global_coord, box))

    return final_conformer


def embed_rigid(molecule: Chem.Mol,
                cg_molecule: nx.Graph,
                box: np.ndarray) -> Chem.Mol:
    """
    Rigid Aligner Workshop: Rigidity-preserving alignment using unified topology metadata.

    Calculates virtual geometric centers of all-atom residues based on 'local2atoms',
    aligns them via PCA/RMSD optimization to target CG coordinates, and maps the
    transformation globally across the rigid molecule.

    Args:
        molecule (Chem.Mol): All-atom RDKit molecule with an initial conformation.
        cg_molecule (nx.Graph): Coarse-grained graph containing target coordinates 'x'.
        box (np.ndarray): Simulation box dimensions for PBC wrapping.

    Returns:
        np.ndarray: The finalized, aligned 3D coordinates for all atoms.
    """
    # 1. Fetch the raw initial all-atom coordinates
    conf = molecule.GetConformer(0)
    aa_pos = conf.GetPositions()

    cg_rigid_pos = []

    # 2. Extract paired coordinates, strictly anchored by CG node order to guarantee alignment
    for node in cg_molecule.nodes:
        # Fetch target CG position
        cg_rigid_pos.append(cg_molecule.nodes[node]['x'])
    cg_rigid_pos = np.array(cg_rigid_pos)
    # 3. Perform rigid-body optimization (Rotation & Translation search)
    # get_best_alignment and rotate_confs remain identical to your mathematical primitives
    body_id = cg_molecule.graph.get('body_id')[0]
    pairs = cg_molecule.graph.get('rigid_pairs', {}).get(body_id)
    if pairs is not None:
        paired_A = pairs['paired_CG']
        paired_B = pairs['paired_AA']
    else:
        paired_A, paired_B = None, None
    best_R, target_com = get_best_alignment(cg_rigid_pos, aa_pos, box, paired_A=paired_A, paired_B=paired_B)

    # 4. Apply transformation globally to all constituent atoms and wrap via PBC
    transformed_aa_pos = rotate_confs(aa_pos, best_R, box, target_com)
    conf = Chem.Conformer()
    r_aa_pos = pbc(transformed_aa_pos, box)
    for i, p in enumerate(r_aa_pos):
        conf.SetAtomPosition(i, p)
    # if molecule.GetNumConformers() == 0 and conf is not None:
    #    molecule.AddConformer(conf, assignId=True)
    return conf


def embed_hybrid(
        molecule: Chem.Mol,
        molecule_graph: nx.Graph,
        cg_graph: nx.Graph,
        local2atoms: Dict[int, List[int]],
        box: np.ndarray,
        large: int = 500,
        chunks_per_d: int = 1
) -> Chem.Conformer:
    """
    Specialized Hybrid Solver. Extracts pre-aligned global coordinates for rigid bodies
    from template layout files, constructs isolated pseudo-flexible topological graphs
    for the remaining flexible chains plus their junction anchors, and runs analytical
    stitching optimization.
    """
    logger.info("Executing specialized hybrid anchor-and-grow embedding workflow.")
    all_local_coords: Dict[int, np.ndarray] = {}

    # -----------------------------------------------------------------
    # Rigid Pre-alignment
    # -----------------------------------------------------------------
    rigid_groups = cg_graph.graph.get('rigid_groups', {})
    rigid_transforms: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    rigid_files = cg_graph.graph.get('rigid_files', {})
    aa_rigid_groups = molecule_graph.graph.get('rigid_groups', {})

    if not aa_rigid_groups:
        for i in molecule_graph.nodes:
            body_id = molecule_graph.nodes[i].get('body', -1)
            if body_id >= 0:
                aa_rigid_groups.setdefault(body_id, set()).add(i)

    if not aa_rigid_groups:
        logger.error("No rigid groups detected in the all-atom molecular graph. Cannot proceed.")
        raise ValueError("No rigid groups detected in the all-atom molecular graph.")

    for body_id, cg_nodes in rigid_groups.items():
        if body_id == -1:
            continue

        # get the target CG positions for the current rigid body
        cg_rigid_pos = np.array([cg_graph.nodes[n]['x'] for n in sorted(cg_nodes)])

        # get the corresponding original all-atom positions for the rigid body
        aa_rigid_atoms_global = sorted(aa_rigid_groups[body_id])
        aa_rigid_atoms_intra = np.arange(len(aa_rigid_atoms_global))
        for global_id, intra_id in zip(aa_rigid_atoms_global, aa_rigid_atoms_intra):
            molecule_graph.nodes[global_id]['intra_mol_id'] = int(intra_id)
        raw_template_positions = np.array(
            [molecule_graph.nodes[aa_id]['pos'] for aa_id in sorted(aa_rigid_atoms_intra)])

        # get the global atom IDs for the current rigid body from the all-atom molecular graph
        global_aa_ids = sorted(list(aa_rigid_groups.get(body_id, [])))

        # refine the intra_mol_id for each atom in the rigid body based on the global atom IDs
        # the intra_mol_id may be changed compared to the original one in the template file, because the
        # rigid body may react with other molecules and some atoms (e.g. H, H2O) may be removed,
        # so we need to reassign the intra_mol_id based on the global atom IDs
        for index, aa_id in enumerate(global_aa_ids):
            if molecule_graph.nodes[aa_id].get('intra_mol_id') is None:
                molecule_graph.nodes[aa_id]['intra_mol_id'] = index

        aligned_aa_pos_list = []
        for aa_id in global_aa_ids:
            intra_atom_id = molecule_graph.nodes[aa_id]['intra_mol_id']
            aligned_aa_pos_list.append(raw_template_positions[intra_atom_id])

        aa_rigid_pos = np.array(aligned_aa_pos_list)

        # use get_best_alignment to compute the optimal rotation via PCA-based alignment
        # if there are paired atoms between CG and AA, use them to compute the optimal rotation via Kabsch algorithm
        pairs = cg_graph.graph.get('rigid_pairs', {}).get(body_id)
        if pairs is not None:
            paired_A = pairs['paired_CG']
            paired_B = pairs['paired_AA']
        else:
            paired_A, paired_B = None, None
        best_R, target_com = get_best_alignment(cg_rigid_pos, aa_rigid_pos, box, paired_A=paired_A, paired_B=paired_B)
        rigid_transforms[body_id] = (best_R, target_com)

        # rotate and translate the rigid body atoms to the target CG positions
        transformed_aa_pos = rotate_confs(aa_rigid_pos, best_R, box, target_com)
        for idx, aa_id in enumerate(global_aa_ids):
            all_local_coords[aa_id] = transformed_aa_pos[idx]

    # -----------------------------------------------------------------
    # Pseudo Graphs Construction
    # -----------------------------------------------------------------
    flex_cg_nodes = [n for n in cg_graph.nodes if cg_graph.nodes[n]['body'] == -1]
    rigid_boundary_cg_nodes = set()

    # Search for all rigid boundary CG beads that are bonded to flexible chains
    for u, v in cg_graph.edges:
        body_u = cg_graph.nodes[u]['body']
        body_v = cg_graph.nodes[v]['body']
        if body_u == -1 and body_v >= 0:
            rigid_boundary_cg_nodes.add(v)
        if body_v == -1 and body_u >= 0:
            rigid_boundary_cg_nodes.add(u)

    pseudo_cg_nodes = sorted(list(set(flex_cg_nodes).union(rigid_boundary_cg_nodes)))

    # 2a. Construct a new coarse-grained pseudo graph `pseudo_cg_graph` that includes only flexible
    # residues and the rigid boundary residues that connect to them.
    pseudo_cg_graph = nx.Graph()
    pseudo_cg_graph.graph['is_rigid'] = False
    pseudo_cg_graph.graph['rigidity'] = 'FLEXIBLE'
    pseudo_cg_graph.graph['box'] = box

    old_cg2pseudo_res = {}
    for pseudo_res_id, old_node in enumerate(pseudo_cg_nodes):
        old_cg2pseudo_res[old_node] = pseudo_res_id
        attrs = cg_graph.nodes[old_node].copy()
        attrs['local_res_id'] = pseudo_res_id
        pseudo_cg_graph.add_node(old_node, **attrs)

    for u, v in cg_graph.edges:
        if u in pseudo_cg_graph.nodes and v in pseudo_cg_graph.nodes:
            pseudo_cg_graph.add_edge(u, v, **cg_graph.edges[u, v])

    # 2b. Construct a new all-atom pseudo graph `pseudo_molecule_graph` that includes only flexible
    # atoms and the rigid boundary atoms that connect to them.
    pseudo_molecule_graph = nx.Graph()
    old_aa2pseudo_aa = {}
    pseudo_aa_idx = 0

    for old_node in pseudo_cg_nodes:
        orig_local_res_id = cg_graph.nodes[old_node]['local_res_id']
        atom_ids = local2atoms[orig_local_res_id]
        body_id = cg_graph.nodes[old_node]['body']
        p_res_id = old_cg2pseudo_res[old_node]

        if body_id >= 0:
            junction_atoms = [aa_id for aa_id in atom_ids if any(
                molecule_graph.nodes[nbr].get('body', -1) == -1 for nbr in molecule_graph.neighbors(aa_id))]
            target_atoms = junction_atoms if junction_atoms else [atom_ids[0]]
        else:
            target_atoms = atom_ids

        for aa_id in target_atoms:
            old_aa2pseudo_aa[aa_id] = pseudo_aa_idx
            attrs = molecule_graph.nodes[aa_id].copy()
            attrs['res_id'] = p_res_id
            pseudo_molecule_graph.add_node(pseudo_aa_idx, **attrs)
            pseudo_aa_idx += 1

    for u, v in molecule_graph.edges:
        if u in old_aa2pseudo_aa and v in old_aa2pseudo_aa:
            pseudo_molecule_graph.add_edge(old_aa2pseudo_aa[u], old_aa2pseudo_aa[v], **molecule_graph.edges[u, v])

    # 2c. Re-invoke standard analyze_topology to generate a new local2atoms mapping for the pseudo graph system
    _, pseudo_local2atoms = analyze_topology(pseudo_molecule_graph, pseudo_cg_graph)
    pseudo_molecule = nxgraphs_to_mols([pseudo_molecule_graph])[0]
    total_pseudo_atoms = pseudo_molecule.GetNumAtoms()

    if total_pseudo_atoms <= large:
        logger.info(
            f"Flexible part in hybrid system ({total_pseudo_atoms} atoms) <= threshold. Invoking embed_by_etkdg.")
        pseudo_conf = embed_by_etkdg(pseudo_molecule, pseudo_cg_graph, pseudo_molecule_graph, pseudo_local2atoms, box,
                                     chunks_per_d)
    else:
        logger.info(
            f"Flexible part in hybrid system ({total_pseudo_atoms} atoms) > threshold. Invoking embed_by_fragment.")
        pseudo_conf = embed_by_fragment(pseudo_molecule, pseudo_molecule_graph, pseudo_cg_graph, pseudo_local2atoms,
                                        box, chunks_per_d)

    # -----------------------------------------------------------------
    #  Assembly
    # -----------------------------------------------------------------
    for old_aa_id, p_aa_id in old_aa2pseudo_aa.items():
        if molecule_graph.nodes[old_aa_id]['body_id'] == -1:
            pos = pseudo_conf.GetAtomPosition(p_aa_id)
            all_local_coords[old_aa_id] = np.array([pos.x, pos.y, pos.z])

    final_full_conformer = Chem.Conformer()
    for g_id, coord in all_local_coords.items():
        final_full_conformer.SetAtomPosition(g_id, pbc(coord, box))

    return final_full_conformer


def embed_by_fragment(
        molecule: Chem.Mol,
        molecule_graph: nx.Graph,
        cg_graph: nx.Graph,
        local2atoms: Dict[int, List[int]],
        box: np.ndarray,
        chunks_per_d: int = 1
) -> Chem.Conformer:
    """
    Specialized Fragment Solver. Only invoked for non-rigid, large macromolecular systems.
    Orchestrates sequential fragment embedding and global spatial stitching.

    Args:
        molecule (Chem.Mol): All-atom RDKit molecule topology.
        molecule_graph (nx.Graph): Global all-atom molecular network metadata.
        cg_graph (nx.Graph): Coarse-grained graph configuration layout.
        local2atoms (Dict[int, List[int]]): Standardized mapping of local_res_id -> atom indices.
        box (np.ndarray): Simulation box bounds.
        chunks_per_d (int, default 1): Spatial grid slicing factor for orientation optimization.

    Returns:
        Chem.Conformer: The finalized, fully stitched 3D conformer for the large system.
    """
    logger.info("Executing specialized fragment-based embedding workflow for large system.")
    all_local_coords: Dict[int, np.ndarray] = {}
    adj_dict = dict(cg_graph.adjacency())

    # 1. Sequentially drive fragment generators across every coarse-grained residue node
    for cg_node in cg_graph.nodes:
        local_res_id = cg_graph.nodes[cg_node]['local_res_id']

        # Discover neighboring residues mapped as CG node index keys
        neighbor_cg_nodes = list(adj_dict[cg_node].keys())
        neighbor_local_ids = [cg_graph.nodes[nb]['local_res_id'] for nb in neighbor_cg_nodes]

        # Generate local zero-centered coordinates for this specific fragment block
        residue_local_coords = generate_local_fragment_coords(
            molecule, molecule_graph, local2atoms, local_res_id, neighbor_local_ids
        )
        all_local_coords.update(residue_local_coords)

    # 2. Invoke Workshop E to optimize orientations and sew the fragments together in global space
    final_conformer = assemble_and_stitch_system(
        molecule, molecule_graph, cg_graph, local2atoms, all_local_coords, box, chunks_per_d
    )
    # if molecule.GetNumConformers() == 0 and conf is not None:
    #    molecule.AddConformer(final_conformer, assignId=True)
    return final_conformer


def embed_by_etkdg(
        molecule: Chem.Mol,
        cg_molecule: nx.Graph,
        molecule_graph: nx.Graph,
        local2atoms: Dict[int, List[int]],
        box: np.ndarray,
        chunk_per_d: int = 1
) -> Chem.Conformer:
    """
    Standard ETKDG Solver. Only invoked for non-rigid systems smaller than the 'large' threshold.
    Generates initial global coordinates via ETKDG, then applies global alignment stitching.

    Args:
        molecule (Chem.Mol): All-atom RDKit molecule topology.
        cg_molecule (nx.Graph): Coarse-grained graph template.
        molecule_graph (nx.Graph): Global all-atom molecular network metadata.
        local2atoms (Dict[int, List[int]]): Mapping of local_res_id -> atom indices.
        box (np.ndarray): Simulation box bounds.
        chunk_per_d (int, default 1): Spatial grid slicing factor for orientation optimization.

    Returns:
        Chem.Conformer: The finalized, aligned 3D conformer for the small system.
    """
    conf_id = -1
    attempts = 10000

    # Determine if random coordinates are needed based on structural chiral centers
    has_chiral = any(
        len(AllChem.FindMolChiralCenters(Chem.AddHs(Chem.MolFromSmiles(cg_molecule.nodes[n]['smiles'])))) > 0
        for n in cg_molecule.nodes
    )
    use_random = not has_chiral

    # Iterative global distance geometry embedding loop
    while conf_id == -1:
        conf_id = AllChem.EmbedMolecule(molecule, useRandomCoords=use_random, maxAttempts=attempts)
        attempts += 5000

    base_conf = molecule.GetConformer(conf_id)

    # Normalize fragments at origin to match the global stitcher's protocol
    all_local_coords = {a.GetIdx(): np.array(base_conf.GetAtomPosition(a.GetIdx())) for a in molecule.GetAtoms()}

    # Pass through the global stitcher to correctly align and translate to CG positions
    return assemble_and_stitch_system(
        molecule, molecule_graph, cg_molecule, local2atoms, all_local_coords, box, chunk_per_d
    )
