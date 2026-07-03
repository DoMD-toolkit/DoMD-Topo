import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Dict, List, Tuple
from domd_xyz.embed.optimize_orientation import Meta, optimize_res_orientation
from misc.logger import logger
from domd_xyz.embed.embed_with_cg_xyz import (
    generate_local_fragment_coords,
    analyze_topology,
    get_best_alignment,
    rotate_confs,
    pbc
)


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
    for bond in molecule.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
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
    atom_pos_initial = np.zeros((molecule.GetNumAtoms(), 3))
    atom_res_ids = np.zeros(molecule.GetNumAtoms(), dtype=np.int64)

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
    logger.info("Solving analytical rotation matrices across localized boundaries...")
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
                local2atoms: Dict[int, List[int]],
                box: np.ndarray) -> Chem.Mol:
    """
    Rigid Aligner Workshop: Rigidity-preserving alignment using unified topology metadata.

    Calculates virtual geometric centers of all-atom residues based on 'local2atoms',
    aligns them via PCA/RMSD optimization to target CG coordinates, and maps the
    transformation globally across the rigid molecule.

    Args:
        molecule (Chem.Mol): All-atom RDKit molecule with an initial conformation.
        cg_molecule (nx.Graph): Coarse-grained graph containing target coordinates 'x'.
        local2atoms (Dict[int, List[int]]): Mapping of local_res_id -> atom indices.
        box (np.ndarray): Simulation box dimensions for PBC wrapping.

    Returns:
        np.ndarray: The finalized, aligned 3D coordinates for all atoms.
    """
    # 1. Fetch the raw initial all-atom coordinates
    conf = molecule.GetConformer(0)
    aa_pos = conf.GetPositions()

    aa_rigid_pos = []
    cg_rigid_pos = []

    # 2. Extract paired coordinates, strictly anchored by CG node order to guarantee alignment
    for node in cg_molecule.nodes:
        # Fetch target CG position
        cg_rigid_pos.append(cg_molecule.nodes[node]['x'])

        # Fetch corresponding initial AA atom positions using our standardized local_res_id
        local_res_id = cg_molecule.nodes[node]['local_res_id']
        atom_indices = local2atoms[local_res_id]

        # Calculate the geometric mean position of this residue in initial AA space
        aa_rigid_pos.append(np.mean(aa_pos[atom_indices], axis=0))

    # Convert to standard NumPy arrays for high-performance matrix operations
    cg_rigid_pos = np.array(cg_rigid_pos)
    aa_rigid_pos = np.array(aa_rigid_pos)

    # 3. Perform rigid-body optimization (Rotation & Translation search)
    # get_best_alignment and rotate_confs remain identical to your mathematical primitives
    _, best_R, _, target_com = get_best_alignment(cg_rigid_pos, aa_rigid_pos, box)

    # 4. Apply transformation globally to all constituent atoms and wrap via PBC
    transformed_aa_pos = rotate_confs(aa_pos, best_R, box, target_com)
    conf = Chem.Conformer()
    r_aa_pos = pbc(transformed_aa_pos, box)
    for i, p in enumerate(r_aa_pos):
        conf.SetAtomPosition(i, p)
    #if molecule.GetNumConformers() == 0 and conf is not None:
    #    molecule.AddConformer(conf, assignId=True)
    return conf


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
    #if molecule.GetNumConformers() == 0 and conf is not None:
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


from typing import Tuple
from misc.logger import logger


def embed_molecule(
        molecule: Chem.Mol,
        cg_molecule: nx.Graph,
        molecule_graph: nx.Graph,
        box: np.ndarray = None,
        large: int = 500,
        chunk_per_d: int = 1
) -> Tuple[Chem.Mol, nx.Graph]:
    """
    The Ultimate Top-Level Orchestrator. Standardizes topology maps, routes the system
    to specialized solvers to fetch a Conformer, updates all properties, and returns both objects.

    Args:
        molecule (Chem.Mol): All-atom RDKit molecule topology.
        cg_molecule (nx.Graph): Coarse-grained configuration template layout.
        molecule_graph (nx.Graph): Global all-atom molecular network metadata.
        box (np.ndarray, optional): Simulation box dimensions. Fallback provided if None.
        large (int, default 500): Atom count threshold defining the macro-system boundary.
        chunk_per_d (int, default 1): Spatial grid subdivisions for optimization.

    Returns:
        Tuple[Chem.Mol, nx.Graph]:
            - molecule: Standardized RDKit molecule containing the successfully bound 3D Conformer.
            - molecule_graph: The molecular graph injected with 3D coordinate tensors under node attribute 'x'.
    """
    # Step 1: Execute Workshop A (Topology Analyzer) to standardize unified res_id signatures
    global2local, local2atoms = analyze_topology(molecule_graph, cg_molecule)

    # Infinite fallback boundary handling if box parameters are omitted
    if box is None:
        box = np.ones(3) * 10000.0
        logger.warning(f"Simulation box bounds missing. Temporarily fallback to: {box}")

    # Step 2: Conformer Extraction Pass via clean 3-way branching logic
    # --- MODE 1: Rigid-Body Architecture System ---
    if cg_molecule.graph.get('is_rigid', False):
        logger.info("Routing system straight into the Rigid Aligner Workshop.")
        conf = embed_rigid(molecule, cg_molecule, local2atoms, box)

    # --- MODE 2: Small Non-Rigid System (Standard ETKDG Route) ---
    elif molecule.GetNumAtoms() <= large:
        logger.info(f"System size ({molecule.GetNumAtoms()} atoms) <= threshold ({large}). Routing to embed_by_etkdg.")
        conf = embed_by_etkdg(molecule, cg_molecule, molecule_graph, local2atoms, box, chunk_per_d)

    # --- MODE 3: Massive Macromolecular System (Fragment Solver Route) ---
    else:
        logger.info(
            f"System size ({molecule.GetNumAtoms()} atoms) > threshold ({large}). Routing to embed_by_fragment.")
        conf = embed_by_fragment(molecule, molecule_graph, cg_molecule, local2atoms, box, chunk_per_d)

    # Step 3: Global State Injection
    if conf is not None:
        # Clear any residual un-optimized conformation layouts
        if molecule.GetNumConformers() > 0:
            molecule.RemoveAllConformers()

        # Commit the generated 3D conformer directly into the RDKit molecule
        molecule.AddConformer(conf, assignId=True)

        # Extract atomic 3D coordinates and inject them as 'x' attributes into the molecular graph
        for atom in molecule.GetAtoms():
            a_id = atom.GetIdx()
            pos = conf.GetAtomPosition(a_id)

            # Bound natively as a high-performance 3D NumPy coordinate tensor
            molecule_graph.nodes[a_id]['x'] = np.array([pos.x, pos.y, pos.z])

        logger.info("Successfully bound Conformer properties and mapped 'x' tensor coordinates into the graph.")
    else:
        logger.error("Failed to generate valid geometric parameters across all available workshops.")

    return molecule, molecule_graph