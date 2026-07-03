from typing import Union, Any, Dict, List, Tuple, Set
import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from misc.logger import logger
from scipy.stats import circmean
import numba as nb


@nb.jit(nopython=True)
def pbc(x, l):
    """Applies periodic boundary conditions to a coordinate or vector.

        Args:
            x (float or np.ndarray): Input coordinate.
            l (float or np.ndarray): Box length(s).

        Returns:
            float or np.ndarray: The wrapped coordinate within [-l/2, l/2].
    """
    return x - l * np.rint(x / l)


def get_best_alignment(coords_A, coords_B, box):
    r"""Aligns molecule B to molecule A by minimizing RMSD under PBC.

    This function performs a rigid body alignment (rotation and translation) using
    Principal Component Analysis (PCA) on the gyration tensors. It resolves the
    eigenvector sign ambiguity by exhaustively checking all axis permutations.

    The algorithm proceeds as follows:

    1.  **Centering**: Compute the Center of Mass (COM) for both molecules using
        circular mean to handle Periodic Boundary Conditions (PBC).
        .. math::
            X_{centered} = X - \text{COM}(X)
    2.  **Gyration Tensor**: Compute the covariance matrix (Gyration Tensor) roughly
        representing the moment of inertia.

        .. math::
            R_g = \frac{1}{N} X_{centered}^T X_{centered}

    3.  **Eigendecomposition**: Obtain the principal axes (eigenvectors $V$) by decomposing $R_g$.

        .. math::
            R_g = V \Lambda V^T

    4.  **Optimal Rotation Search**: Construct candidate rotation matrices $R$ by mapping
        the principal axes of B ($V_B$) to A ($V_A$), iterating through all possible
        sign flips $S$ (diagonal matrix with $\pm 1$).

        .. math::
            R = V_A \cdot S \cdot V_B^T

        The algorithm selects the $R$ that minimizes the Root Mean Square Deviation (RMSD)
        and ensures a proper rotation ($\det(R) = 1$).

    Args:
        coords_A (np.ndarray): Coordinates of the reference molecule A, shape (N, 3).
        coords_B (np.ndarray): Coordinates of the mobile molecule B, shape (N, 3).
        box (np.ndarray): Simulation box dimensions [Lx, Ly, Lz] for PBC handling.

    Returns:
        tuple: A tuple containing:
            - best_rotated_B (np.ndarray): The aligned coordinates of B.
            - best_R (np.ndarray): The optimal 3x3 rotation matrix.
            - best_rmsd (float): The minimum RMSD achieved.
            - comA (np.ndarray): The center of mass of A (used for final translation).
    """
    nA, D = coords_A.shape
    comA = np.array(
        [circmean(coords_A[:, i:i + 1], low=-box[i] / 2., high=box[i] / 2., axis=0) for i in range(D)]).ravel()
    comB = np.array(
        [circmean(coords_B[:, i:i + 1], low=-box[i] / 2., high=box[i] / 2., axis=0) for i in range(D)]).ravel()
    cA = pbc(coords_A - comA, box)  # coords_A- np.mean(coords_A, axis=0)
    cB = pbc(coords_B - comB, box)  # coords_B - np.mean(coords_B, axis=0)

    RgA = np.dot(cA.T, cA) / len(cA)
    RgB = np.dot(cB.T, cB) / len(cB)

    def get_sorted_eigenvectors(rg_tensor):
        vals, vecs = np.linalg.eigh(rg_tensor)
        idx = np.argsort(vals)[::-1]
        return vecs[:, idx]

    VA = get_sorted_eigenvectors(RgA)
    VB = get_sorted_eigenvectors(RgB)

    best_rmsd = float('inf')
    best_rotated_B = None
    best_R = None

    import itertools
    for signs in itertools.product([1, -1], repeat=3):
        sign_mat = np.diag(signs)

        R_candidate = VA @ sign_mat @ VB.T

        if np.isclose(np.linalg.det(R_candidate), 1.0):
            rotated_B = np.dot(cB, R_candidate.T)

            diff = cA - rotated_B
            rmsd = np.sqrt(np.mean(diff ** 2))

            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_rotated_B = rotated_B
                best_R = R_candidate

    return best_rotated_B, best_R, best_rmsd, comA


def rotate_confs(pos, R, box, com_TP):
    """Applies a rotation and translation to a set of coordinates under PBC.

        Args:
            pos (np.ndarray): Input coordinates (N, 3).
            R (np.ndarray): Rotation matrix (3, 3).
            box (np.ndarray): Box dimensions.
            com_TP (np.ndarray): Target center of mass position (translation vector).

        Returns:
            np.ndarray: Transformed coordinates.
    """
    N, D = pos.shape
    com = np.array([circmean(pos[:, i:i + 1], low=-box[i] / 2., high=box[i] / 2., axis=0) for i in range(D)]).ravel()
    cA = pbc(pos - com, box)
    r_cA = np.dot(cA, R.T)
    return r_cA + com_TP


def analyze_topology(molecule_graph: nx.Graph, cg_graph: nx.Graph) -> Tuple[Dict[Any, int], Dict[int, List[int]]]:
    """
    Topology Analyzer: Standardizes and bridges residue ID mappings between AA and CG graphs.

    This manager resolves the mapping between 'global_res_id' (CG node key) and
    'local_res_id' (0-indexed sequence used for internal matrix optimizations).
    It injects the unified 'res_id' back into the molecule_graph.

    Args:
        molecule_graph (nx.Graph): All-atom molecular graph. Nodes must contain 'global_res_id'.
        cg_graph (nx.Graph): Coarse-grained graph template.

    Returns:
        Tuple[Dict, Dict]:
            - global2local: Maps global_res_id -> local_res_id
            - local2atoms: Maps local_res_id -> List of atom indices within this residue
    """
    # Step 1: Strict Validation - global_res_id is a mandatory prerequisite
    for atom_id, data in molecule_graph.nodes(data=True):
        if 'global_res_id' not in data:
            raise KeyError(
                f"Mandatory attribute 'global_res_id' missing at atom node {atom_id}. "
                f"The topology analyzer cannot map atoms to their coarse-grained counterparts."
            )

    global2local: Dict[Any, int] = {}
    # Step 2: Extract existing mapping metadata from both graphs if available
    # Check if cg_graph has predefined 'local_res_id'
    #for cg_node, data in cg_graph.nodes(data=True):
    #    if 'local_res_id' in data:
    #        global2local[cg_node] = data['local_res_id']

    ## Cross-reference with molecule_graph's 'res_id' to complement or verify mapping
    #for atom_id, data in molecule_graph.nodes(data=True):
    #    g_id = data['global_res_id']
    #    if 'res_id' in data:
    #        r_id = data['res_id']
    #        # Integrity check: Ensure no conflicting mappings exist between the two graphs
    #        if g_id in global2local and global2local[g_id] != r_id:
    #            molecule_graph.nodes[atom_id]['res_id'] = global2local[g_id]

    # Step 3: Fallback Logic - If no local tracking IDs were provided, build from scratch
    if not global2local or len(global2local) < len(cg_graph):
        logger.debug("No local_res_id detected. Generating 0-indexed local sequences from cg_graph.")
        for idx, cg_node in enumerate(cg_graph.nodes):
            global2local[cg_node] = idx

    # Step 4: Back-propagate finalized information and compile outputs
    # Synchronize cg_graph attributes
    for cg_node in cg_graph.nodes:
        cg_graph.nodes[cg_node]['local_res_id'] = global2local[cg_node]

    # Initialize the atom accumulator dictionary
    local2atoms: Dict[int, List[int]] = {local_id: [] for local_id in global2local.values()}
    # Update molecule_graph and harvest atom lists
    for atom_id, data in molecule_graph.nodes(data=True):
        g_id = data['global_res_id']
        local_id = global2local[g_id]
        # Inject the standardized token back into the all-atom graph reference
        data['res_id'] = local_id
        local2atoms[local_id].append(atom_id)
    logger.info(f"Topology analysis complete. Successfully mapped {len(global2local)} residues.")
    return global2local, local2atoms


def build_isolated_fragment(
        molecule: Chem.Mol,
        molecule_graph: nx.Graph,
        local2atoms: Dict[int, List[int]],
        target_res_id: int,
        neighbor_res_ids: List[int]
) -> Tuple[Chem.RWMol, Dict[int, int]]:
    """
    Fragment & Frontier Repair Workshop: Safely extracts a local molecular subgraph
    and repairs broken aromatic/conjugated systems at the cutting frontiers.

    Args:
        molecule (Chem.Mol): Full all-atom reference molecule.
        molecule_graph (nx.Graph): All-atom molecular graph with synchronized 'res_id'.
        local2atoms (Dict[int, List[int]]): Mapping of local_res_id -> atom indices.
        target_res_id (int): The central local_res_id to be embedded.
        neighbor_res_ids (List[int]): Immediate neighbor local_res_ids to preserve connection environments.

    Returns:
        Tuple[Chem.RWMol, Dict[int, int]]:
            - fragment: A sanitized, mutable RDKit molecule ready for 3D embedding.
            - global_to_frag_map: Mapping of global atom index -> local fragment atom index.
    """
    fragment = Chem.RWMol()
    allowed_res_ids = [target_res_id] + neighbor_res_ids

    global_to_frag_map: Dict[int, int] = {}
    atom_count = 0
    frontier_atoms: Set[int] = set()

    # Step 1: Harvest and map all candidate atoms within the allowed residue neighborhood
    for r_id in allowed_res_ids:
        atom_ids = local2atoms.get(r_id)
        for a_id in atom_ids:
            atom = molecule.GetAtomWithIdx(a_id)
            # Add a copy of the atom into the new editable fragment
            frag_aid = fragment.AddAtom(atom)
            global_to_frag_map[a_id] = frag_aid
            atom_count += 1

    # Step 2: Extract internal bonds and identify cutting frontiers
    bonds_to_add = set()
    for g_id in global_to_frag_map:
        atom = molecule.GetAtomWithIdx(g_id)
        for bond in atom.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()

            # If the bond extends outside our allowed local neighborhood, it's a broken frontier
            if molecule_graph.nodes[u]['res_id'] not in allowed_res_ids or \
                    molecule_graph.nodes[v]['res_id'] not in allowed_res_ids:
                frontier_atoms.add(g_id)
                continue

            # Store unique internal bonds (ordered to avoid duplicate (u,v) vs (v,u))
            bonds_to_add.add((min(u, v), max(u, v), bond.GetBondType()))

    # Add the discovered internal bonds into the fragment
    for u, v, b_type in bonds_to_add:
        fragment.AddBond(global_to_frag_map[u], global_to_frag_map[v], b_type)

    # Step 3: Map stereochemistry and double-bond geometry safely to the fragment
    for u, v, _ in bonds_to_add:
        orig_bond = molecule.GetBondBetweenAtoms(u, v)
        frag_bond = fragment.GetBondBetweenAtoms(global_to_frag_map[u], global_to_frag_map[v])

        frag_bond.SetBondDir(orig_bond.GetBondDir())
        frag_bond.SetStereo(orig_bond.GetStereo())

        # If it is a geometric double bond (E/Z), correctly map the tracking stereo-marker atoms
        if orig_bond.GetStereo() in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE):
            orig_stereo_atoms = list(orig_bond.GetStereoAtoms())
            frag_bond.SetStereoAtoms(
                global_to_frag_map[orig_stereo_atoms[0]],
                global_to_frag_map[orig_stereo_atoms[1]]
            )

    # Step 4: Chemical Surgery - Repair broken conjugate/aromatic systems at the frontiers
    for g_id in frontier_atoms:
        f_id = global_to_frag_map[g_id]
        frontier_atom = fragment.GetAtomWithIdx(f_id)

        # Demote the frontier atom's aromatic status since its original ring system is truncated
        frontier_atom.SetIsAromatic(0)
        if frontier_atom.IsInRing() and molecule.GetAtomWithIdx(g_id).GetIsAromatic():
            # If the frontier atom was originally aromatic and is now truncated, degrade its ring status
            frontier_atom.SetIsAromatic(1)

        # Recursively stabilize the immediate neighbors of the frontier atom
        for nb_atom in frontier_atom.GetNeighbors():
            # If the neighbor is deeply embedded inside a complete ring, preserve its aromaticity
            if nb_atom.GetIsAromatic() and nb_atom.IsInRing():
                continue

            # Otherwise, degrade its aromaticity and force the modified linkage into a stable SINGLE bond
            nb_atom.SetIsAromatic(0)
            f_bond = fragment.GetBondBetweenAtoms(f_id, nb_atom.GetIdx())
            f_bond.SetBondType(Chem.rdchem.BondType.SINGLE)

    # Step 5: Final Sanitize and Validation check
    Chem.SanitizeMol(fragment, Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    sanitize_status = Chem.SanitizeMol(fragment, catchErrors=True)

    if sanitize_status is not Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
        logger.warning(
            f"Fragment sanitization flagged anomalies for residue {target_res_id}. "
            f"Status: {sanitize_status}"
        )
    fragment_h = Chem.AddHs(fragment)
    return fragment, global_to_frag_map

def _embed_with_chirality_check(
        fragment: Chem.RWMol,
        global_molecule: Chem.Mol,
        global_to_frag_map: Dict[int, int],
        target_atom_ids: List[int],
        target_res_id: int
) -> Chem.Conformer:
    """
    Sub-Workshop D1: Embeds a fragment using ETKDG and actively detects/corrects
    any inverted chiral centers caused by localized fragmentation.
    """
    fragment_h = AllChem.AddHs(fragment)

    # Track global reference chiral configurations for the target residue atoms
    chiral_reference = {}
    for g_id in target_atom_ids:
        f_id = global_to_frag_map[g_id]
        chiral_reference[f_id] = global_molecule.GetAtomWithIdx(g_id).GetChiralTag()
        fragment_h.GetAtomWithIdx(f_id).SetChiralTag(chiral_reference[f_id])

    # Set neighbor scaling atoms to unspecified to allow free flexible embedding rotation
    for atom in fragment_h.GetAtoms():
        if atom.GetIdx() not in chiral_reference:
            atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)

    # Initial ETKDG coordinates generation loop
    conf_id = -1
    attempts, max_attempts = 10000, 25000
    while conf_id == -1 and attempts <= max_attempts:
        conf_id = AllChem.EmbedMolecule(fragment_h, maxAttempts=attempts, useRandomCoords=False)
        attempts += 5000

    if conf_id == -1:
        raise ValueError(f"ETKDG failed to generate standard conformer for fragment of residue {target_res_id}")

    # Analyze post-embedding chiral signatures against global reference definitions
    ref_centers = dict(AllChem.FindMolChiralCenters(global_molecule, includeUnassigned=False))
    frag_centers = dict(AllChem.FindMolChiralCenters(fragment_h, includeUnassigned=False))

    chiral_flip_needed = False
    for g_id in target_atom_ids:
        f_id = global_to_frag_map[g_id]
        if f_id in frag_centers and g_id in ref_centers:
            # If RDKit inverted the mirror symmetry during embedding, trigger a tag inversion
            if frag_centers[f_id] != ref_centers[g_id]:
                chiral_flip_needed = True
                atom = fragment_h.GetAtomWithIdx(f_id)
                tag = atom.GetChiralTag()
                if tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
                    atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
                elif tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
                    atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)

    # If inversion occurred, re-embed using the compensated tetrahedral configurations (Negative * Negative = Positive)
    if chiral_flip_needed:
        logger.info(f"Chiral inversion detected in residue {target_res_id}. Re-embedding with adjusted tags.")
        AllChem.EmbedMolecule(fragment_h, maxAttempts=10000, useRandomCoords=False)

    # Perform local forcefield minimization to clean up structural bond lengths
    AllChem.MMFFOptimizeMolecule(fragment_h, maxIters=5000)
    return fragment_h.GetConformer()


def generate_local_fragment_coords(
        molecule: Chem.Mol,
        molecule_graph: nx.Graph,
        local2atoms: Dict[int, List[int]],
        target_res_id: int,
        neighbor_res_ids: List[int]
) -> Dict[int, np.ndarray]:
    """
    Sub-Workshop D2: Coordinates Generator. Extracts fragment, executes embedding,
    and harvests origin-centered (0,0,0) local coordinates for the central residue.
    """
    # 1. Delegate to Workshop C to safely clip the local subgraph
    fragment, global_to_frag_map = build_isolated_fragment(
        molecule, molecule_graph, local2atoms, target_res_id, neighbor_res_ids
    )

    # 2. Delegate to Sub-Workshop D1 to embed with accurate chirality controls
    target_atom_ids = local2atoms[target_res_id]
    frag_conf = _embed_with_chirality_check(
        fragment, molecule, global_to_frag_map, target_atom_ids, target_res_id
    )

    # 3. Harvest raw absolute coordinates belonging strictly to the central target residue
    local_coords: Dict[int, np.ndarray] = {}
    com = np.zeros(3)
    total_mass = 0.0

    for g_id in target_atom_ids:
        f_id = global_to_frag_map[g_id]
        pos = frag_conf.GetAtomPosition(f_id)
        pos_array = np.array([pos.x, pos.y, pos.z])

        # Accumulate Center of Mass properties
        mass = molecule.GetAtomWithIdx(g_id).GetMass()
        com += pos_array * mass
        total_mass += mass
        local_coords[g_id] = pos_array

    # 4. Zero-centering normalization pass: Force residue center of mass to (0, 0, 0)
    com_vector = com / total_mass
    for g_id in local_coords:
        local_coords[g_id] -= com_vector

    return local_coords


