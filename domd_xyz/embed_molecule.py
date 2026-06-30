from typing import Union, Any, Dict

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import tqdm
from domd_xyz.embed import embd, Meta, optimize_res_orientation
from misc.logger import logger
from scipy.stats import circmean
import numba as nb

@nb.jit(nopython=True)
def pbc(x,l):
    """Applies periodic boundary conditions to a coordinate or vector.

        Args:
            x (float or np.ndarray): Input coordinate.
            l (float or np.ndarray): Box length(s).

        Returns:
            float or np.ndarray: The wrapped coordinate within [-l/2, l/2].
    """
    return x - l*np.rint(x/l)

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
    comA = np.array([circmean(coords_A[:,i:i+1],low=-box[i]/2.,high=box[i]/2.,axis=0) for i in range(D)]).ravel()
    comB = np.array([circmean(coords_B[:,i:i+1],low=-box[i]/2.,high=box[i]/2.,axis=0) for i in range(D)]).ravel()
    cA = pbc(coords_A - comA,box)# coords_A- np.mean(coords_A, axis=0)
    cB = pbc(coords_B - comB,box) #coords_B - np.mean(coords_B, axis=0)

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
            rmsd = np.sqrt(np.mean(diff**2))
            
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_rotated_B = rotated_B
                best_R = R_candidate

    return best_rotated_B, best_R, best_rmsd, comA
def rotate_confs(pos,R,box,com_TP):
    """Applies a rotation and translation to a set of coordinates under PBC.

        Args:
            pos (np.ndarray): Input coordinates (N, 3).
            R (np.ndarray): Rotation matrix (3, 3).
            box (np.ndarray): Box dimensions.
            com_TP (np.ndarray): Target center of mass position (translation vector).

        Returns:
            np.ndarray: Transformed coordinates.
    """
    N,D = pos.shape
    com = np.array([circmean(pos[:,i:i+1],low=-box[i]/2.,high=box[i]/2.,axis=0) for i in range(D)]).ravel()
    cA = pbc(pos - com,box)
    r_cA = np.dot(cA, R.T)
    return r_cA + com_TP


def embed_rigid(molecule:Union[Chem.Mol, Chem.RWMol],
        cg_molecule: Union[nx.Graph, None] = None, rigid_aidxs: Dict[int, str] = None, box=None): 
    """Embeds a rigid molecule by aligning it to the Coarse-Grained (CG) bead positions.

        Takes an initial all-atom conformation, calculates the centers of the groups of atoms
        corresponding to each CG bead, and then rigidly rotates/translates the entire molecule
        to best match the target CG configuration.

        Args:
            molecule (Union[Chem.Mol, Chem.RWMol]): The all-atom molecule with an initial conformation.
            cg_molecule (Union[nx.Graph, None], optional): The coarse-grained graph containing target positions. Defaults to None.
            rigid_aidxs (Dict[int, str], optional): Mapping from CG bead index to list of atom indices. Defaults to None.
            box (np.ndarray, optional): Simulation box dimensions. Defaults to None.

        Returns:
            np.ndarray: The new positions of the atoms.
    """
    conf = molecule.GetConformer(0)
    aa_pos = conf.GetPositions()# * 0.1
    aa_rigid_pos = []
    for i in rigid_aidxs:
        aidxs = rigid_aidxs[i]
        aa_rigid_pos.append(np.mean(aa_pos[aidxs],axis=0))
    aa_rigid_pos = np.array(aa_rigid_pos)
    aa_cm = aa_pos.mean(axis=0)
    cg_rigid_pos = []
    for i in cg_molecule.nodes:
        cg_rigid_pos.append(cg_molecule.nodes[i]['x'])
    cg_rigid_pos = np.array(cg_rigid_pos)
    best_rotated_B, best_R, best_rmsd, com = get_best_alignment(cg_rigid_pos, aa_rigid_pos, box)
    #print(com)
    return pbc(rotate_confs(aa_pos,best_R, box, com),box)#(np.dot(aa_pos, best_R.T) + com)*10 # unit Angstrom


def embed_molecule(molecule: Union[Chem.Mol, Chem.RWMol],
                   cg_molecule: Union[nx.Graph, None] = None,
                   box=None, large=500, chunk_per_d=1, custom_confs=None, custom_fragment_confs = None) -> Any:
    """Generates 3D coordinates for a large molecule based on a Coarse-Grained template.

        This function performs a hierarchical embedding:
        1.  **Fragmentation**: Splits the molecule into residues based on `res_id`.
        2.  **Local Embedding**: Generates conformations for each residue (fragment).
        3.  **Orientation Optimization**: Rotates each residue to align bonded atoms with
            the vector connecting the corresponding CG beads.
        4.  **Assembly**: Translates residues to the CG bead positions.



        Args:
            molecule (Union[Chem.Mol, Chem.RWMol]): The all-atom topology (RDKit molecule).
            cg_molecule (Union[nx.Graph, None], optional): The CG graph with 'x' coordinates. Defaults to None.
            box (np.ndarray, optional): Box dimensions. Defaults to None.
            large (int, optional): Threshold for "large" molecule handling. Defaults to 500.
            chunk_per_d (int, optional): Optimization parameter for orientation. Defaults to 1.
            custom_confs (optional): Unused. Defaults to None.
            custom_fragment_confs (optional): Unused. Defaults to None.

        Returns:
            Chem.Conformer: The generated RDKit conformer with 3D coordinates.
    """
    _no_res_id = False
    _residue_map = {}
    ret = None
    for atom in molecule.GetAtoms():
        if 'res_id' not in atom.GetPropNames():
            _no_res_id = True
            logger.warning(f"No res_id property found in atom {atom.GetIdx()}, {atom.GetSymbol()} "
                           "the whole molecule is considered as one residue.")
    if _no_res_id:
        for atom in molecule.GetAtoms():
            atom.SetIntProp("res_id", 0)

    for atom in molecule.GetAtoms():
        if not _residue_map.get(atom.GetIntProp("res_id")):
            _residue_map[atom.GetIntProp("res_id")] = set()
        _residue_map[atom.GetIntProp("res_id")].add(atom.GetIdx())

    if _no_res_id:
        logger.warning("The entire molecule is considered "
                       "as one residue, for large molecules "
                       "this may cause problems.")
        if len(cg_molecule) > 1 and not cg_molecule.graph['is_rigid']:
            logger.error("The ref CG molecule contains more than 1"
                         " residues but the whole aa molecule is considered as one residue")
        return
    if cg_molecule.graph['is_rigid']:
        logger.info("Embedding rigid molecule")
        ret = Chem.Conformer()
        rigid_aidxs_map = cg_molecule.graph['rigid_aidxs_map']
        r_aa_pos = embed_rigid(molecule, cg_molecule, rigid_aidxs_map, box)
        for i,p in enumerate(r_aa_pos):
            ret.SetAtomPosition(i, p)
        return ret

    if cg_molecule is None:
        logger.warning("There is no ref CG info, so that the whole molecule is generated at once."
                       " This may be slow or fail for large molecule (e.g., >500)")

        conf_id = AllChem.EmbedMolecule(molecule, useRandomCoords=True)
        if conf_id == -1:
            logger.error("Configuration generation failed!")
        ret = molecule.GetConformer(conf_id)
    else:
        if len(_residue_map) != len(cg_molecule):
            logger.error("The number of residue in aa molecule is not equal to cg molecule "
                         f"{len(_residue_map)} != {len(cg_molecule)}")
            return
        #--#
        if molecule.GetNumAtoms()> 1000:
            logger.info('Embed molecule by using ETKDG ...')
        conf = embd(molecule, cg_molecule, large=large, custom_conf=None)
        #print(conf.GetPositions(),'***** after embd *****')
        #--#
        if molecule.GetNumAtoms()>1000:
            logger.info('Embed finished.')
        if conf is None:
            return
        # make residue centered at 0
        #for res_id in _residue_map:
        for res_id in tqdm.tqdm(_residue_map,total=len(_residue_map),desc='transition of cg cm',disable=True):
            atom_ids = _residue_map[res_id]
            com = np.zeros(3)
            sam = 0
            for atom_id in atom_ids:
                mas = molecule.GetAtomWithIdx(atom_id).GetMass()
                pos = conf.GetAtomPosition(atom_id)
                com += np.array([pos.x, pos.y, pos.z]) * mas
                sam += mas
            for atom_id in atom_ids:
                pos = conf.GetAtomPosition(atom_id)
                conf.SetAtomPosition(atom_id, np.array([pos.x, pos.y, pos.z]) - com / sam)
            # debug
            if logger.level <= 10:
                com = np.zeros(3)
                sam = 0
                for atom_id in atom_ids:
                    mas = molecule.GetAtomWithIdx(atom_id).GetMass()
                    pos = conf.GetAtomPosition(atom_id)
                    com += np.array([pos.x, pos.y, pos.z]) * mas
                    sam += mas
                logger.debug(f"The com of residue {res_id} is {com / sam}.")
        # debug
        if logger.level <= 10:
            for res_id in _residue_map:
                debug_xyz = ""
                debug_xyz += f"{len(_residue_map[res_id])}\n\n"
                for atom_id in _residue_map[res_id]:
                    p = conf.GetAtomPosition(atom_id)
                    atom = molecule.GetAtomWithIdx(atom_id)
                    debug_xyz += f"{atom.GetSymbol()} {p.x} {p.y} {p.z}\n"
                logger.debug(f"structure of res {res_id}:\n{debug_xyz}")

        atom_pos = conf.GetPositions()
        atom_res_id = np.array([a.GetIntProp("res_id") for a in molecule.GetAtoms()])
        n_residue = len(_residue_map)
        bonds = []
        trans = np.zeros((n_residue, 3))
        local_frame_idx = []
        for bond in molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            atom_i = molecule.GetAtomWithIdx(i)
            atom_j = molecule.GetAtomWithIdx(j)
            res_id_i = atom_i.GetIntProp("res_id")
            res_id_j = atom_j.GetIntProp("res_id")
            # print(res_id_i,i,'*',res_id_j,j)
            if res_id_i != res_id_j:
                bonds.append((res_id_i, res_id_j))
                local_frame_idx.append((i, j))
                # print(i,j)
        for res_id in _residue_map:
            for node in cg_molecule.nodes:
                if cg_molecule.nodes[node].get('local_res_id') == res_id:
                    trans[res_id] = cg_molecule.nodes[node].get("x")
        if box is None:
            box = np.ones(3) * abs(conf.GetPositions().max()) * 100.0
            logger.info(f"Box is not given. Set to {box} to eliminate pbc.")
        meta = Meta(np.array(bonds, dtype=np.int64),
                    trans,
                    np.array(local_frame_idx, dtype=np.int64),
                    atom_pos,
                    atom_res_id,
                    box)
        # pickle.dump(meta,open('meta.pkl','wb'))
        #--#
        if molecule.GetNumAtoms() > 200:
            logger.info("Optimizing orientations...")
        rot = optimize_res_orientation(n_residue, meta, chunk_per_d=chunk_per_d)
        # rot = np.asarray([np.eye(3),] * n_residue)
        #--#
        if molecule.GetNumAtoms() > 200:
            logger.info(f"Optimize finished.")
        # debug
        for ir, r in enumerate(rot):
            logger.debug(f"Rotation for residue {ir} is {r} and r.T.dot(r) is {r.T.dot(r)}")
        for res_id in _residue_map:
            atoms = _residue_map[res_id]
            for atom_id in atoms:
                p = rot[res_id].dot(atom_pos[atom_id])
                # p = conf.GetAtomPosition(atom_id)
                conf.SetAtomPosition(atom_id, p + trans[res_id])
        #print(conf.GetPositions(),'**** after optimization ****')
        ret = Chem.Conformer()
        for atom in molecule.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            p = np.array([pos.x, pos.y, pos.z])
            ret.SetAtomPosition(atom.GetIdx(), p)
        #print(ret.GetPositions(), '**** man make conf ****')
    #print(ret.GetPositions())
    return ret

